################################################################
# Generalisation of Geometric Vector Perceptron, Jing et al.
# for explicit multi-state biomolecule representation learning.
# Original repository: https://github.com/drorlab/gvp-pytorch
################################################################

from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch_geometric

from src.layers import *
from src.sampling import choose_nts

import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.constants import NUM_TO_LETTER


# TODO: change this later !
tokenizer = AutoTokenizer.from_pretrained("buetnlpbio/birna-tokenizer")
bert_config = transformers.BertConfig.from_pretrained("buetnlpbio/birna-bert")
#config.alibi_starting_size = 5000 # maximum sequence length updated to 5000 from config default of 1024 -- IF INCREASED TO 5,00 IT KILLS PROCESS !
mysterybert = AutoModelForMaskedLM.from_pretrained("buetnlpbio/birna-bert",config=bert_config,trust_remote_code=True)
mysterybert.cls = torch.nn.Identity()

MAX_TOKENS = 1022
k = 1 #tradeoff
ONLY_BPE = False
ONLY_NUC = False

import torch

def dynamic_tokenize_preprocessing(tensor_seq):
    """
    Process a batch of sequences.
    Args:
        seqs (torch.Tensor): Tensor of sequences to process.
    Returns:
        torch.Tensor: Processed sequences tensor.
    """
    
    # Process each sequence in the batch
    letter_seq = "".join([NUM_TO_LETTER[int(token)] for token in tensor_seq])

    if ONLY_NUC:
        return " ".join(letter_seq)
    elif ONLY_BPE:
        return letter_seq
    
    if len(letter_seq) < k*MAX_TOKENS:
        letter_seq = " ".join(letter_seq[:MAX_TOKENS])
    print(letter_seq)
    # Note: This assumes all processed sequences have compatible shapes
    return letter_seq


class AutoregressiveMultiGNNv1(torch.nn.Module):
    '''
    Autoregressive GVP-GNN for **multiple** structure-conditioned RNA design.
    
    Takes in RNA structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 4 bases at each position in a `torch.Tensor` of shape [n_nodes, 4].
    
    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.

    Args:
        node_in_dim (tuple): node dimensions in input graph
        node_h_dim (tuple): node dimensions to use in GVP-GNN layers
        node_in_dim (tuple): edge dimensions in input graph
        edge_h_dim (tuple): edge dimensions to embed in GVP-GNN layers
        num_layers (int): number of GVP-GNN layers in encoder/decoder
        drop_rate (float): rate to use in all dropout layers
        out_dim (int): output dimension (4 bases)
    '''
    def __init__(
        self,
        node_in_dim = (64, 4), 
        node_h_dim = (128, 16), 
        edge_in_dim = (32, 1), 
        edge_h_dim = (32, 1),
        num_layers = 3, 
        drop_rate = 0.1,
        out_dim = 4,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        activations = (F.silu, None)
        
        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, 
                activations=(None, None), vector_gate=True)
        )
        
        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
                MultiGVPConvLayer(self.node_h_dim, self.edge_h_dim, 
                                  activations=activations, vector_gate=True,
                                  drop_rate=drop_rate, norm_first=True)
            for _ in range(num_layers))
        
        # Decoder layers
        self.W_s = nn.Embedding(self.out_dim, self.out_dim)
        self.edge_h_dim = (self.edge_h_dim[0] + self.out_dim, self.edge_h_dim[1])
        self.decoder_layers = nn.ModuleList(
                GVPConvLayer(self.node_h_dim, self.edge_h_dim,
                             activations=activations, vector_gate=True, 
                             drop_rate=drop_rate, autoregressive=True, norm_first=True) 
            for _ in range(num_layers))
        
        # Output
        self.W_out = GVP(self.node_h_dim, (self.out_dim, 0), activations=(None, None))
    
    def forward(self, batch):

        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        seq = batch.seq

        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        # Pool multi-conformation features: 
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)

        encoder_embeddings = h_V
        
        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
        
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x = encoder_embeddings)
        
        logits = self.W_out(h_V)
        
        return logits
    
    @torch.no_grad()
    def sample(
            self, 
            batch, 
            n_samples, 
            temperature: Optional[float] = 0.1, 
            logit_bias: Optional[torch.Tensor] = None,
            return_logits: Optional[bool] = False,
            sampling_strategy: Optional[str] = "categorical",
            top_k: Optional[int] = 0,
            top_p: Optional[float] = 0.0,
            min_p: Optional[float] = 0.0,
        ):
        '''
        Samples sequences autoregressively from the distribution
        learned by the model.

        Args:
            batch (torch_geometric.data.Data): mini-batch containing one
                RNA backbone to design sequences for
            n_samples (int): number of samples
            temperature (float): temperature to use in softmax over 
                the categorical distribution
            logit_bias (torch.Tensor): bias to add to logits during sampling
                to manually fix or control nucleotides in designed sequences,
                of shape [n_nodes, 4]
            return_logits (bool): whether to return logits or 
            sampling_strategy (str): one of "categorical", "greedy", "top_k", "top_p", "min_p", etc.
            top_k (int): if using top-k sampling, how many tokens to keep
            top_p (float): if using nucleus (top-p) sampling, what cumulative probability threshold
            min_p (float): if using min-p sampling, probability threshold w.r.t max prob
        
        Returns:
            seq (torch.Tensor): int tensor of shape [n_samples, n_nodes]
                                based on the residue-to-int mapping of
                                the original training data
            logits (torch.Tensor): logits of shape [n_samples, n_nodes, 4]
                                   (only if return_logits is True)
        ''' 
        print('Batch size: ', batch.size())
        h_V = (batch.node_s, batch.node_v)
        print('node_s size:', batch.node_s.size())
        print('node_v size:', batch.node_v.size())
        h_E = (batch.edge_s, batch.edge_v)
        print('edge_s size:', batch.edge_s.size())
        print('edge_v size:', batch.edge_v.size())
        edge_index = batch.edge_index
        print('edge_index size:', batch.edge_index.size())
    
        device = edge_index.device
        num_nodes = h_V[0].shape[0]
        print('Number of nodes: ', num_nodes)
        
        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        
        # Pool multi-conformation features
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        
        # Repeat features for sampling n_samples times
        h_V = (h_V[0].repeat(n_samples, 1),
            h_V[1].repeat(n_samples, 1, 1))
        h_E = (h_E[0].repeat(n_samples, 1),
            h_E[1].repeat(n_samples, 1, 1))
        
        # Expand edge index for autoregressive decoding
        edge_index = edge_index.expand(n_samples, -1, -1)
        offset = num_nodes * torch.arange(n_samples, device=device).view(-1, 1, 1)
        edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
        # This is akin to 'batching' (in PyG style) n_samples copies of the graph
        
        seq = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.int)
        h_S = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)
        logits = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)
        print('Size logits in models_mod: ', logits.size())

        h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]

        # Decode one token at a time
        for i in range(num_nodes):
            print('Iteration : ', i)
            h_S_ = h_S[edge_index[0]]
            h_S_[edge_index[0] >= edge_index[1]] = 0
            print('new h_S_:', h_S_)
            print('size he0:', h_E[0].size())
            print('size he1:', h_E[1].size())
            print('size hs:', h_S.size())
            h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
            print('h_E_:', h_E_)
                    
            edge_mask = edge_index[1] % num_nodes == i  # True for all edges where dst is node i
            edge_index_ = edge_index[:, edge_mask]  # subset all incoming edges to node i
            h_E_ = tuple_index(h_E_, edge_mask)
            node_mask = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.bool)
            node_mask[i::num_nodes] = True  # True for all nodes i and its repeats
            
            for j, layer in enumerate(self.decoder_layers):
                out = layer(h_V_cache[j], edge_index_, h_E_,
                        autoregressive_x=h_V_cache[0], node_mask=node_mask)
                
                out = tuple_index(out, node_mask)  # subset out to only node i and its repeats
                
                if j < len(self.decoder_layers)-1:
                    h_V_cache[j+1][0][i::num_nodes] = out[0]
                    h_V_cache[j+1][1][i::num_nodes] = out[1]
                
            lgts = self.W_out(out)
            # Add logit bias if provided to fix or bias positions
            if logit_bias is not None:
                lgts += logit_bias[i]
            # Sample from logits
            seq[i::num_nodes], _ = choose_nts(lgts, strategy=sampling_strategy, 
                                    temperature=temperature, beam_branch=1, top_k=top_k, top_p=top_p, min_p=min_p)
            h_S[i::num_nodes] = self.W_s(seq[i::num_nodes])
            logits[i::num_nodes] = lgts
            print('Sequence models_mod size after modifications:', seq.size())
            print('Final sequence: ', seq)

        if return_logits:
            return seq.view(n_samples, num_nodes), logits.view(n_samples, num_nodes, self.out_dim)
        else:    
            return seq.view(n_samples, num_nodes)
        
    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):

        if mask_confs.size(1) == 1:
            # Number of conformations is 1, no need to pool
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])
        
        # True num_conf for masked mean pooling
        n_conf_true = mask_confs.sum(1, keepdim=True)  # (n_nodes, 1)
        
        # Mask scalar features
        mask = mask_confs.unsqueeze(2)  # (n_nodes, n_conf, 1)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]

        # Mask vector features
        mask = mask.unsqueeze(3)  # (n_nodes, n_conf, 1, 1)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]
        
        # Average pooling multi-conformation features
        h_V = (h_V0.sum(dim=1) / n_conf_true,               # (n_nodes, d_s)
               h_V1.sum(dim=1) / n_conf_true.unsqueeze(2))  # (n_nodes, d_v, 3)
        h_E = (h_E0.sum(dim=1) / n_conf_true[edge_index[0]],               # (n_edges, d_se)
               h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2))  # (n_edges, d_ve, 3)

        return h_V, h_E

    @torch.no_grad()
    def beam_search(
            self, 
            batch, 
            beam_width: int = 16,
            #beam_depth: int = 10,
            temperature: float = 1.0,
            logit_bias: Optional[torch.Tensor] = None,
            return_logits: bool = False
        ):
        '''
        Performs beam search for sequence generation.
        
        Args:
            batch: Input batch containing RNA structure
            beam_width: Number of beams to maintain
            temperature: Temperature for softmax
            logit_bias: Optional bias for logits
            return_scores: Whether to return beam scores
        
        Returns:
            best_sequences: Top sequences found by beam search
            scores: (Optional) Scores for each sequence
        '''
        edge_index = batch.edge_index
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)

        device = edge_index.device
        num_nodes = h_V[0].shape[0]
        
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        
        # Pool multi-conformation features
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        
        # Create initial hidden states for beams
        #h_V = (h_V[0].repeat(beam_width, 1),
        #    h_V[1].repeat(beam_width, 1, 1))
        #h_E = (h_E[0].repeat(beam_width, 1),
        #    h_E[1].repeat(beam_width, 1, 1))
        
        # Cache for decoder layers
        #edge_index = edge_index.expand(beam_width, -1, -1)
        #print(edge_index.size())
        #offset = num_nodes * torch.arange(beam_width, device=device).view(-1, 1, 1)
        #edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
        
        # Initialize tensors for the sequence, decoder state, and logits.
        # For beam search, each beam candidate holds its own state.
        beams = []
        for b in range(beam_width):
            beams.append({
                "score": 0.0,  # cumulative log-probability
                "seq": torch.zeros(num_nodes, dtype=torch.int, device=device),  # decoded tokens (to be filled)
                "h_S": torch.zeros(num_nodes, self.out_dim, device=device),
                # Each decoder layer keeps its own cache (here cloned from the pooled encoder features)
                "h_V_cache": [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers],
                # Optionally, you can store logits for later inspection
                "logits": torch.zeros(num_nodes, self.out_dim, device=device),
            })

        # === Autoregressive Decoding with Beam Search ===
        for i in range(num_nodes):
            new_beams = []
            # Process each beam candidate separately
            for beam in beams:
                # Prepare local copies of the decoder state for beam expansion
                h_S = beam["h_S"]
                #h_V_cache = [ (cache0.clone(), cache1.clone()) for cache0, cache1 in beam["h_V_cache"] ]
                h_V_cache = beam['h_V_cache']
                seq = beam["seq"]

                # --- Prepare messages for decoding token at position i ---
                # In the original sample(), h_S is used via indexing with edge_index.
                # Here we prepare the subset h_S_ corresponding to incoming edges for node i.
                h_S_ = h_S[edge_index[0]]
                # Zero out contributions from nodes not yet decoded:
                h_S_[edge_index[0] >= edge_index[1]] = 0

                # Concatenate h_S_ with edge features (as in sample code)
                h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
                # Select only the incoming edges for node i:
                edge_mask = edge_index[1] % num_nodes == i
                edge_index_i = edge_index[:, edge_mask]
                h_E_i = tuple_index(h_E_, edge_mask)  # Assume tuple_index is a helper function

                # Create a mask that is True only for the current node i (across all copies in the beam)
                node_mask = torch.zeros(num_nodes, device=device, dtype=torch.bool)
                node_mask[i::num_nodes] = True

                # --- Pass through decoder layers ---
                # We simulate the same decoder forward pass as in sample(), updating the cache.
                print('num_node:', i)
                for j, layer in enumerate(self.decoder_layers):
                    out = layer(h_V_cache[j], edge_index_i, h_E_i,
                                autoregressive_x=h_V_cache[0],
                                node_mask=node_mask)
                    # next line added
                    out = tuple_index(out, node_mask)  # subset out to only node i and its repeats
                    # Update the cache for the next layer if needed
                    if j < len(self.decoder_layers) - 1:
                        #cache0, cache1 = h_V_cache[j+1]
                        # Update only the i-th node (and its beam copies)
                        h_V_cache[j+1][0][i::num_nodes] = out[0]
                        h_V_cache[j+1][1][i::num_nodes] = out[1]
                # Final logits for node i:
                lgts = self.W_out(out)

                # Optionally add logit bias
                if logit_bias is not None:
                    lgts += logit_bias[i]

                # Compute log probabilities with temperature
                #log_probs = torch.log_softmax(lgts / temperature, dim=-1)

                # --- Beam Expansion: Consider each token candidate ---
                # For beam search, we get the top-k tokens (k=beam_width) for the current node.
                #top_log_probs, top_tokens = torch.topk(log_probs, k=beam_width, dim=-1)
                top_tokens, top_log_probs = choose_nts(lgts, strategy='min_p', 
                                   temperature=1.0, beam_branch=10, top_k=2, top_p=0.9, min_p=0.05)

                print('top_log_probs', top_log_probs)
                print('top_tokens', top_tokens)
                # For each candidate token, create a new beam candidate.
                # (Assume that each beam is repeated once here; for beam_width=2, each beam yields 2 new beams.)
                for token_idx in top_tokens[0]:
                    new_beam = {
                        "score": beam["score"] + top_log_probs[0][token_idx], # weird [0] indexing
                        "seq": seq.clone(),
                        "h_S": h_S.clone(),
                        "h_V_cache": [ (cache0.clone(), cache1.clone()) for cache0, cache1 in h_V_cache ],
                        "logits": beam["logits"].clone(),
                    }
                    print('token_idx:', token_idx)
                    # Update the decoded token at position i:
                    # wrong -- it is selecting two times the top tokens
                    new_beam["seq"][i::num_nodes] = token_idx # weird [0] indexing
                    print('new_beam seq:', new_beam['seq'])
                    print('new_beam score:', new_beam['score'])
                    # Update the decoder state for the current node:
                    new_beam["h_S"][i::num_nodes] = self.W_s(new_beam['seq'][i::num_nodes]) # weird [0] indexing
                    new_beam["logits"][i::num_nodes] = lgts  # store the logits for analysis
                    #print('new_beam h_S:',new_beam["h_S"])
                    new_beams.append(new_beam)

            # After expanding all beams, select the top `beam_width` beams based on cumulative score.
            new_beams.sort(key=lambda x: x["score"], reverse=True)
            #print('new_beams:', new_beams)
            beams = new_beams[:beam_width]

        # --- After decoding all nodes, select the best scoring beam ---
        best_beam = max(beams, key=lambda x: x["score"])
        best_seq = best_beam["seq"]
        print('best_beam:', best_beam)
        if return_logits:
            return best_seq, best_beam["logits"].view(num_nodes, self.out_dim)
        else:
            return best_seq # kind of works except for output on the other function -- several things are wrong though
        
    @torch.no_grad()
    def sample_mod(
            self, 
            batch, 
            n_samples, 
            temperature: Optional[float] = 0.1, 
            logit_bias: Optional[torch.Tensor] = None,
            return_logits: Optional[bool] = False,
            beam_width: Optional[int] = 2,
            beam_branch: Optional[int] = 2,
            sampling_strategy: Optional[str] = "categorical",
            top_k: Optional[int] = 0,
            top_p: Optional[float] = 0.0,
            min_p: Optional[float] = 0.0,
        ):
        '''
        Samples sequences autoregressively from the distribution
        learned by the model.

        Args:
            batch (torch_geometric.data.Data): mini-batch containing one
                RNA backbone to design sequences for
            n_samples (int): number of samples
            temperature (float): temperature to use in softmax over 
                the categorical distribution
            logit_bias (torch.Tensor): bias to add to logits during sampling
                to manually fix or control nucleotides in designed sequences,
                of shape [n_nodes, 4]
            return_logits (bool): whether to return logits or 
            sampling_strategy (str): one of "categorical", "greedy", "top_k", "top_p", "min_p", etc.
            top_k (int): if using top-k sampling, how many tokens to keep
            top_p (float): if using nucleus (top-p) sampling, what cumulative probability threshold
            min_p (float): if using min-p sampling, probability threshold w.r.t max prob
            beam_width (int): number of beams to maintain during search
            beam_branch (int): number of samples to get from sampling strategy
        Returns:
            seq (torch.Tensor): int tensor of shape [n_samples, n_nodes]
                                based on the residue-to-int mapping of
                                the original training data
            logits (torch.Tensor): logits of shape [n_samples, n_nodes, 4]
                                   (only if return_logits is True)
        ''' 
        print('Batch size: ', batch.size())
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
    
        device = edge_index.device
        num_nodes = h_V[0].shape[0]
        print('Number of nodes: ', num_nodes)
        
        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        
        # Pool multi-conformation features
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        
        # Repeat features for sampling n_samples times
        h_V = (h_V[0].repeat(n_samples, 1),
            h_V[1].repeat(n_samples, 1, 1))
        h_E = (h_E[0].repeat(n_samples, 1),
            h_E[1].repeat(n_samples, 1, 1))
        
        # Expand edge index for autoregressive decoding
        edge_index = edge_index.expand(n_samples, -1, -1)
        offset = num_nodes * torch.arange(n_samples, device=device).view(-1, 1, 1)
        edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
        # This is akin to 'batching' (in PyG style) n_samples copies of the graph
        
        beams = []
        for b in range(beam_width):
            beams.append({
                "score": torch.zeros(n_samples, dtype=torch.float, device=device),  # cumulative log-probability
                "seq": torch.zeros(num_nodes*n_samples, dtype=torch.int, device=device),  # decoded tokens (to be filled)
                "h_S": torch.zeros(num_nodes*n_samples, self.out_dim, device=device),
                # Each decoder layer keeps its own cache (here cloned from the pooled encoder features)
                "h_V_cache": [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers],
                # Optionally, you can store logits for later inspection
                "logits": torch.zeros(num_nodes*n_samples, self.out_dim, device=device),
            })

        # Decode one token at a time
        for i in range(num_nodes):
            new_beams = []
            print('Iteration : ', i)

            for beam in beams:
                # Prepare local copies of the decoder state for beam expansion
                h_S = beam["h_S"]
                h_V_cache = beam['h_V_cache']
                seq = beam["seq"]

                # --- Prepare messages for decoding token at position i ---
                # In the original sample(), h_S is used via indexing with edge_index.
                # Here we prepare the subset h_S_ corresponding to incoming edges for node i.
                h_S_ = h_S[edge_index[0]]
                # Zero out contributions from nodes not yet decoded:
                h_S_[edge_index[0] >= edge_index[1]] = 0

                # Concatenate h_S_ with edge features
                h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
                # Select only the incoming edges for node i:
                edge_mask = edge_index[1] % num_nodes == i  # True for all edges where dst is node i
                edge_index_ = edge_index[:, edge_mask]  # subset all incoming edges to node i
                h_E_ = tuple_index(h_E_, edge_mask)

                # Create a mask that is True only for the current node i (across all copies in the beam)
                node_mask = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.bool)
                node_mask[i::num_nodes] = True  # True for all nodes i and its repeats

                # --- Pass through decoder layers ---
                # We simulate the same decoder forward pass as in sample(), updating the cache.
                for j, layer in enumerate(self.decoder_layers):
                    out = layer(h_V_cache[j], edge_index_, h_E_,
                            autoregressive_x=h_V_cache[0], node_mask=node_mask)
                    out = tuple_index(out, node_mask)  # subset out to only node i and its repeats
                    # Update the cache for the next layer if needed
                    if j < len(self.decoder_layers)-1:
                        h_V_cache[j+1][0][i::num_nodes] = out[0]
                        h_V_cache[j+1][1][i::num_nodes] = out[1]
                # Final logits for node i:
                lgts = self.W_out(out)

                # Add logit bias if provided to fix or bias positions
                if logit_bias is not None:
                    lgts += logit_bias[i]
                # Sample from logits
                # ADD HERE THE BRANCHING FACTOR !!
                top_tokens, log_probs = choose_nts(lgts, strategy=sampling_strategy, 
                                    temperature=temperature, beam_branch=beam_branch, top_k=top_k, top_p=top_p, min_p=min_p)
                # log probs will return probabilities for each nucleotide type
                # top tokens will return beam_branch samples of tokens for each of the sequences in n_samples
                print('top_log_probs', log_probs)
                print('top_tokens', top_tokens)
                print('size top_tokens:', top_tokens.size())

                # For each candidate token, create a new beam candidate.
                new_beam_scores = torch.zeros(beam_branch, n_samples, dtype=torch.float, device=device)

                for idx in range(beam_branch):
                    print('top tokens now:', top_tokens[:,idx])
                    print('log probs chosen: ', log_probs.gather(dim=1, index=top_tokens[:,idx].reshape(-1, 1)))
                    #print(beam["score"] + log_probs[:,top_tokens[:, idx]])
                    top_tokens_beam = top_tokens[:, idx].reshape(-1, 1)
                    top_log_probs_beam = log_probs.gather(dim=1, index=top_tokens_beam).reshape(1, -1)
                    new_beam = {
                        "score": beam["score"] + top_log_probs_beam,
                        "seq": seq.clone(),
                        "h_S": h_S.clone(),
                        "h_V_cache": [ (cache0.clone(), cache1.clone()) for cache0, cache1 in h_V_cache ],
                        "logits": beam["logits"].clone(),
                    }
                    print('token_idx:', top_tokens[:,idx])
                    # Update the decoded token at position i:
                    # wrong -- it is selecting two times the top tokens
                    new_beam["seq"][i::num_nodes] = top_tokens[:,idx]
                    print('new_beam seq:', new_beam['seq'])
                    #print('new_beam seq length:', len(new_beam['seq']))
                    print('new_beam score:', new_beam['score'])
                    # Update the decoder state for the current node:
                    new_beam["h_S"][i::num_nodes] = self.W_s(new_beam['seq'][i::num_nodes]) # weird [0] indexing
                    new_beam["logits"][i::num_nodes] = lgts  # store the logits for analysis
                    #print('new_beam h_S:',new_beam["h_S"])
                    #print('new_beam:', new_beam)
                    new_beams.append(new_beam)
                    new_beam_scores[idx] = new_beam['score']

            # After expanding all beams, select the top `beam_width` beams based on cumulative score.
            scores_tensor = torch.cat([beam["score"] for beam in new_beams], dim=0)
            print('scores tensor:', scores_tensor)
            sorted_scores, sorted_indices = torch.sort(scores_tensor, dim=0, descending=True)
            print('sorted_scores:', sorted_scores)
            print('sorted_indices:', sorted_indices)
            #print('new_beams:', new_beams)
            # this is where it fails ! can't sort the list !!
            new_beams.sort(key=lambda x: x['score'], reverse=True)
            #print('new_beams:', new_beams)
            beams = new_beams[:beam_width]

        # --- After decoding all nodes, select the best scoring beam ---
        best_beam = max(beams, key=lambda x: x["score"])
        seq = best_beam["seq"]
        logits = best_beam["logits"]
        print('best_beam:', best_beam)
        print('size logits:', logits.size())

        if return_logits:
            return seq.view(n_samples, num_nodes), logits.view(n_samples, num_nodes, self.out_dim)
        else:    
            return seq.view(n_samples, num_nodes)
        
    @torch.no_grad()
    def sample_mod2(
            self, 
            batch, 
            n_samples, 
            temperature: Optional[float] = 0.1, 
            logit_bias: Optional[torch.Tensor] = None,
            return_logits: Optional[bool] = False,
            beam_width: Optional[int] = 2,
            beam_branch: Optional[int] = 2,
            sampling_strategy: Optional[str] = "categorical",
            top_k: Optional[int] = 0,
            top_p: Optional[float] = 0.0,
            min_p: Optional[float] = 0.0,
        ):
        '''
        Samples sequences autoregressively from the distribution
        learned by the model.

        Args:
            batch (torch_geometric.data.Data): mini-batch containing one
                RNA backbone to design sequences for
            n_samples (int): number of samples
            temperature (float): temperature to use in softmax over 
                the categorical distribution
            logit_bias (torch.Tensor): bias to add to logits during sampling
                to manually fix or control nucleotides in designed sequences,
                of shape [n_nodes, 4]
            return_logits (bool): whether to return logits or 
            sampling_strategy (str): one of "categorical", "greedy", "top_k", "top_p", "min_p", etc.
            top_k (int): if using top-k sampling, how many tokens to keep
            top_p (float): if using nucleus (top-p) sampling, what cumulative probability threshold
            min_p (float): if using min-p sampling, probability threshold w.r.t max prob
            beam_width (int): number of beams to maintain during search
            beam_branch (int): number of samples to get from sampling strategy
        Returns:
            seq (torch.Tensor): int tensor of shape [n_samples, n_nodes]
                                based on the residue-to-int mapping of
                                the original training data
            logits (torch.Tensor): logits of shape [n_samples, n_nodes, 4]
                                   (only if return_logits is True)
        ''' 
        print('Batch size: ', batch.size())
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
    
        device = edge_index.device
        num_nodes = h_V[0].shape[0]
        print('Number of nodes: ', num_nodes)
        
        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        
        # Pool multi-conformation features
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        
        # Repeat features for sampling n_samples times
        # might have to change this
        h_V = (h_V[0].repeat(beam_width*n_samples, 1),
            h_V[1].repeat(beam_width*n_samples, 1, 1))
        h_E = (h_E[0].repeat(beam_width*n_samples, 1),
            h_E[1].repeat(beam_width*n_samples, 1, 1))

        # Expand edge index for autoregressive decoding
        edge_index = edge_index.expand(beam_width*n_samples, -1, -1)
        print('edge_index:', edge_index)
        offset = num_nodes * torch.arange(beam_width*n_samples, device=device).view(beam_width*n_samples, 1, 1)
        print('offset size:', offset.size())
        print('offset:', offset)
        edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
        # This is akin to 'batching' (in PyG style) n_samples copies of the graph
        
        scores = torch.zeros(beam_width*n_samples, dtype=torch.float, device=device)  # cumulative log-probability
        seq = torch.zeros(beam_width*num_nodes*n_samples, dtype=torch.int, device=device)  # decoded tokens (to be filled)
        h_S = torch.zeros(beam_width*num_nodes*n_samples, self.out_dim, device=device)
        # Each decoder layer keeps its own cache (here cloned from the pooled encoder features)
        h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]
        # Optionally, you can store logits for later inspection
        logits = torch.zeros(beam_width*num_nodes*n_samples, self.out_dim, device=device)

        # Decode one token at a time
        for i in range(num_nodes):
            print('Iteration : ', i)

            # --- Prepare messages for decoding token at position i ---
            # In the original sample(), h_S is used via indexing with edge_index.
            # Here we prepare the subset h_S_ corresponding to incoming edges for node i.
            h_S_ = h_S[edge_index[0]]
            # Zero out contributions from nodes not yet decoded:
            h_S_[edge_index[0] >= edge_index[1]] = 0

            # Concatenate h_S_ with edge features
            h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
            # Select only the incoming edges for node i:
            edge_mask = edge_index[1] % num_nodes == i  # True for all edges where dst is node i
            edge_index_ = edge_index[:, edge_mask]  # subset all incoming edges to node i
            h_E_ = tuple_index(h_E_, edge_mask)

            # Create a mask that is True only for the current node i (across all copies in the beam)
            node_mask = torch.zeros(beam_width*n_samples*num_nodes, device=device, dtype=torch.bool)
            node_mask[i::num_nodes] = True  # True for all nodes i and its repeats
            # not entirely sure if the thing above is correct

            # --- Pass through decoder layers ---
            # We simulate the same decoder forward pass as in sample(), updating the cache.
            for j, layer in enumerate(self.decoder_layers):
                out = layer(h_V_cache[j], edge_index_, h_E_,
                        autoregressive_x=h_V_cache[0], node_mask=node_mask)
                out = tuple_index(out, node_mask)  # subset out to only node i and its repeats
                # Update the cache for the next layer if needed
                if j < len(self.decoder_layers)-1:
                    h_V_cache[j+1][0][i::num_nodes] = out[0]
                    h_V_cache[j+1][1][i::num_nodes] = out[1]
            # Final logits for node i:
            lgts = self.W_out(out)

            # Add logit bias if provided to fix or bias positions
            if logit_bias is not None:
                lgts += logit_bias[i]
            # Sample from logits
            # ADD HERE THE BRANCHING FACTOR !!
            top_tokens, log_probs = choose_nts(lgts, strategy=sampling_strategy, beam_branch=beam_branch,
                                    temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)
            # log probs will return probabilities for each nucleotide type
            # top tokens will return beam_branch samples of tokens for each of the sequences in n_samples
            print('top_log_probs', log_probs.size())
            print('top_tokens', top_tokens)
            print('size top_tokens:', top_tokens.size())

            # For each candidate token, create a new beam candidate.
            #new_beam_scores = torch.zeros(beam_branch, beam_width*n_samples, dtype=torch.float, device=device)
            #new_beam_seq = torch.zeros(beam_width*num_nodes*n_samples, dtype=torch.int, device=device)
            #new_beam_h_S = torch.zeros(beam_width*num_nodes*n_samples, dtype=torch.int, device=device)
            new_beam_seq = seq.clone().repeat(beam_branch, 1)
            print('size of new_beam_seq:', new_beam_seq.size())
            new_beam_h_S = h_S.clone().repeat(beam_branch, 1, 1) ## ADD STH HERE !!!!!
            print('size of new_beam_h_S:', new_beam_h_S.size())
            #new_beam_h_V_cache = [(h_V[0].clone().repeat(beam_branch, 1, 1), h_V[1].clone().repeat(beam_branch, 1, 1)) for _ in self.decoder_layers]
            #new_beam_logits = torch.zeros(beam_width*num_nodes*n_samples, self.out_dim, device=device)
            new_beam_logits = logits.clone().repeat(beam_branch, 1, 1)
            print('size of new_beam_logits:', new_beam_logits.size())

            print('size log probs:', log_probs.size())
            print('top tokens size:', top_tokens.size())
            top_log_probs_beam = log_probs.gather(dim=1, index=top_tokens)
            top_log_probs_beam = top_log_probs_beam.transpose(0, 1)
            
            print('top_log_probs_beam.size:', top_log_probs_beam.size())
            print(top_log_probs_beam)
            #print('new_beam_scores size:', new_beam_scores.size())
            
            print('scores:', scores)
            #print('new_beam_scores[:, i::num_nodes]', new_beam_scores[:, i::num_nodes])
            new_beam_scores = scores.repeat(beam_branch, 1) + top_log_probs_beam
            #new_beam_scores[:,i::num_nodes] = scores.repeat(beam_branch, 1)[:,i::num_nodes] + top_log_probs_beam
            print('new_beam_scores:', new_beam_scores)
            #print(new_beam_scores.size())

            new_beam_seq[:,i::num_nodes] = top_tokens.transpose(0,1)
            print('new_beam seq:', new_beam_seq[:,i::num_nodes])
            #print('new_beam score:', new_beam_scores[:,i::num_nodes])
            new_beam_logits[:,i::num_nodes] = lgts  # store the logits for analysis
            print('new_beam_logits size:', new_beam_logits.size())
            new_beam_h_S[:,i::num_nodes] = self.W_s(new_beam_seq[:,i::num_nodes]) # weird [0] indexing
            print('new_beam_h_S size:', new_beam_h_S.size())
            
            sorted_scores, sorted_indices = torch.sort(new_beam_scores, dim=0, descending=True)
            print('sorted_scores:', sorted_scores)
            print('sorted_indices:', sorted_indices)
            new_beam_seq[:,i::num_nodes] = torch.gather(new_beam_seq[:,i::num_nodes], dim=0, index=sorted_indices)

            # reorganize h_S and logits
            expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, new_beam_h_S[:, i::num_nodes].size(-1))
            print('size of expanded_indices:', expanded_indices.size())
            new_beam_h_S[:,i::num_nodes] =  torch.gather(new_beam_h_S[:,i::num_nodes], dim=0, index=expanded_indices)
            new_beam_logits[:,i::num_nodes] =  torch.gather(new_beam_logits[:,i::num_nodes], dim=0, index=expanded_indices)

            # prints
            print('size of new_beam_seq_sorted:', new_beam_seq.size())
            print('new_beam_seq_sorted reshaped:', new_beam_seq[0,i::num_nodes])
            print('seq:', seq[i::num_nodes])
            print('size seq:', seq[i::num_nodes].size())

            # update metrics for both beams
            seq[i::num_nodes] = new_beam_seq[0,i::num_nodes]
            print('seq:', seq)
            logits[i::num_nodes] = new_beam_logits[0,i::num_nodes]
            print('logits:', logits)
            h_S[i::num_nodes] = new_beam_h_S[0,i::num_nodes]
            print('h_S size:', h_S.size())

            scores = sorted_scores[0]            
        
        # get ordered scores
        beamw_scores = scores.view(beam_width, -1)
        beamw_sorted_scores, beamw_sorted_indices = torch.sort(beamw_scores, dim=0, descending=True)

        # reshape tensors
        final_seq = seq.view(beam_width, n_samples, num_nodes)
        print(beamw_sorted_indices)
        final_logits = logits.view(beam_width, n_samples, num_nodes, self.out_dim)

        # reorganize according to indices
        expanded_indices = beamw_sorted_indices.unsqueeze(-1).expand(-1, n_samples, num_nodes)
        print('size expanded indices:', expanded_indices.size())
        final_seq = torch.gather(final_seq, dim=0, index=expanded_indices)
        print(final_seq)
        expanded_indices = beamw_sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, n_samples, num_nodes, self.out_dim)
        final_logits = torch.gather(final_logits, dim=0, index=expanded_indices)

        # use sorted indices to get final tensors
        final_scores = beamw_sorted_scores[0]
        print('final_scores:', final_scores)
        final_seq = final_seq[0]
        print('final_seq:', final_seq)
        final_logits = final_logits[0]
        print('final_logits:', final_logits)

        if return_logits:
            return final_seq.view(n_samples, num_nodes), final_logits.view(n_samples, num_nodes, self.out_dim)
        else:    
            return final_seq.view(n_samples, num_nodes)
        
    @torch.no_grad()
    def sample_mod2_bert(
            self, 
            batch, 
            n_samples, 
            temperature: Optional[float] = 0.1, 
            logit_bias: Optional[torch.Tensor] = None,
            return_logits: Optional[bool] = False,
            beam_width: Optional[int] = 2,
            beam_branch: Optional[int] = 2,
            sampling_strategy: Optional[str] = "categorical",
            top_k: Optional[int] = 0,
            top_p: Optional[float] = 0.0,
            min_p: Optional[float] = 0.0,
            bert: Optional[bool] = True,
        ):
        '''
        Samples sequences autoregressively from the distribution
        learned by the model.

        Args:
            batch (torch_geometric.data.Data): mini-batch containing one
                RNA backbone to design sequences for
            n_samples (int): number of samples
            temperature (float): temperature to use in softmax over 
                the categorical distribution
            logit_bias (torch.Tensor): bias to add to logits during sampling
                to manually fix or control nucleotides in designed sequences,
                of shape [n_nodes, 4]
            return_logits (bool): whether to return logits or 
            sampling_strategy (str): one of "categorical", "greedy", "top_k", "top_p", "min_p", etc.
            top_k (int): if using top-k sampling, how many tokens to keep
            top_p (float): if using nucleus (top-p) sampling, what cumulative probability threshold
            min_p (float): if using min-p sampling, probability threshold w.r.t max prob
            beam_width (int): number of beams to maintain during search
            beam_branch (int): number of samples to get from sampling strategy
        Returns:
            seq (torch.Tensor): int tensor of shape [n_samples, n_nodes]
                                based on the residue-to-int mapping of
                                the original training data
            logits (torch.Tensor): logits of shape [n_samples, n_nodes, 4]
                                   (only if return_logits is True)
        ''' 
        print('Batch size: ', batch.size())
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
    
        device = edge_index.device
        num_nodes = h_V[0].shape[0]
        print('Number of nodes: ', num_nodes)
        
        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        
        # Pool multi-conformation features
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        
        # Repeat features for sampling n_samples times
        # might have to change this
        h_V = (h_V[0].repeat(beam_width*n_samples, 1),
            h_V[1].repeat(beam_width*n_samples, 1, 1))
        h_E = (h_E[0].repeat(beam_width*n_samples, 1),
            h_E[1].repeat(beam_width*n_samples, 1, 1))

        # Expand edge index for autoregressive decoding
        edge_index = edge_index.expand(beam_width*n_samples, -1, -1)
        offset = num_nodes * torch.arange(beam_width*n_samples, device=device).view(beam_width*n_samples, 1, 1)
        edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
        # This is akin to 'batching' (in PyG style) n_samples copies of the graph
        
        scores = torch.zeros(beam_width*n_samples, dtype=torch.float, device=device)  # cumulative log-probability
        seq = torch.zeros(beam_width*num_nodes*n_samples, dtype=torch.int, device=device)  # decoded tokens (to be filled)
        h_S = torch.zeros(beam_width*num_nodes*n_samples, self.out_dim, device=device)
        # Each decoder layer keeps its own cache (here cloned from the pooled encoder features)
        h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]
        # Optionally, you can store logits for later inspection
        logits = torch.zeros(beam_width*num_nodes*n_samples, self.out_dim, device=device)

        # Decode one token at a time
        for i in range(num_nodes):
            print('Iteration : ', i)

            # --- Prepare messages for decoding token at position i ---
            # In the original sample(), h_S is used via indexing with edge_index.
            # Here we prepare the subset h_S_ corresponding to incoming edges for node i.
            h_S_ = h_S[edge_index[0]]
            # Zero out contributions from nodes not yet decoded:
            h_S_[edge_index[0] >= edge_index[1]] = 0

            # Concatenate h_S_ with edge features
            h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
            # Select only the incoming edges for node i:
            edge_mask = edge_index[1] % num_nodes == i  # True for all edges where dst is node i
            edge_index_ = edge_index[:, edge_mask]  # subset all incoming edges to node i
            h_E_ = tuple_index(h_E_, edge_mask)

            # Create a mask that is True only for the current node i (across all copies in the beam)
            node_mask = torch.zeros(beam_width*n_samples*num_nodes, device=device, dtype=torch.bool)
            node_mask[i::num_nodes] = True  # True for all nodes i and its repeats
            # not entirely sure if the thing above is correct

            # --- Pass through decoder layers ---
            # We simulate the same decoder forward pass as in sample(), updating the cache.
            for j, layer in enumerate(self.decoder_layers):
                out = layer(h_V_cache[j], edge_index_, h_E_,
                        autoregressive_x=h_V_cache[0], node_mask=node_mask)
                out = tuple_index(out, node_mask)  # subset out to only node i and its repeats
                # Update the cache for the next layer if needed
                if j < len(self.decoder_layers)-1:
                    h_V_cache[j+1][0][i::num_nodes] = out[0]
                    h_V_cache[j+1][1][i::num_nodes] = out[1]
            
            # Final logits for node i:
            lgts = self.W_out(out)
            print('lgts size:', lgts.size())
            
            # Adding bert embeddings
            if bert:
                print('Batch sequence:', batch.seq)
                dynamic_tokens = dynamic_tokenize_preprocessing(batch.seq)
                bert_ids = tokenizer(dynamic_tokens, return_tensors='pt', padding=True, truncation=True)
                print('bert ids:', bert_ids)
                
                with torch.no_grad():
                    bert_emb = mysterybert(**bert_ids)
                    logit_bias = bert_emb.logits
                    print('bert embeddings size:', bert_emb)
                    print('logit bias size:', logit_bias.size())
                    #logits = autoregressive_sampling_with_embeddings(logits, bert_emb)

            # Add logit bias if provided to fix or bias positions
            if logit_bias is not None:
                lgts += logit_bias[i]

            # Sample from logits
            top_tokens, log_probs = choose_nts(lgts, strategy=sampling_strategy, beam_branch=beam_branch,
                                    temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)
            # log probs will return probabilities for each nucleotide type
            # top tokens will return beam_branch samples of tokens for each of the sequences in n_samples
            print('logits size:', lgts.size())

            # For each candidate token, create a new beam candidate.
            new_beam_seq = seq.clone().repeat(beam_branch, 1)
            new_beam_h_S = h_S.clone().repeat(beam_branch, 1, 1) ## ADD STH HERE !!!!!
            new_beam_logits = logits.clone().repeat(beam_branch, 1, 1)
            top_log_probs_beam = log_probs.gather(dim=1, index=top_tokens)
            top_log_probs_beam = top_log_probs_beam.transpose(0, 1)

            new_beam_scores = scores.repeat(beam_branch, 1) + top_log_probs_beam
            new_beam_seq[:,i::num_nodes] = top_tokens.transpose(0,1)
            print('new_beam_seq:', new_beam_seq)
            new_beam_logits[:,i::num_nodes] = lgts  # store the logits for analysis
            new_beam_h_S[:,i::num_nodes] = self.W_s(new_beam_seq[:,i::num_nodes]) # weird [0] indexing
            
            sorted_scores, sorted_indices = torch.sort(new_beam_scores, dim=0, descending=True)
            new_beam_seq[:,i::num_nodes] = torch.gather(new_beam_seq[:,i::num_nodes], dim=0, index=sorted_indices)

            # reorganize h_S and logits
            expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, new_beam_h_S[:, i::num_nodes].size(-1))
            new_beam_h_S[:,i::num_nodes] =  torch.gather(new_beam_h_S[:,i::num_nodes], dim=0, index=expanded_indices)
            new_beam_logits[:,i::num_nodes] =  torch.gather(new_beam_logits[:,i::num_nodes], dim=0, index=expanded_indices)

            # update metrics for both beams
            seq[i::num_nodes] = new_beam_seq[0,i::num_nodes]
            logits[i::num_nodes] = new_beam_logits[0,i::num_nodes]
            h_S[i::num_nodes] = new_beam_h_S[0,i::num_nodes]

            scores = sorted_scores[0]            
        
        # get ordered scores
        beamw_scores = scores.view(beam_width, -1)
        beamw_sorted_scores, beamw_sorted_indices = torch.sort(beamw_scores, dim=0, descending=True)

        # reshape tensors
        final_seq = seq.view(beam_width, n_samples, num_nodes)
        final_logits = logits.view(beam_width, n_samples, num_nodes, self.out_dim)

        # reorganize according to indices
        expanded_indices = beamw_sorted_indices.unsqueeze(-1).expand(-1, n_samples, num_nodes)
        final_seq = torch.gather(final_seq, dim=0, index=expanded_indices)
        expanded_indices = beamw_sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, n_samples, num_nodes, self.out_dim)
        final_logits = torch.gather(final_logits, dim=0, index=expanded_indices)

        # use sorted indices to get final tensors
        final_scores = beamw_sorted_scores[0]
        final_seq = final_seq[0]
        final_logits = final_logits[0]

        if return_logits:
            return final_seq.view(n_samples, num_nodes), final_logits.view(n_samples, num_nodes, self.out_dim)
        else:    
            return final_seq.view(n_samples, num_nodes)

    @torch.no_grad()
    def sample_mod3(
            self, 
            batch, 
            n_samples, 
            temperature: Optional[float] = 0.1, 
            logit_bias: Optional[torch.Tensor] = None,
            return_logits: Optional[bool] = False,
            beam_width: Optional[int] = 2,
            beam_branch: Optional[int] = 2,
            sampling_strategy: Optional[str] = "categorical",
            top_k: Optional[int] = 0,
            top_p: Optional[float] = 0.0,
            min_p: Optional[float] = 0.0,
        ):
        """
        Vectorized beam–search version of sample_mod.
        The beams are maintained in tensors with shape:
            score:    (n_samples, beam_width)
            seq:      (n_samples, beam_width, n_nodes)
            h_S:      (n_samples, beam_width, n_nodes, self.out_dim)
            h_V_cache: for each decoder layer, a tuple of tensors with shape (n_samples, beam_width, n_nodes, d)
            logits:   (n_samples, beam_width, n_nodes, self.out_dim)
        At each autoregressive step (over nodes) we expand each beam by beam_branch candidates,
        update the cumulative scores, and then select the top beam_width beams using torch.topk.
        """
        print('Batch size: ', batch.size())
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index

        device = edge_index.device
        num_nodes = h_V[0].shape[0]
        print('Number of nodes: ', num_nodes)

        # Encode the input graph
        h_V = self.W_v(h_V)  # (n_nodes, d_s), (n_nodes, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, d_se), (n_edges, d_ve, 3)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)
        
        # Pool multi–conformation features.
        # (After pooling, h_V[0] is (n_nodes, d_s) and h_V[1] is (n_nodes, d_v, 3))
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        
        # Repeat features for n_samples (note: original code flattens samples over nodes)
        # Here, we “tile” the encoder outputs so that later we can reshape to (n_samples, n_nodes, ...)
        h_V0 = h_V[0].repeat(n_samples, 1)         # (n_samples*n_nodes, d_s)
        h_V1 = h_V[1].repeat(n_samples, 1, 1)        # (n_samples*n_nodes, d_v, 3)
        # Reshape so that each sample becomes a separate graph:
        h_V0 = h_V0.view(n_samples, num_nodes, -1)   # (n_samples, n_nodes, d_s)
        h_V1 = h_V1.view(n_samples, num_nodes, -1)   # (n_samples, n_nodes, d_v, ...)
        
        # --- Initialize beam state tensors ---
        # beam_scores: cumulative log–probabilities, shape (n_samples, beam_width)
        beam_scores = torch.zeros(n_samples, beam_width, device=device)
        # beam_seq: decoded tokens for each node, shape (n_samples, beam_width, num_nodes)
        beam_seq = torch.zeros(n_samples, beam_width, num_nodes, dtype=torch.int, device=device)
        # beam_h_S: decoder state per node, shape (n_samples, beam_width, num_nodes, self.out_dim)
        beam_h_S = torch.zeros(n_samples, beam_width, num_nodes, self.out_dim, device=device)
        # beam_logits: store logits at each node, shape (n_samples, beam_width, num_nodes, self.out_dim)
        beam_logits = torch.zeros(n_samples, beam_width, num_nodes, self.out_dim, device=device)
        
        # Initialize decoder caches for each layer.
        # Here we assume that each layer’s cache is per node.
        beam_h_V_cache = []
        for _ in self.decoder_layers:
            cache0 = h_V0.unsqueeze(1).expand(n_samples, beam_width, num_nodes, -1).clone()
            cache1 = h_V1.unsqueeze(1).expand(n_samples, beam_width, num_nodes, -1).clone()
            beam_h_V_cache.append((cache0, cache1))
        
        # For the autoregressive loop we assume that batch.edge_index corresponds to a single graph
        # with node indices in 0...num_nodes-1.
        edge_index_graph = batch.edge_index  # shape (2, n_edges)

        # --- Decode tokens one at a time (for each node) ---
        for i in range(num_nodes):
            print('Decoding node:', i)
            # --- Prepare a vectorized decoder forward pass for the current node ---
            # For each beam we only need the decoder state at node i.
            # We flatten the first two dimensions so that our layers can process all beam candidates at once.
            flat_beam_h_S = beam_h_S.view(n_samples * beam_width, num_nodes, self.out_dim)
            # Here we extract the current node’s state: shape (n_samples*beam_width, self.out_dim)
            current_state = flat_beam_h_S[:, i, :]  

            # (In your original code you use h_S together with the graph structure to compute messages.
            # In this simplified example, we assume that each decoder layer operates on the current node state.)
            # Process through the decoder layers:
            out = current_state  # initial state for this decoding step
            for j, layer in enumerate(self.decoder_layers):
                # If your layer needs to incorporate messages from other nodes using edge_index,
                # you can “expand” the beam dimension by flattening (n_samples, beam_width) to (n_samples*beam_width)
                # and using the original edge_index (which is valid for a single graph).
                # (Adjust the following call as needed.)
                out = layer(h_V_cache[j], edge_index_, h_E_,
                        autoregressive_x=h_V_cache[0], node_mask=node_mask)
                
                out = tuple_index(out, node_mask)  # subset out to only node i and its repeats
                #out = layer(out)  # expect out shape (n_samples*beam_width, self.out_dim)
                # Optionally update the decoder cache for layer j+1.
                # For example, if you need to store out in the cache for node i:
                # (Here we reshape and store it in the cache for later use.)
                if j < len(self.decoder_layers)-1:
                    cache0, cache1 = beam_h_V_cache[j+1]
                    # Update the cached state at node i (reshape out appropriately)
                    cache0 = cache0.view(n_samples * beam_width, num_nodes, -1)
                    cache0[:, i, :] = out
                    beam_h_V_cache[j+1] = (cache0.view(n_samples, beam_width, num_nodes, -1), cache1)
            # Final logits for node i:
            lgts = self.W_out(out)  # shape (n_samples*beam_width, self.out_dim)
            lgts = lgts.view(n_samples, beam_width, self.out_dim)
            if logit_bias is not None:
                lgts = lgts + logit_bias[i]  # broadcast over beams

            # --- Branching: sample beam_branch candidates for each beam ---
            # Assume choose_nts is modified so that it works on a tensor of shape (n_samples, beam_width, self.out_dim)
            # and returns:
            #   top_tokens: (n_samples, beam_width, beam_branch)   [candidate tokens]
            #   top_log_probs: (n_samples, beam_width, beam_branch)  [log probability for each candidate]
            top_tokens, top_log_probs = choose_nts(
                lgts, 
                strategy=sampling_strategy, 
                temperature=temperature, 
                beam_branch=beam_branch, 
                top_k=top_k, 
                top_p=top_p, 
                min_p=min_p
            )
            print(top_tokens)
            print(top_log_probs)
            # --- Update scores: add the candidate log–probs to the current beam scores.
            # beam_scores: (n_samples, beam_width)  → unsqueeze to (n_samples, beam_width, 1)
            expanded_scores = beam_scores.unsqueeze(-1) + top_log_probs  # (n_samples, beam_width, beam_branch)
            # Flatten the beam and branch dimensions:
            flat_scores = expanded_scores.view(n_samples, -1)  # (n_samples, beam_width * beam_branch)
            # --- Select the top beam_width beams using torch.topk ---
            top_scores, top_indices = torch.topk(flat_scores, beam_width, dim=1)
            # Convert flat indices into original beam index and branch index:
            new_beam_idx = top_indices // beam_branch   # (n_samples, beam_width) from the old beam dimension
            new_branch_idx = top_indices % beam_branch   # (n_samples, beam_width) from the branch dimension
            # Update beam_scores:
            beam_scores = top_scores  # new cumulative scores

            # --- Update sequence tokens ---
            # First, bring the current beam_seq (shape (n_samples, beam_width, num_nodes)) into the new order.
            beam_seq = torch.gather(
                beam_seq, 
                dim=1, 
                index=new_beam_idx.unsqueeze(-1).expand(n_samples, beam_width, num_nodes)
            )
            # Then update the token at the current position i.
            # Reshape top_tokens to flatten the beam and branch dims so that we can select the chosen candidate:
            flat_top_tokens = top_tokens.view(n_samples, -1)  # (n_samples, beam_width*beam_branch)
            selected_tokens = flat_top_tokens.gather(dim=1, index=top_indices)  # (n_samples, beam_width)
            beam_seq[:, :, i] = selected_tokens

            # --- Update decoder state h_S ---
            # First reorder the full h_S state according to the selected beams.
            beam_h_S = torch.gather(
                beam_h_S, 
                dim=1, 
                index=new_beam_idx.unsqueeze(-1).unsqueeze(-1).expand(n_samples, beam_width, num_nodes, self.out_dim)
            )
            # Then update the state at position i based on the newly chosen token.
            # (Assuming self.W_s converts tokens to a state vector.)
            beam_h_S[:, :, i, :] = self.W_s(selected_tokens)

            # --- Update logits similarly ---
            beam_logits = torch.gather(
                beam_logits, 
                dim=1, 
                index=new_beam_idx.unsqueeze(-1).unsqueeze(-1).expand(n_samples, beam_width, num_nodes, self.out_dim)
            )
            beam_logits[:, :, i, :] = lgts.view(n_samples, beam_width, self.out_dim)
            
            # --- Update decoder caches for each layer ---
            new_beam_h_V_cache = []
            for j, (cache0, cache1) in enumerate(beam_h_V_cache):
                cache0 = torch.gather(
                    cache0, 
                    dim=1, 
                    index=new_beam_idx.unsqueeze(-1).unsqueeze(-1).expand(n_samples, beam_width, num_nodes, cache0.size(-1))
                )
                cache1 = torch.gather(
                    cache1, 
                    dim=1, 
                    index=new_beam_idx.unsqueeze(-1).unsqueeze(-1).expand(n_samples, beam_width, num_nodes, cache1.size(-1))
                )
                new_beam_h_V_cache.append((cache0, cache1))
            beam_h_V_cache = new_beam_h_V_cache

        # --- After decoding all nodes, select the best beam (highest score) per sample ---
        best_scores, best_beam_idx = beam_scores.max(dim=1)  # (n_samples,)
        best_seq = beam_seq[torch.arange(n_samples, device=device), best_beam_idx, :]  # (n_samples, num_nodes)
        best_logits = beam_logits[torch.arange(n_samples, device=device), best_beam_idx, :, :]
        
        print('Best beam scores:', best_scores)
        if return_logits:
            return best_seq, best_logits
        else:
            return best_seq


class NonAutoregressiveMultiGNNv1(torch.nn.Module):
    '''
    Non-Autoregressive GVP-GNN for **multiple** structure-conditioned RNA design.
    
    Takes in RNA structure graphs of type `torch_geometric.data.Data` 
    or `torch_geometric.data.Batch` and returns a categorical distribution
    over 4 bases at each position in a `torch.Tensor` of shape [n_nodes, 4].
    
    The standard forward pass requires sequence information as input
    and should be used for training or evaluating likelihood.
    For sampling or design, use `self.sample`.
    
    Args:
        node_in_dim (tuple): node dimensions in input graph
        node_h_dim (tuple): node dimensions to use in GVP-GNN layers
        node_in_dim (tuple): edge dimensions in input graph
        edge_h_dim (tuple): edge dimensions to embed in GVP-GNN layers
        num_layers (int): number of GVP-GNN layers in encoder/decoder
        drop_rate (float): rate to use in all dropout layers
        out_dim (int): output dimension (4 bases)
    '''
    def __init__(
        self,
        node_in_dim = (64, 4), 
        node_h_dim = (128, 16), 
        edge_in_dim = (32, 1), 
        edge_h_dim = (32, 1),
        num_layers = 3, 
        drop_rate = 0.1,
        out_dim = 4,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        activations = (F.silu, None)
        
        # Node input embedding
        self.W_v = torch.nn.Sequential(
            LayerNorm(self.node_in_dim),
            GVP(self.node_in_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True)
        )

        # Edge input embedding
        self.W_e = torch.nn.Sequential(
            LayerNorm(self.edge_in_dim),
            GVP(self.edge_in_dim, self.edge_h_dim, 
                activations=(None, None), vector_gate=True)
        )
        
        # Encoder layers (supports multiple conformations)
        self.encoder_layers = nn.ModuleList(
                MultiGVPConvLayer(self.node_h_dim, self.edge_h_dim, 
                                  activations=activations, vector_gate=True,
                                  drop_rate=drop_rate, norm_first=True)
            for _ in range(num_layers))
        
        # Output
        self.W_out = torch.nn.Sequential(
            LayerNorm(self.node_h_dim),
            GVP(self.node_h_dim, self.node_h_dim,
                activations=(None, None), vector_gate=True),
            GVP(self.node_h_dim, (self.out_dim, 0), 
                activations=(None, None))   
        )
    
    def forward(self, batch):

        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        
        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        # Pool multi-conformation features: 
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        # h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        h_V = (h_V[0].mean(dim=1), h_V[1].mean(dim=1))

        logits = self.W_out(h_V)  # (n_nodes, out_dim)
        
        return logits
    
    def sample(self, batch, n_samples, temperature=0.1, return_logits=False):
        
        with torch.no_grad():

            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)
            edge_index = batch.edge_index
        
            h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
            h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)
            
            for layer in self.encoder_layers:
                h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
            
            # Pool multi-conformation features
            # h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
            h_V = (h_V[0].mean(dim=1), h_V[1].mean(dim=1))
            
            logits = self.W_out(h_V)  # (n_nodes, out_dim)
            probs = F.softmax(logits / temperature, dim=-1)
            seq = torch.multinomial(probs, n_samples, replacement=True)  # (n_nodes, n_samples)

            if return_logits:
                return seq.permute(1, 0).contiguous(), logits.unsqueeze(0).repeat(n_samples, 1, 1)
            else:
                return seq.permute(1, 0).contiguous()
        
    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):

        if mask_confs.size(1) == 1:
            # Number of conformations is 1, no need to pool
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])
        
        # True num_conf for masked mean pooling
        n_conf_true = mask_confs.sum(1, keepdim=True)  # (n_nodes, 1)
        
        # Mask scalar features
        mask = mask_confs.unsqueeze(2)  # (n_nodes, n_conf, 1)
        h_V0 = h_V[0] * mask
        h_E0 = h_E[0] * mask[edge_index[0]]

        # Mask vector features
        mask = mask.unsqueeze(3)  # (n_nodes, n_conf, 1, 1)
        h_V1 = h_V[1] * mask
        h_E1 = h_E[1] * mask[edge_index[0]]
        
        # Average pooling multi-conformation features
        h_V = (h_V0.sum(dim=1) / n_conf_true,               # (n_nodes, d_s)
               h_V1.sum(dim=1) / n_conf_true.unsqueeze(2))  # (n_nodes, d_v, 3)
        h_E = (h_E0.sum(dim=1) / n_conf_true[edge_index[0]],               # (n_edges, d_se)
               h_E1.sum(dim=1) / n_conf_true[edge_index[0]].unsqueeze(2))  # (n_edges, d_ve, 3)

        return h_V, h_E

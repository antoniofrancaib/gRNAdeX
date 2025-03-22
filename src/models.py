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

#import transformers
#from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.constants import NUM_TO_LETTER, LETTER_TO_NUM


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
        #bert_dim = 768
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        #self.bert_dim = bert_dim
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
        # Add a linear projection layer to map BERT embeddings to the output dimension
        #self.bert_projection = nn.Linear(self.bert_dim, self.out_dim)

    def forward(self, batch):
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
        device = edge_index.device
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

        # TODO: change later!
        use_bert = False
        # Add bert-birna embeddings
        if use_bert and len(seq) <= 128: # TODO: adjust tokenizer or finetune it to our specific use-case
            # TODO: change this later !
            tokenizer = AutoTokenizer.from_pretrained("../BiRNA-BERT/TOKENIZER")
            bert_config = transformers.BertConfig.from_pretrained("buetnlpbio/birna-bert")
            #config.alibi_starting_size = 5000 # maximum sequence length updated to 5000 from config default of 1024 -- IF INCREASED TO 5,00 IT KILLS PROCESS !
            mysterybert = AutoModelForMaskedLM.from_pretrained("buetnlpbio/birna-bert",config=bert_config,trust_remote_code=True).to(device)
            mysterybert.cls = torch.nn.Identity()

            # Convert sequence tensor to strings for BERT tokenizer
            print('Shape of seq in tokens:', seq.shape)
            print('Length of sequence:', len(seq))
            seq_strings = " ".join([NUM_TO_LETTER[int(token)] for token in seq])
            bert_inputs = tokenizer(seq_strings, return_tensors='pt', add_special_tokens=False, padding=True, truncation=True).to(device)
            print('bert inputs:', bert_inputs)
            print('seq strings:', seq_strings)

            with torch.no_grad():
                bert_outputs = mysterybert(**bert_inputs)
                bert_embeddings = bert_outputs.logits.to(device)  # Shape: (batch_size, seq_len, hidden_dim)
                print('Shape of bert embeddings:', bert_embeddings.shape)

                # Project BERT embeddings to match the output dimension
                bert_logits = self.bert_projection(bert_embeddings).to(device)  # Shape: (batch_size, seq_len, out_dim)
                print('Shape of bert logits:', bert_logits.shape)

                # Add BERT logits to the model's logits
                print('Logits:', logits)
                print('Shape of logits:', logits.shape)
                logits += bert_logits  # Add BERT logits for the current position
        
        return logits
    
    @torch.no_grad()
    def sample(
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
            use_bert: Optional[bool] = False,
            avoid_sequences: Optional[list] = None
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
            avoid_sequences (list): list of sequences to avoid sampling
        Returns:
            seq (torch.Tensor): int tensor of shape [n_samples, n_nodes]
                                based on the residue-to-int mapping of
                                the original training data
            logits (torch.Tensor): logits of shape [n_samples, n_nodes, 4]
                                   (only if return_logits is True)
        ''' 
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        edge_index = batch.edge_index
    
        device = edge_index.device
        num_nodes = h_V[0].shape[0]
        
        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)
        
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        
        # Pool multi-conformation features
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
        
        # Repeat features for sampling n_samples times
        h_V = (h_V[0].repeat(beam_width*n_samples, 1),
            h_V[1].repeat(beam_width*n_samples, 1, 1))
        h_E = (h_E[0].repeat(beam_width*n_samples, 1),
            h_E[1].repeat(beam_width*n_samples, 1, 1))

        # Expand edge index for autoregressive decoding
        edge_index = edge_index.expand(beam_width*n_samples, -1, -1)
        offset = num_nodes * torch.arange(beam_width*n_samples, device=device).view(beam_width*n_samples, 1, 1)
        edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
        
        scores = torch.zeros(beam_width*n_samples, dtype=torch.float, device=device)  # cumulative log-probability
        seq = torch.zeros(beam_width*num_nodes*n_samples, dtype=torch.int, device=device)  # decoded tokens (to be filled)
        h_S = torch.zeros(beam_width*num_nodes*n_samples, self.out_dim, device=device)
        h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]
        logits = torch.zeros(beam_width*num_nodes*n_samples, self.out_dim, device=device)

        # Convert avoid_sequences to tensor format if provided
        if avoid_sequences is not None:
            avoid_tensors = []
            for seq_str in avoid_sequences:
                seq_tensor = torch.tensor([LETTER_TO_NUM[residue] for residue in seq_str], device=device)
                avoid_tensors.append(seq_tensor)
            avoid_tensors = torch.stack(avoid_tensors)

        print('Avoid tensors:', avoid_tensors)
        # Decode one token at a time
        for i in range(num_nodes):
            print('Iteration:', i)
            if i > 0:
                print('Sequence so far:', seq[i-1::num_nodes])
            #print('Sequence entire so far:', seq)
            h_S_ = h_S[edge_index[0]]
            h_S_[edge_index[0] >= edge_index[1]] = 0
            h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
            edge_mask = edge_index[1] % num_nodes == i
            edge_index_ = edge_index[:, edge_mask]
            h_E_ = tuple_index(h_E_, edge_mask)

            node_mask = torch.zeros(beam_width*n_samples*num_nodes, device=device, dtype=torch.bool)
            node_mask[i::num_nodes] = True

            for j, layer in enumerate(self.decoder_layers):
                out = layer(h_V_cache[j], edge_index_, h_E_,
                        autoregressive_x=h_V_cache[0], node_mask=node_mask)
                out = tuple_index(out, node_mask)
                if j < len(self.decoder_layers)-1:
                    h_V_cache[j+1][0][i::num_nodes] = out[0]
                    h_V_cache[j+1][1][i::num_nodes] = out[1]
            lgts = self.W_out(out)

            if logit_bias is not None:
                lgts += logit_bias[i]
            
            print(f"lgts before avoiding: {lgts}")

            # Add negative infinity to logits for sequences to avoid
            if avoid_sequences is not None:
                for avoid_seq in avoid_tensors:
                    print(f"Avoiding sequences: {avoid_seq}")
                    # Direct avoidance of current position in avoid_seq
                    #if i < len(avoid_seq):
                    #    lgts[..., avoid_seq[i]] = float('-inf')
                    #    print(f"lgts after avoiding: {lgts}")
                    
                    # Check if we're about to complete an avoid_seq pattern
                    if i >= len(avoid_seq) - 1:
                        print('Entering in iteration:', i)
                        # Get the current sequence window that would match all but the last token of avoid_seq
                        sequence_reshaped = seq.view(beam_width*n_samples, num_nodes)
                        print('sequence_reshaped:', sequence_reshaped)
                        # Extract the window of tokens that would match the avoid_seq pattern
                        # Get the last len(avoid_seq)-1 tokens generated so far
                        start_idx = i - (len(avoid_seq) - 1)
                        if start_idx < 0:
                            # Not enough tokens generated yet to match the pattern
                            seq_window = torch.zeros((beam_width*n_samples, len(avoid_seq)-1), dtype=torch.long, device=device)
                        else:
                            # Extract the relevant window from the sequence
                            seq_window = sequence_reshaped[:, start_idx:i]
                        print('seq_window:', seq_window)
                        # Make sure avoid_window has the same shape as seq_window
                        avoid_window = avoid_seq[:-1]
                        print('avoid_window:', avoid_window)
                        
                        # Check which rows match the avoid pattern
                        if len(avoid_window) == seq_window.size(1):  # Make sure dimensions match
                            # Compare each row with avoid_window
                            matches = torch.all(seq_window == avoid_window.unsqueeze(0), dim=1)
                            print('matches:', matches)
                            # Only set -inf for matching rows
                            if torch.any(matches):
                                # Create a mask for the matching rows
                                mask = matches  # Add dimension for broadcasting
                                
                                # Only modify logits for rows that match the pattern
                                lgts[mask, avoid_seq[-1]] = float('-inf')
                                print(f"Preventing completion of avoid_seq pattern in {torch.sum(matches)} sequences.")
                                print(f"Setting logit for {avoid_seq[-1]} to -inf for matching sequences")
                                print(f"lgts after preventing completion: {lgts}")

            top_tokens, log_probs = choose_nts(lgts, strategy=sampling_strategy, beam_branch=beam_branch,
                                    temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)

            new_beam_seq = seq.clone().repeat(beam_branch, 1)
            new_beam_h_S = h_S.clone().repeat(beam_branch, 1, 1)
            new_beam_logits = logits.clone().repeat(beam_branch, 1, 1)
            top_log_probs_beam = log_probs.gather(dim=1, index=top_tokens)
            top_log_probs_beam = top_log_probs_beam.transpose(0, 1)
            new_beam_scores = scores.repeat(beam_branch, 1) + top_log_probs_beam

            new_beam_seq[:,i::num_nodes] = top_tokens.transpose(0,1)
            new_beam_logits[:,i::num_nodes] = lgts
            new_beam_h_S[:,i::num_nodes] = self.W_s(new_beam_seq[:,i::num_nodes])
            
            sorted_scores, sorted_indices = torch.sort(new_beam_scores, dim=0, descending=True)
            new_beam_seq[:,i::num_nodes] = torch.gather(new_beam_seq[:,i::num_nodes], dim=0, index=sorted_indices)

            expanded_indices = sorted_indices.unsqueeze(-1).expand(-1, -1, new_beam_h_S[:, i::num_nodes].size(-1))
            new_beam_h_S[:,i::num_nodes] =  torch.gather(new_beam_h_S[:,i::num_nodes], dim=0, index=expanded_indices)
            new_beam_logits[:,i::num_nodes] =  torch.gather(new_beam_logits[:,i::num_nodes], dim=0, index=expanded_indices)

            seq[i::num_nodes] = new_beam_seq[0,i::num_nodes]
            logits[i::num_nodes] = new_beam_logits[0,i::num_nodes]
            h_S[i::num_nodes] = new_beam_h_S[0,i::num_nodes]
            scores = sorted_scores[0]
            print('Sequence new:', seq[i::num_nodes])
        
        beamw_scores = scores.view(beam_width, -1)
        beamw_sorted_scores, beamw_sorted_indices = torch.sort(beamw_scores, dim=0, descending=True)

        final_seq = seq.view(beam_width, n_samples, num_nodes)
        final_logits = logits.view(beam_width, n_samples, num_nodes, self.out_dim)

        expanded_indices = beamw_sorted_indices.unsqueeze(-1).expand(-1, n_samples, num_nodes)
        final_seq = torch.gather(final_seq, dim=0, index=expanded_indices)
        expanded_indices = beamw_sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, n_samples, num_nodes, self.out_dim)
        final_logits = torch.gather(final_logits, dim=0, index=expanded_indices)

        final_scores = beamw_sorted_scores[0]
        final_seq = final_seq[0]
        final_logits = final_logits[0]

        if return_logits:
            return final_seq.view(n_samples, num_nodes), final_logits.view(n_samples, num_nodes, self.out_dim)
        else:    
            return final_seq.view(n_samples, num_nodes)
        
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

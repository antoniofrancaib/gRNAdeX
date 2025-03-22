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
        bert_dim = 768,
        use_bert = True,
        bert_model_name = "buetnlpbio/birna-bert",
        bert_tokenizer_name = "buetnlpbio/birna-tokenizer",
        max_seq_len = 128  # Increased from 128
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.node_h_dim = node_h_dim
        self.edge_in_dim = edge_in_dim
        self.edge_h_dim = edge_h_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.bert_dim = bert_dim
        self.use_bert = use_bert
        self.max_seq_len = max_seq_len
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
        
        # BERT Integration
        if self.use_bert:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer_name)
            bert_config = transformers.BertConfig.from_pretrained(bert_model_name)
            # Adjust config if needed for longer sequences
            self.bert_model = AutoModelForMaskedLM.from_pretrained(
                bert_model_name, 
                config=bert_config, 
                trust_remote_code=True
            )
            # Replace classification layer with identity
            self.bert_model.cls = torch.nn.Identity()
            
            # Linear projection for BERT embeddings
            self.bert_projection = nn.Linear(self.bert_dim, self.out_dim)
            
            # Adapter layer to handle potential shape mismatches
            self.bert_adapter = nn.Linear(self.out_dim, self.out_dim)
            
            # Weighting factor for BERT contribution (learnable parameter)
            self.bert_weight = nn.Parameter(torch.tensor(0.5))

    def convert_seq_to_tokens(self, seq):
        """
        Convert numeric sequence tensor to string tokens for BERT
        
        Args:
            seq: Tensor of shape [n_nodes] containing integer indices
            
        Returns:
            String of tokens with spaces between them
        """
        return " ".join([NUM_TO_LETTER[int(token)] for token in seq])
        
    def get_bert_embeddings(self, seq, device):
        """
        Process sequence through BERT model to get embeddings
        
        Args:
            seq: Tensor of shape [n_nodes] containing integer indices
            device: Device to run computation on
            
        Returns:
            Tensor of shape [n_nodes, out_dim] containing projected BERT embeddings
        """
        if not self.use_bert or len(seq) > self.max_seq_len:
            return None
            
        # Convert sequence to string tokens
        seq_string = self.convert_seq_to_tokens(seq)
        
        # Tokenize for BERT input
        bert_inputs = self.tokenizer(
            seq_string, 
            return_tensors='pt', 
            add_special_tokens=False,
            return_attention_mask=True
        ).to(device)
        
        # Get BERT outputs
        with torch.no_grad():
            bert_outputs = self.bert_model(**bert_inputs)
            bert_embeddings = bert_outputs.logits  # Shape: [1, seq_len, hidden_dim]
            
            # Handle potential padding in tokenizer output
            attention_mask = bert_inputs['attention_mask'].squeeze()
            active_embeddings = bert_embeddings.squeeze()[attention_mask.bool()]
            
            # Match the size with our original sequence
            if active_embeddings.size(0) != len(seq):
                bert_embeddings = F.interpolate(
                    bert_embeddings.permute(0, 2, 1),
                    size=len(seq),
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1).squeeze()
            else:
                bert_embeddings = active_embeddings
            
            # Project to match our model's output dimension
            bert_logits = self.bert_projection(bert_embeddings)
            
            # If we're at the final softmax layer, we need to realign the logits
            # to match our GNN's token ordering
            
            # Define the mapping from BERT token IDs to our model's token IDs
            # This depends on your specific encoding schemes
            bert_to_gnn_mapping = {
                7: 0,  # BERT's A (7) maps to GNN's A (0)
                8: 2,  # BERT's G (8) maps to GNN's G (2)
                1: 1,  # BERT's C (1) maps to GNN's C (1)
                5: 3,  # BERT's U (5) maps to GNN's U (3)
            }
            
            # Rearrange BERT logits to match our GNN's token ordering
            # This is a simplified version - you might need a more complex approach
            # depending on the full tokenization differences
            aligned_bert_logits = torch.zeros_like(bert_logits)
            for bert_id, gnn_id in bert_to_gnn_mapping.items():
                # If BERT has a token that corresponds to one of our nucleotides,
                # map its logit to the correct position in our output
                if bert_id < bert_logits.size(1):
                    aligned_bert_logits[:, gnn_id] = bert_logits[:, bert_id]
            
        return aligned_bert_logits

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
            h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)
        
        # Get base model logits
        logits = self.W_out(h_V)

        # Add BERT-BiRNA embeddings if enabled and sequence is within length limit
        if self.use_bert and len(seq) <= self.max_seq_len:
            bert_logits = self.get_bert_embeddings(seq, device)
            if bert_logits is not None:
                # Combine GNN logits with BERT logits using learnable weight
                bert_adapted = self.bert_adapter(bert_logits)
                logits = logits + self.bert_weight * bert_adapted
        
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
        ):
        '''
        Samples sequences autoregressively from the distribution
        learned by the model.
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
        # Each decoder layer keeps its own cache (here cloned from the pooled encoder features)
        h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]
        # Optionally, you can store logits for later inspection
        logits = torch.zeros(beam_width*num_nodes*n_samples, self.out_dim, device=device)

        # For BERT integration during sampling
        bert_logits_cache = None
        current_sequence = torch.zeros(num_nodes, dtype=torch.int, device=device)
        
        # Decode one token at a time
        for i in range(num_nodes):
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

            # --- Pass through decoder layers ---
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
            
            # BERT integration during sampling:
            if self.use_bert and i > 0 and i % 4 == 0 and i <= self.max_seq_len:
                # Every few steps (to avoid excessive computation), we update our BERT predictions
                # Based on the sequence generated so far
                # For the first beam
                if i <= num_nodes // 2:  # Only run this for first half of sequence to improve speed
                    current_sequence[:i] = seq[:num_nodes][:i]  # Update with generated sequence
                    bert_logits_cache = self.get_bert_embeddings(current_sequence[:i], device)
                    
                    # If we have valid BERT predictions, use them to influence logits
                    if bert_logits_cache is not None:
                        bert_adapted = self.bert_adapter(bert_logits_cache)
                        # Only apply to current position
                        bert_influence = torch.zeros_like(lgts)
                        if i < bert_adapted.shape[0]:
                            bert_position = bert_adapted[i].repeat(lgts.shape[0], 1)
                            bert_influence = self.bert_weight * bert_position
                            lgts = lgts + bert_influence

            # Add logit bias if provided to fix or bias positions
            if logit_bias is not None:
                lgts += logit_bias[i]
                
            # Sample from logits
            top_tokens, log_probs = choose_nts(lgts, strategy=sampling_strategy, beam_branch=beam_branch,
                                    temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p)
                                    
            # For each candidate token, create a new beam candidate
            new_beam_seq = seq.clone().repeat(beam_branch, 1)
            new_beam_h_S = h_S.clone().repeat(beam_branch, 1)
            new_beam_logits = logits.clone().repeat(beam_branch, 1)
            top_log_probs_beam = log_probs.gather(dim=1, index=top_tokens)
            top_log_probs_beam = top_log_probs_beam.transpose(0, 1)
            new_beam_scores = scores.repeat(beam_branch) + top_log_probs_beam.flatten()

            new_beam_seq[:,i::num_nodes] = top_tokens.transpose(0,1).flatten().unsqueeze(0)
            new_beam_logits[:,i::num_nodes] = lgts.repeat(beam_branch, 1)  # store the logits for analysis
            new_beam_h_S[:,i::num_nodes] = self.W_s(new_beam_seq[:,i::num_nodes])
            
            # Select top beams
            _, sorted_indices = torch.sort(new_beam_scores.view(beam_branch, -1), dim=0, descending=True)
            sorted_indices = sorted_indices[0]  # Keep top beam_width candidates
            
            # Update sequence, scores, and h_S for next iteration
            seq[i::num_nodes] = new_beam_seq[sorted_indices,i::num_nodes]
            logits[i::num_nodes] = new_beam_logits[sorted_indices,i::num_nodes]
            h_S[i::num_nodes] = new_beam_h_S[sorted_indices,i::num_nodes]
            scores = new_beam_scores.view(beam_branch, -1)[0, sorted_indices]
        
        # Reshape final outputs
        final_seq = seq.view(beam_width*n_samples, num_nodes)[:n_samples]
        
        if return_logits:
            final_logits = logits.view(beam_width*n_samples, num_nodes, self.out_dim)[:n_samples]
            return final_seq, final_logits
        else:    
            return final_seq
            
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

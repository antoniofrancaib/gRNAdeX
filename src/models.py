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


class AutoregressiveMultiGNNv1(torch.nn.Module):
    '''
    Autoregressive GVP-GNN with attention for **multiple** structure-conditioned RNA design.
    
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
        attention_heads (int): number of attention heads in hybrid layer
        attention_dropout (float): dropout rate for attention mechanism
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
        attention_heads = 4,
        attention_dropout = 0.1,
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
        # Uses the updated HybridGVPAttentionLayer with MultiGVPConv
        self.encoder_layers = nn.ModuleList(
                HybridGVPAttentionLayer(self.node_h_dim, self.edge_h_dim, 
                          activations=activations, vector_gate=True,
                          drop_rate=drop_rate, norm_first=True,
                          n_heads=attention_heads,
                          attention_dropout=attention_dropout)
            for _ in range(num_layers))

        # Initialize cross-attention parameters (for pooling multi-conformation features)
        # Create attention projection layers
        self.conf_attn_projections = nn.ModuleDict({
            # Node projections
            'node_q': nn.Linear(self.node_h_dim[0], self.node_h_dim[0], bias=False),
            'node_k': nn.Linear(self.node_h_dim[0], self.node_h_dim[0], bias=False),
            'node_v_scalar': nn.Linear(self.node_h_dim[0], self.node_h_dim[0], bias=False),
            'node_out': nn.Linear(self.node_h_dim[0], self.node_h_dim[0], bias=True),
            
            # Edge projections
            'edge_q': nn.Linear(self.edge_h_dim[0], self.edge_h_dim[0], bias=False),
            'edge_k': nn.Linear(self.edge_h_dim[0], self.edge_h_dim[0], bias=False),
            'edge_v_scalar': nn.Linear(self.edge_h_dim[0], self.edge_h_dim[0], bias=False),
            'edge_out': nn.Linear(self.edge_h_dim[0], self.edge_h_dim[0], bias=True)
        })
        
        # Scaling factors for dot-product attention
        self.register_buffer('node_scale', torch.sqrt(torch.tensor(self.node_h_dim[0], dtype=torch.float32)))
        self.register_buffer('edge_scale', torch.sqrt(torch.tensor(self.edge_h_dim[0], dtype=torch.float32)))
        
        # Attention dropout for regularization
        self.attn_dropout = nn.Dropout(attention_dropout)
    
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
        # Extract node features, edge features, edge indices and sequence
        h_V = (batch.node_s, batch.node_v)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = (batch.edge_s, batch.edge_v)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)
        edge_index = batch.edge_index  # [2, n_edges]
        seq = batch.seq  # [n_nodes]

        # Embed node and edge features
        h_V = self.W_v(h_V)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)
        h_E = self.W_e(h_E)  # (n_edges, n_conf, d_se), (n_edges, n_conf, d_ve, 3)

        # Process through encoder layers
        # Each layer maintains the multi-conformation structure
        for layer in self.encoder_layers:
            h_V = layer(h_V, edge_index, h_E)  # (n_nodes, n_conf, d_s), (n_nodes, n_conf, d_v, 3)

        # Pool multi-conformation features: 
        # nodes: (n_nodes, d_s), (n_nodes, d_v, 3)
        # edges: (n_edges, d_se), (n_edges, d_ve, 3)
        h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)

        encoder_embeddings = h_V
        
        # Add sequence information to edge features for autoregressive processing
        h_S = self.W_s(seq)
        h_S = h_S[edge_index[0]]
        h_S[edge_index[0] >= edge_index[1]] = 0
        h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
        
        # Process through decoder layers
        for layer in self.decoder_layers:
            h_V = layer(h_V, edge_index, h_E, autoregressive_x = encoder_embeddings)
        
        # Final output projection
        logits = self.W_out(h_V)
        
        return logits
    
    @torch.no_grad()
    def sample(
            self, 
            batch, 
            n_samples, 
            temperature: Optional[float] = 0.1, 
            logit_bias: Optional[torch.Tensor] = None,
            return_logits: Optional[bool] = False
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
            return_logits (bool): whether to return logits or not
        
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

        h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]

        # Decode one token at a time
        for i in range(num_nodes):
            
            h_S_ = h_S[edge_index[0]]
            h_S_[edge_index[0] >= edge_index[1]] = 0
            h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
                    
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
            seq[i::num_nodes] = Categorical(logits=lgts / temperature).sample()
            h_S[i::num_nodes] = self.W_s(seq[i::num_nodes])
            logits[i::num_nodes] = lgts

        if return_logits:
            return seq.view(n_samples, num_nodes), logits.view(n_samples, num_nodes, self.out_dim)
        else:    
            return seq.view(n_samples, num_nodes)
        
    def pool_multi_conf(self, h_V, h_E, mask_confs, edge_index):
        """
        Pool features from multiple conformations using self-attention.
        Attention calculations incorporate both scalar and vector features.
        Takes the first token from the attended output instead of averaging.
        
        Args:
            h_V (tuple): (scalar, vector) node features, shapes:
                         scalar: [n_nodes, n_conf, d_s]
                         vector: [n_nodes, n_conf, d_v, 3]
            h_E (tuple): (scalar, vector) edge features, shapes:
                         scalar: [n_edges, n_conf, d_se]
                         vector: [n_edges, n_conf, d_ve, 3]
            mask_confs (torch.Tensor): binary mask of valid conformations [n_nodes, n_conf]
            edge_index (torch.Tensor): edge indices [2, n_edges]
            
        Returns:
            tuple: ((node_scalar, node_vector), (edge_scalar, edge_vector))
                   pooled features with shapes:
                   node_scalar: [n_nodes, d_s]
                   node_vector: [n_nodes, d_v, 3]
                   edge_scalar: [n_edges, d_se]
                   edge_vector: [n_edges, d_ve, 3]
        """
        if mask_confs.size(1) == 1:
            # Number of conformations is 1, no need to pool
            return (h_V[0][:, 0], h_V[1][:, 0]), (h_E[0][:, 0], h_E[1][:, 0])
        
        # Extract dimensions
        n_nodes, n_conf, d_s = h_V[0].shape
        _, _, d_v, _ = h_V[1].shape
        n_edges = h_E[0].shape[0]
        d_se = h_E[0].shape[2]
        d_ve = h_E[1].shape[2]
        device = h_V[0].device
        
        # ------ Process node features with cross-attention ------
        # Create mask for attention (1 for valid confs, 0 for invalid)
        # Shape: [n_nodes, n_conf, n_conf]
        node_attention_mask = mask_confs.unsqueeze(2) & mask_confs.unsqueeze(1)
        
        # Apply mask by setting invalid positions to -1e9 (will be ~0 after softmax)
        attention_bias = torch.zeros(n_nodes, n_conf, n_conf, device=device)
        attention_bias = attention_bias.masked_fill(~node_attention_mask, -1e9)
        
        # For each node, compute self-attention across conformations
        node_s = h_V[0]  # [n_nodes, n_conf, d_s]
        node_v = h_V[1]  # [n_nodes, n_conf, d_v, 3]
        
        # Project scalar features to queries, keys, values
        node_s_queries = self.conf_attn_projections['node_q'](node_s)  # [n_nodes, n_conf, d_s]
        node_s_keys = self.conf_attn_projections['node_k'](node_s)     # [n_nodes, n_conf, d_s]
        node_s_values = self.conf_attn_projections['node_v_scalar'](node_s)  # [n_nodes, n_conf, d_s]
        
        # Compute vector magnitudes for attention
        # [n_nodes, n_conf, d_v, 3] -> [n_nodes, n_conf, d_v]
        node_v_norms = torch.norm(node_v, dim=3)
        
        # Compute combined attention scores using both scalar and vector features
        # First, scalar-scalar attention: [n_nodes, n_conf, n_conf]
        scalar_scores = torch.bmm(
            node_s_queries.view(n_nodes * n_conf, 1, d_s),
            node_s_keys.view(n_nodes * n_conf, d_s, 1)
        ).view(n_nodes, n_conf, n_conf) / self.node_scale
        
        # Second, vector-vector attention (using vector norms)
        # [n_nodes, n_conf, d_v] -> [n_nodes * n_conf, d_v]
        node_v_flat = node_v_norms.view(n_nodes * n_conf, d_v)
        
        # Vector features attention (using dot product of vector norms)
        # [n_nodes * n_conf, d_v] @ [n_nodes * n_conf, d_v]ᵀ -> [n_nodes * n_conf, n_nodes * n_conf]
        vector_scores = torch.mm(node_v_flat, node_v_flat.t())
        
        # Reshape to [n_nodes, n_conf, n_nodes, n_conf]
        vector_scores = vector_scores.view(n_nodes, n_conf, n_nodes, n_conf)
        
        # Extract only the relevant connections (between same node, different conformations)
        # [n_nodes, n_conf, n_conf]
        vector_scores = torch.diagonal(vector_scores, dim1=0, dim2=2).permute(2, 0, 1)
        vector_scores = vector_scores / torch.sqrt(torch.tensor(d_v, dtype=torch.float32, device=device))
        
        # Combine scalar and vector attention scores
        node_scores = scalar_scores + vector_scores
        
        # Add the attention mask bias and apply softmax
        node_scores = node_scores + attention_bias
        node_attn_weights = F.softmax(node_scores, dim=-1)  # [n_nodes, n_conf, n_conf]
        
        # Apply dropout to attention weights
        node_attn_weights = self.attn_dropout(node_attn_weights)
        
        # Apply attention weights to scalar values
        # [n_nodes, n_conf, n_conf] @ [n_nodes, n_conf, d_s] -> [n_nodes, n_conf, d_s]
        node_s_pooled = torch.bmm(
            node_attn_weights,
            node_s_values
        )  # [n_nodes, n_conf, d_s]
        
        # Take the first token from the attended output
        node_s_pooled = node_s_pooled[:, 0, :]  # [n_nodes, d_s]
        
        # Apply output projection
        node_s_pooled = self.conf_attn_projections['node_out'](node_s_pooled)  # [n_nodes, d_s]
        
        # For vector features, apply the same attention weights
        node_v_pooled = torch.zeros(n_nodes, d_v, 3, device=device)
        
        # Apply attention to vector features directly
        for i in range(3):  # Handle X, Y, Z coordinates separately
            node_v_i = node_v[:, :, :, i]  # [n_nodes, n_conf, d_v]
            # Apply attention weights to vector features
            # [n_nodes, n_conf, n_conf] @ [n_nodes, n_conf, d_v] -> [n_nodes, n_conf, d_v]
            node_v_pooled_i = torch.bmm(node_attn_weights, node_v_i)  # [n_nodes, n_conf, d_v]
            node_v_pooled[:, :, i] = node_v_pooled_i[:, 0, :]  # Take first token
        
        # ------ Process edge features with cross-attention (similar approach) ------
        # For edges, use the same attention mechanism but with edge features
        edge_src = edge_index[0]  # Source nodes for each edge
        edge_attention_mask = node_attention_mask[edge_src]  # [n_edges, n_conf, n_conf]
        
        # Create edge attention bias
        edge_attention_bias = torch.zeros(n_edges, n_conf, n_conf, device=device)
        edge_attention_bias = edge_attention_bias.masked_fill(~edge_attention_mask, -1e9)
        
        # For each edge, compute cross-attention across conformations
        edge_s = h_E[0]  # [n_edges, n_conf, d_se]
        edge_v = h_E[1]  # [n_edges, n_conf, d_ve, 3]
        
        # Project scalar features
        edge_s_queries = self.conf_attn_projections['edge_q'](edge_s)  # [n_edges, n_conf, d_se]
        edge_s_keys = self.conf_attn_projections['edge_k'](edge_s)     # [n_edges, n_conf, d_se]
        edge_s_values = self.conf_attn_projections['edge_v_scalar'](edge_s)  # [n_edges, n_conf, d_se]
        
        # Compute vector magnitudes for attention
        # [n_edges, n_conf, d_ve, 3] -> [n_edges, n_conf, d_ve]
        edge_v_norms = torch.norm(edge_v, dim=3)
        
        # Compute scalar-scalar attention: [n_edges, n_conf, n_conf]
        edge_scalar_scores = torch.bmm(
            edge_s_queries.view(n_edges * n_conf, 1, d_se),
            edge_s_keys.view(n_edges * n_conf, d_se, 1)
        ).view(n_edges, n_conf, n_conf) / self.edge_scale
        
        # Compute vector-vector attention
        # [n_edges, n_conf, d_ve] -> [n_edges * n_conf, d_ve]
        edge_v_flat = edge_v_norms.view(n_edges * n_conf, d_ve)
        
        # Vector features attention (using dot product of vector norms)
        # For edges, we need to be careful about memory usage, so process in chunks if needed
        if n_edges > 10000:
            # Process in chunks to avoid OOM
            chunk_size = 5000
            edge_vector_scores = torch.zeros(n_edges, n_conf, n_conf, device=device)
            
            for i in range(0, n_edges, chunk_size):
                end_idx = min(i + chunk_size, n_edges)
                edge_v_chunk = edge_v_flat[i*n_conf:(end_idx)*n_conf]  # [chunk_size*n_conf, d_ve]
                
                # Compute chunk scores
                chunk_scores = torch.mm(edge_v_chunk, edge_v_chunk.t())  # [chunk_size*n_conf, chunk_size*n_conf]
                
                # Reshape to [chunk_size, n_conf, chunk_size, n_conf]
                chunk_scores = chunk_scores.view(end_idx-i, n_conf, end_idx-i, n_conf)
                
                # Extract diagonal elements
                for j in range(end_idx-i):
                    edge_vector_scores[i+j] = chunk_scores[j, :, j, :]
        else:
            # If dataset is small enough, compute all at once
            edge_vector_scores = torch.mm(edge_v_flat, edge_v_flat.t())
            
            # Reshape to [n_edges, n_conf, n_edges, n_conf]
            edge_vector_scores = edge_vector_scores.view(n_edges, n_conf, n_edges, n_conf)
            
            # Extract only the relevant connections (between same edge, different conformations)
            # [n_edges, n_conf, n_conf]
            edge_vector_scores = torch.diagonal(edge_vector_scores, dim1=0, dim2=2).permute(2, 0, 1)
        
        # Scale the vector scores
        edge_vector_scores = edge_vector_scores / torch.sqrt(torch.tensor(d_ve, dtype=torch.float32, device=device))
        
        # Combine scalar and vector attention scores
        edge_scores = edge_scalar_scores + edge_vector_scores
        
        # Add the attention mask bias and apply softmax
        edge_scores = edge_scores + edge_attention_bias
        edge_attn_weights = F.softmax(edge_scores, dim=-1)  # [n_edges, n_conf, n_conf]
        
        # Apply dropout to attention weights
        edge_attn_weights = self.attn_dropout(edge_attn_weights)
        
        # Apply attention weights to values
        # [n_edges, n_conf, n_conf] @ [n_edges, n_conf, d_se] -> [n_edges, n_conf, d_se]
        edge_s_pooled = torch.bmm(
            edge_attn_weights,
            edge_s_values
        )  # [n_edges, n_conf, d_se]
        
        # Take the first token from the attended output
        edge_s_pooled = edge_s_pooled[:, 0, :]  # [n_edges, d_se]
        
        # Apply output projection
        edge_s_pooled = self.conf_attn_projections['edge_out'](edge_s_pooled)  # [n_edges, d_se]
        
        # For vector features, apply the same attention weights
        edge_v_pooled = torch.zeros(n_edges, d_ve, 3, device=device)
        
        # Reshape for matrix multiplication and then apply attention
        for i in range(3):  # Handle X, Y, Z coordinates separately
            edge_v_i = edge_v[:, :, :, i]  # [n_edges, n_conf, d_ve]
            # Apply attention weights
            # [n_edges, n_conf, n_conf] @ [n_edges, n_conf, d_ve] -> [n_edges, n_conf, d_ve]
            edge_v_pooled_i = torch.bmm(edge_attn_weights, edge_v_i)  # [n_edges, n_conf, d_ve]
            edge_v_pooled[:, :, i] = edge_v_pooled_i[:, 0, :]  # Take first token
        
        return (node_s_pooled, node_v_pooled), (edge_s_pooled, edge_v_pooled)

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
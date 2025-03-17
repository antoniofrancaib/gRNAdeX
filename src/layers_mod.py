################################################################
# Generalisation of Geometric Vector Perceptron, Jing et al.
# for explicit multi-state biomolecule representation learning.
# Original repository: https://github.com/drorlab/gvp-pytorch
################################################################

import functools
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add


#########################################################################
# Contributions from original GVP repository
#########################################################################

class GraphAttentionLayer(nn.Module):
    '''
    Self-attention layer for updating node scalar features.
    Takes scalar and vector features as input, uses vector norms
    as part of attention computation, and returns updated scalar features
    along with unchanged vector features.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param n_heads: number of attention heads
    :param drop_rate: dropout probability
    :param norm_first: whether to apply normalization before attention
    '''
    def __init__(
            self,
            node_dims,
            n_heads=4,
            drop_rate=0.1,
            norm_first=False,
            residual=True,
        ):
        super(GraphAttentionLayer, self).__init__()
        self.node_dims = node_dims
        self.n_scalar, self.n_vector = node_dims
        self.n_heads = n_heads
        self.head_dim = self.n_scalar // n_heads
        assert self.n_scalar % n_heads == 0, "n_scalar must be divisible by n_heads"
        
        # Input feature dimension after concatenating scalar and vector norm features
        self.combined_dim = self.n_scalar + self.n_vector
        
        # Multi-head attention layers
        self.q_proj = nn.Linear(self.combined_dim, self.n_heads * self.head_dim)
        self.k_proj = nn.Linear(self.combined_dim, self.n_heads * self.head_dim)
        self.v_proj = nn.Linear(self.n_scalar, self.n_heads * self.head_dim)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.n_scalar)
        
        # Normalization and dropout (using ModuleList for consistency with GVPConvLayer)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])
        self.attn_dropout = nn.Dropout(drop_rate)
        
        self.norm_first = norm_first
        self.residual = residual
        
        # Scaling factor for attention scores
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor` for a single conformation
        :return: tuple (s_updated, V) with updated scalar features
        '''
        s, v = x
        batch_size = s.shape[0]
        
        # Prepare input features (concatenate scalar with vector norms)
        if v is not None:
            v_norm = _norm_no_nan(v, axis=-1)  # [n_nodes, n_vector]
            combined_features = torch.cat([s, v_norm], dim=-1)  # [n_nodes, n_scalar + n_vector]
        else:
            combined_features = s
        
        if self.norm_first:
            # Apply normalization before attention if norm_first is True
            s_norm, v_norm = self.norm[0]((s, v))
        else:
            s_norm, v_norm = s, v
            
        # Project to queries, keys, and values
        q = self.q_proj(combined_features)  # [n_nodes, n_heads * head_dim]
        k = self.k_proj(combined_features)  # [n_nodes, n_heads * head_dim]
        v = self.v_proj(s_norm)  # [n_nodes, n_heads * head_dim]
        
        # Reshape to [n_heads, n_nodes, head_dim]
        q = q.view(batch_size, self.n_heads, self.head_dim).transpose(0, 1)
        k = k.view(batch_size, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.view(batch_size, self.n_heads, self.head_dim).transpose(0, 1)
        
        # Compute attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [n_heads, n_nodes, n_nodes]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.bmm(attn_probs, v)  # [n_heads, n_nodes, head_dim]
        
        # Reshape back to [n_nodes, n_scalar]
        attn_output = attn_output.transpose(0, 1).contiguous().view(batch_size, -1)
        attn_output = self.o_proj(attn_output)  # [n_nodes, n_scalar]
        
        # Apply first dropout
        attn_output = self.dropout[0](attn_output)
        
        # First residual connection
        if self.residual:
            s_out = tuple_sum((s, None), (attn_output, None))[0]
        else:
            s_out = attn_output
        
        # Apply normalization if not norm_first
        if not self.norm_first:
            s_out, _ = self.norm[1]((s_out, v))
        
        return (s_out, v)

class MultiAttentiveGVPLayer(nn.Module):
    '''
    A hybrid layer that combines GVP-based message passing with self-attention
    for processing multiple conformations. 
    
    For each conformation:
    1. Uses MultiGVPConvLayer for geometric message passing
    2. Uses GraphAttentionLayer for self-attention on node features
    3. Fuses the outputs of both branches
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs in message function
    :param n_feedforward: number of GVPs in feedforward function
    :param n_heads: number of attention heads
    :param drop_rate: dropout probability
    :param activations: tuple of activation functions for GVPs
    :param vector_gate: whether to use vector gating in GVPs
    :param residual: whether to use residual connections
    :param norm_first: whether to apply normalization before attention/convolution
    :param fusion_weight: weight for combining GVP and attention outputs (0-1)
    '''
    def __init__(
            self,
            node_dims,
            edge_dims,
            n_message=3,
            n_feedforward=2,
            n_heads=4,
            drop_rate=0.1,
            activations=(F.silu, torch.sigmoid),
            vector_gate=True,
            residual=True,
            norm_first=False,
            fusion_weight=0.5,
        ):
        super(MultiAttentiveGVPLayer, self).__init__()
        
        # GVP message passing branch
        self.gvp_branch = MultiGVPConvLayer(
            node_dims=node_dims,
            edge_dims=edge_dims,
            n_message=n_message,
            n_feedforward=n_feedforward,
            drop_rate=drop_rate,
            activations=activations,
            vector_gate=vector_gate,
            residual=residual,
            norm_first=norm_first
        )
        
        # Attention branch
        self.attention_layer = GraphAttentionLayer(
            node_dims=node_dims,
            n_heads=n_heads,
            drop_rate=drop_rate,
            norm_first=norm_first,
            residual=residual
        )
        
        # Normalization and dropout for the combined output
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])
        
        # Layer parameters
        self.fusion_weight = fusion_weight
        self.residual = residual
        self.norm_first = norm_first
        
    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
                s: [n_nodes, n_conf, d_scalar]
                V: [n_nodes, n_conf, d_vector, 3]
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :return: tuple (s_updated, V_updated) with updated features
        '''
        # Apply first normalization if norm_first
        if self.norm_first:
            x_norm = self.norm[0](x)
            # 1. Process with GVP message passing branch
            gvp_s, gvp_v = self.gvp_branch(x_norm, edge_index, edge_attr)
        else:
            # 1. Process with GVP message passing branch
            gvp_s, gvp_v = self.gvp_branch(x, edge_index, edge_attr)
        
        # 2. Process each conformation with attention branch
        s, v = x if self.norm_first else x
        n_nodes, n_conf, d_scalar = s.shape
        _, _, d_vector, _ = v.shape
        
        attn_s = torch.zeros_like(s)
        
        for i in range(n_conf):
            # Extract features for current conformation
            s_i = s[:, i, :]  # [n_nodes, d_scalar]
            v_i = v[:, i, :, :]  # [n_nodes, d_vector, 3]
            
            # Process with attention layer
            s_i_updated, _ = self.attention_layer((s_i, v_i))
            
            # Store updated scalar features
            attn_s[:, i, :] = s_i_updated
        
        # 3. Fuse outputs from both branches
        combined_s = (1 - self.fusion_weight) * gvp_s + self.fusion_weight * attn_s
        combined_v = gvp_v  # Keep vector features from GVP branch
        
        # Apply dropout to the combined outputs
        combined_s = self.dropout[0](combined_s)
        combined_v = self.dropout[0](combined_v)
        
        # Apply residual connection
        if self.residual:
            result = tuple_sum(x, (combined_s, combined_v))
        else:
            result = (combined_s, combined_v)
        
        # Apply second normalization if not norm_first
        if not self.norm_first:
            result = self.norm[1](result)
        
        return result
    
class GeometricMultiHeadAttention(nn.Module):
    """
    Graph Attention Layer for operating on both scalar and vector node features.
    
    Implements multi-head attention as described in:
    "Graph Attention Networks" (Veličković et al., ICLR 2018)
    
    This version incorporates vector norms concatenated with scalar features 
    when computing attention weights.
    
    Args:
        node_h_dim (int): Dimension of scalar node features
        vector_h_dim (int): Dimension of vector node features (set to 0 if no vectors)
        n_heads (int): Number of attention heads
        dropout (float): Dropout probability for attention weights
        concat (bool): Whether to concatenate outputs from different heads
                       or average them
    """
    def __init__(self, node_h_dim, vector_h_dim=0, n_heads=4, dropout=0.1, concat=False):
        super().__init__()
        
        self.node_h_dim = node_h_dim
        self.vector_h_dim = vector_h_dim
        self.n_heads = n_heads
        self.concat = concat
        
        # Combined dimension for attention calculation (scalar + vector norms)
        self.combined_dim = node_h_dim + (vector_h_dim if vector_h_dim > 0 else 0)
        
        # Dimension of each attention head
        self.head_dim = self.combined_dim
        
        # Query, key, value projections (one per head)
        # These now operate on combined features (scalar + vector norms)
        self.W_Q = nn.Linear(self.combined_dim, n_heads * self.head_dim)
        self.W_K = nn.Linear(self.combined_dim, n_heads * self.head_dim)
        self.W_V = nn.Linear(node_h_dim, n_heads * node_h_dim)  # Value still uses only scalars
        
        # Output projection
        if concat:
            # If concatenating, the output dim is n_heads * head_dim
            self.W_O = nn.Linear(n_heads * node_h_dim, node_h_dim)
        else:
            # If averaging, the output dim is head_dim
            self.W_O = nn.Linear(node_h_dim, node_h_dim)
            
        # Attention dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for dot product attention
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        """
        Forward pass of the attention layer with vector norm incorporation.
        
        Args:
            x (tuple): (node_s, node_v) tuple of node features
                        node_s has shape [n_nodes, d_s]
                        node_v has shape [n_nodes, d_v, 3] or None
        
        Returns:
            tuple: Updated (node_s, node_v) tuple after attention
                  node_v is passed through unchanged
        """
        s, v = x  # Unpack node features
        
        # Shape info
        n_nodes = s.shape[0]  # Number of nodes
        
        # Create combined features with vector norms
        if v is not None and self.vector_h_dim > 0:
            # Calculate vector norms [n_nodes, d_v]
            v_norms = torch.norm(v, dim=2)
            
            # Concatenate scalar features with vector norms [n_nodes, d_s + d_v]
            combined_features = torch.cat([s, v_norms], dim=1)
        else:
            combined_features = s
        
        # Linear projections for query, key using combined features
        # [n_nodes, combined_dim] -> [n_nodes, n_heads * head_dim]
        Q = self.W_Q(combined_features)
        K = self.W_K(combined_features)
        
        # For values, we still use only scalar features
        # [n_nodes, d_s] -> [n_nodes, n_heads * node_h_dim]
        V = self.W_V(s)
        
        # Reshape for multi-head attention
        # [n_nodes, n_heads * head_dim] -> [n_nodes, n_heads, head_dim]
        Q = Q.view(n_nodes, self.n_heads, self.head_dim)
        K = K.view(n_nodes, self.n_heads, self.head_dim)
        V = V.view(n_nodes, self.n_heads, self.node_h_dim)
        
        # Transpose for batch matrix multiply
        # [n_nodes, n_heads, dim] -> [n_heads, n_nodes, dim]
        Q = Q.permute(1, 0, 2)
        K = K.permute(1, 0, 2)
        V = V.permute(1, 0, 2)
        
        # Scaled dot-product attention
        # [n_heads, n_nodes, head_dim] @ [n_heads, head_dim, n_nodes] -> [n_heads, n_nodes, n_nodes]
        scores = torch.bmm(Q, K.transpose(-2, -1)) * self.scale
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        # [n_heads, n_nodes, n_nodes] @ [n_heads, n_nodes, node_h_dim] -> [n_heads, n_nodes, node_h_dim]
        h_prime = torch.bmm(attn_weights, V)
        
        # Transpose back and reshape
        # [n_heads, n_nodes, node_h_dim] -> [n_nodes, n_heads, node_h_dim]
        h_prime = h_prime.permute(1, 0, 2)
        
        # Final output projection
        if self.concat:
            # Concatenate heads: [n_nodes, n_heads, node_h_dim] -> [n_nodes, n_heads * node_h_dim]
            h_prime = h_prime.reshape(n_nodes, -1)
            s_out = self.W_O(h_prime)  # [n_nodes, node_h_dim]
        else:
            # Average across heads: [n_nodes, n_heads, node_h_dim] -> [n_nodes, node_h_dim]
            h_prime = h_prime.mean(dim=1)
            s_out = self.W_O(h_prime)  # [n_nodes, node_h_dim]
        
        return (s_out, v)  # Return updated scalars and original vectors

class MultiConformationEncoderLayer(nn.Module):
    """
    Hybrid layer that combines MultiGVPConv with graph attention.
    
    This layer applies both geometric vector perceptron convolution and
    graph attention to handle multiple RNA conformations. The outputs
    are combined with equal weights.
    
    Args:
        node_dims (tuple): (scalar_dim, vector_dim) for node features
        edge_dims (tuple): (scalar_dim, vector_dim) for edge features
        drop_rate (float): Dropout rate for regular dropout
        activations (tuple): Activation functions for scalar and vector features
        vector_gate (bool): Whether to use vector gating
        norm_first (bool): Whether to apply normalization before or after
        n_heads (int): Number of attention heads
        attention_dropout (float): Dropout rate for attention weights
    """
    def __init__(
        self, 
        node_dims, 
        edge_dims, 
        drop_rate=0.1,
        activations=(F.silu, None),
        vector_gate=True,
        norm_first=False, 
        n_heads=4,
        attention_dropout=0.1
    ):
        super().__init__()
        
        # Initialize the Geometric Vector Perceptron convolution layer
        # using MultiGVPConv instead of GVPConv to properly handle multiple conformations
        self.conv = MultiGVPConv(  # Changed from GVPConv to MultiGVPConv
            in_dims = node_dims,
            out_dims = node_dims,
            edge_dims = edge_dims,
            n_layers = 3,
            vector_gate = vector_gate,
            activations = activations
        )
        
        # Attention layer for operating on nodes within each conformation
        # Now passing vector dimension to incorporate vector norms in attention
        self.attention = GeometricMultiHeadAttention(  # Updated class name
            node_h_dim = node_dims[0],  # Scalar feature dimension
            vector_h_dim = node_dims[1],  # Vector feature dimension
            n_heads = n_heads,
            dropout = attention_dropout,
            concat = False  # Use averaging instead of concatenation to avoid dimension issues
        )
        
        # Normalization layers - Use custom LayerNorm that handles vector features correctly
        self.norm_first = norm_first
        self.norm = LayerNorm(node_dims)  # Use the custom LayerNorm class
            
        # Dropout for regularization
        self.dropout = Dropout(drop_rate)  # Use the custom Dropout class
        
    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the hybrid layer.
        
        Args:
            x (tuple): (node_s, node_v) tuple of node features
                      node_s has shape [n_nodes, n_conf, d_s]
                      node_v has shape [n_nodes, n_conf, d_v, 3]
            edge_index (torch.Tensor): Edge indices [2, n_edges]
            edge_attr (tuple): (edge_s, edge_v) tuple of edge features
                              edge_s has shape [n_edges, n_conf, d_se]
                              edge_v has shape [n_edges, n_conf, d_ve, 3]
        
        Returns:
            tuple: Updated (node_s, node_v) tuple after convolution and attention
        """
        s, v = x  # Unpack node features
        
        # Initialize outputs to be the same as inputs (for residual connection)
        out_s, out_v = s, v
        
        # Apply layer normalization before operations if norm_first is True
        if self.norm_first:
            s, v = self.norm((s, v))
        
        # Step 1: Apply MultiGVPConv for message passing
        # This operates on multi-conformation data directly
        conv_s, conv_v = self.conv((s, v), edge_index, edge_attr)
        
        # Step 2: Apply graph attention within each conformation
        # We need to process each conformation separately
        n_nodes, n_conf, d_s = s.shape
        
        # Initialize tensors to store attention outputs for each conformation
        attn_s = torch.zeros_like(s)  # [n_nodes, n_conf, d_s]
        
        # Process each conformation separately
        for conf_idx in range(n_conf):
            # Extract features for this conformation
            s_conf = s[:, conf_idx]  # [n_nodes, d_s]
            v_conf = v[:, conf_idx] if v is not None else None  # [n_nodes, d_v, 3]
            
            # Apply attention to this conformation - now passing both scalar and vector features
            attn_s_conf, _ = self.attention((s_conf, v_conf))  # [n_nodes, d_s]
            
            # Store the result
            attn_s[:, conf_idx] = attn_s_conf
        
        # Combine convolution and attention outputs with equal weights
        combined_s = 0.5 * conv_s + 0.5 * attn_s  # [n_nodes, n_conf, d_s]
        combined_v = conv_v  # [n_nodes, n_conf, d_v, 3] (attention doesn't modify vectors)
        
        # Apply dropout
        combined_s, combined_v = self.dropout((combined_s, combined_v))
        
        # Residual connection
        out_s = out_s + combined_s
        if out_v is not None and combined_v is not None:
            out_v = out_v + combined_v
        
        # Apply layer normalization after operations if norm_first is False
        if not self.norm_first:
            out_s, out_v = self.norm((out_s, out_v))
        
        return (out_s, out_v)

class GlobalAttention(nn.Module):
    '''
    Global attention module that performs full self-attention over all nodes.
    
    This module applies standard transformer attention to all nodes, ignoring
    any graph structure. It supports both self-attention and cross-attention modes.
    
    Args:
        combined_dim (int): dimension of the node features
        n_heads (int): number of attention heads
        dropout (float): dropout rate for attention
    '''
    def __init__(self, combined_dim, n_heads=4, dropout=0.1):
        super(GlobalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=combined_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, query_features, key_value_features=None):
        '''
        Apply global attention between all nodes.
        
        Args:
            query_features: tensor of shape [n_nodes, d]
            key_value_features: tensor of shape [n_nodes, d] for cross-attention, None for self-attention
            
        Returns:
            tensor of shape [n_nodes, d] with attention outputs
        '''
        # Default to self-attention if no separate key/value features provided
        if key_value_features is None:
            key_value_features = query_features
        
        # Perform standard attention over all nodes
        attn_output, _ = self.attention(
            query=query_features.unsqueeze(0),
            key=key_value_features.unsqueeze(0),
            value=key_value_features.unsqueeze(0),
            need_weights=False
        )
        
        return attn_output.squeeze(0)

class GeometricAttentionDecoderLayer(nn.Module):
    '''
    Attention-based decoder layer that uses global attention for processing nodes.
    This layer applies multi-head attention across all nodes while incorporating
    geometric vector features through vector norms.
    
    Note: This layer ignores graph structure and applies attention globally to all nodes.
    
    Args:
        node_dims (tuple): Node dimensions (scalar_dim, vector_dim)
        edge_dims (tuple): Edge dimensions (scalar_dim, vector_dim) - kept for API compatibility
        n_heads (int): Number of attention heads
        dropout (float): Dropout rate
        norm_first (bool): Whether to apply normalization before or after attention
        autoregressive (bool): Whether to use cross-attention with encoder features
        residual (bool): Whether to use residual connections
    '''
    def __init__(
            self,
            node_dims,
            edge_dims,
            n_heads=4,
            dropout=0.1,
            norm_first=True,
            autoregressive=True,
            activations=(F.silu, None),  # Kept for API compatibility
            vector_gate=True,  # Kept for API compatibility
            residual=True,
            **kwargs  # To handle any extra args passed from original GVPConvLayer
        ):
        super(GeometricAttentionDecoderLayer, self).__init__()
        
        # Save dimensions and settings
        self.scalar_dim, self.vector_dim = node_dims
        self.combined_dim = self.scalar_dim + self.vector_dim
        self.autoregressive = autoregressive
        self.residual = residual
        self.norm_first = norm_first
        self.dropout_rate = dropout
        
        # Normalization layers for scalar and vector features
        self.scalar_norm1 = nn.LayerNorm(self.scalar_dim)
        self.scalar_norm2 = nn.LayerNorm(self.scalar_dim)
        
        # Attention module and projection
        self.attention = GlobalAttention(self.combined_dim, n_heads, dropout)
        self.output_projection = nn.Linear(self.combined_dim, self.scalar_dim)
        self.dropout_attn = nn.Dropout(dropout)
        
        # Feed-forward networks
        self.scalar_ff = nn.Sequential(
            nn.Linear(self.scalar_dim, self.scalar_dim * 8),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.scalar_dim * 8, self.scalar_dim)
        )
        
        self.vector_ff = nn.Sequential(
            nn.Linear(self.vector_dim, self.vector_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.vector_dim * 4, self.vector_dim)
        )
        
        self.dropout_ff_scalar = nn.Dropout(dropout)
        self.dropout_ff_vector = nn.Dropout(dropout)
    
    def _apply_node_mask(self, features, node_mask):
        '''
        Apply node mask to features.
        
        Args:
            features (tuple): (scalar_features, vector_features) tuple
            node_mask (torch.Tensor): Boolean mask for nodes to update
            
        Returns:
            tuple: Masked scalar and vector features
        '''
        if node_mask is None:
            return features
        
        scalar, vector = features
        return scalar[node_mask], vector[node_mask]
    
    def _apply_norm_to_vectors(self, vectors):
        '''
        Apply normalization to vector features.
        
        Args:
            vectors (torch.Tensor): Vector features of shape [n_nodes, d_v, 3]
            
        Returns:
            torch.Tensor: Normalized vector features
        '''
        # Compute vector norms (square and mean across vector dimension)
        # Shape: [n_nodes, d_v, 3] -> [n_nodes, d_v] -> [n_nodes, 1, 1]
        v_norm = torch.sqrt(
            torch.mean(
                torch.sum(vectors ** 2, dim=-1, keepdim=False), 
                dim=-1, keepdim=True
            ).unsqueeze(-1).clamp(min=1e-8)
        )
        
        # Normalize vectors
        # Shape: [n_nodes, d_v, 3]
        return vectors / v_norm
    
    def _apply_norm_pair(self, features, norm_idx=0):
        '''
        Apply normalization to scalar and vector features.
        
        Args:
            features (tuple): (scalar_features, vector_features) tuple
            norm_idx (int): 0 for first norm layer, 1 for second
            
        Returns:
            tuple: Normalized scalar and vector features
        '''
        scalar, vector = features
        
        # Apply normalization to scalar features
        norm_layer = self.scalar_norm1 if norm_idx == 0 else self.scalar_norm2
        scalar_norm = norm_layer(scalar)
        
        # Apply normalization to vector features if they exist
        vector_norm = self._apply_norm_to_vectors(vector) if vector is not None else None
        
        return scalar_norm, vector_norm
    
    def _enhance_features(self, features):
        '''
        Create enhanced features by combining scalar features with vector norms.
        
        Args:
            features (tuple): (scalar_features, vector_features) tuple where
                             scalar_features: [n_nodes, d_s]
                             vector_features: [n_nodes, d_v, 3]
                             
        Returns:
            torch.Tensor: Enhanced features of shape [n_nodes, d_s + d_v]
        '''
        scalar, vector = features
        
        # Calculate vector norms
        # Shape: [n_nodes, d_v, 3] -> [n_nodes, d_v]
        vector_norms = torch.norm(vector, dim=-1)
        
        # Concatenate scalar features with vector norms
        # Shape: [n_nodes, d_s + d_v]
        return torch.cat([scalar, vector_norms], dim=-1)
    
    def _apply_attention(self, query_features, key_value_features=None):
        '''
        Apply attention and project the output.
        
        Args:
            query_features (torch.Tensor): Query features of shape [n_nodes, d_s + d_v]
            key_value_features (torch.Tensor, optional): Key/value features for cross-attention
            
        Returns:
            torch.Tensor: Attention output of shape [n_nodes, d_s]
        '''
        # Apply attention
        # Shape: [n_nodes, d_s + d_v] -> [n_nodes, d_s + d_v]
        attn_output = self.attention(
            query_features=query_features,
            key_value_features=key_value_features
        )
        
        # Project attention output to scalar dimension and apply dropout
        # Shape: [n_nodes, d_s + d_v] -> [n_nodes, d_s]
        return self.dropout_attn(self.output_projection(attn_output))
    
    def _apply_residual(self, input_features, update_features):
        '''
        Apply residual connection.
        
        Args:
            input_features (tuple): Original (scalar, vector) tuple
            update_features (tuple): Update (scalar, vector) tuple
            
        Returns:
            tuple: Updated (scalar, vector) tuple after residual connection
        '''
        if not self.residual:
            return update_features
            
        input_scalar, input_vector = input_features
        update_scalar, update_vector = update_features
        
        # Apply residual connections
        output_scalar = input_scalar + update_scalar
        output_vector = input_vector + update_vector if update_vector is not None else input_vector
        
        return output_scalar, output_vector
    
    def _apply_feed_forward(self, features):
        '''
        Apply feed-forward networks to scalar and vector features.
        
        Args:
            features (tuple): (scalar_features, vector_features) tuple
            
        Returns:
            tuple: Updated scalar and vector features after feed-forward
        '''
        scalar, vector = features
        
        # Apply feed-forward to scalar features and dropout
        # Shape: [n_nodes, d_s] -> [n_nodes, d_s]
        scalar_ff = self.dropout_ff_scalar(self.scalar_ff(scalar))
        
        # Apply feed-forward to vector features and dropout
        if vector is not None:
            # Shape: [n_nodes, d_v, 3] -> [n_nodes * d_v, 3] -> [n_nodes, d_v, 3]
            vector_shape = vector.shape
            vector_flat = vector.reshape(-1, self.vector_dim)
            vector_ff = self.vector_ff(vector_flat).reshape(vector_shape)
            vector_ff = self.dropout_ff_vector(vector_ff)
        else:
            vector_ff = None
            
        return scalar_ff, vector_ff
    
    def _restore_node_mask(self, original_features, updated_features, node_mask):
        '''
        Restore original features for nodes that were not in the mask.
        
        Args:
            original_features (tuple): Original (scalar, vector) tuple for all nodes
            updated_features (tuple): Updated (scalar, vector) tuple for masked nodes
            node_mask (torch.Tensor): Boolean mask indicating which nodes were updated
            
        Returns:
            tuple: Complete (scalar, vector) tuple with updates applied only to masked nodes
        '''
        if node_mask is None:
            return updated_features
            
        orig_scalar, orig_vector = original_features
        updated_scalar, updated_vector = updated_features
        
        # Clone original features
        result_scalar = orig_scalar.clone()
        result_vector = orig_vector.clone()
        
        # Update only the masked nodes
        result_scalar[node_mask] = updated_scalar
        result_vector[node_mask] = updated_vector
        
        return result_scalar, result_vector

    def forward(self, x, edge_index, edge_attr=None, autoregressive_x=None, node_mask=None):
        '''
        Forward pass of the GeometricAttentionDecoderLayer.
        
        Args:
            x (tuple): Node features (scalar_features, vector_features) where
                     scalar_features: [n_nodes, d_s]
                     vector_features: [n_nodes, d_v, 3]
            edge_index (torch.Tensor): Edge indices of shape [2, n_edges] (unused)
            edge_attr (tuple, optional): Edge features (unused)
            autoregressive_x (tuple, optional): Encoder features for cross-attention
            node_mask (torch.Tensor, optional): Boolean mask for nodes to update
            
        Returns:
            tuple: Updated node features (scalar_features, vector_features)
        '''
        # Save original features for node masking
        original_features = x
        
        # Step 1: Apply node mask if provided
        scalar, vector = self._apply_node_mask(x, node_mask)
        
        # Step 2: Process encoder features if in autoregressive mode
        encoder_features = None
        if autoregressive_x is not None:
            encoder_features = self._apply_node_mask(autoregressive_x, node_mask)
        
        # Step 3: Apply first normalization if norm_first is True
        if self.norm_first:
            norm_features = self._apply_norm_pair((scalar, vector), norm_idx=0)
        else:
            norm_features = (scalar, vector)
        
        # Step 4: Prepare features for attention
        # Create enhanced features by combining scalar and vector norms
        enhanced_scalar = self._enhance_features(norm_features)
        
        # Create enhanced encoder features if in autoregressive mode
        enhanced_encoder = None
        if encoder_features is not None:
            enhanced_encoder = self._enhance_features(encoder_features)
        
        # Step 5: Apply attention and projection
        attn_scalar = self._apply_attention(
            query_features=enhanced_scalar,
            key_value_features=enhanced_encoder
        )
        
        # Step 6: First residual connection (only for scalar features)
        scalar_after_attn = scalar + attn_scalar if self.residual else attn_scalar
        features_after_attn = (scalar_after_attn, vector)
        
        # Step 7: Apply normalization before or after the first residual connection
        if self.norm_first:
            ff_input_features = self._apply_norm_pair(features_after_attn, norm_idx=1)
        else:
            ff_input_features = self._apply_norm_pair(features_after_attn, norm_idx=0) if self.residual else features_after_attn
        
        # Step 8: Apply feed-forward networks
        ff_output_features = self._apply_feed_forward(ff_input_features)
        
        # Step 9: Second residual connection
        updated_features = self._apply_residual(features_after_attn, ff_output_features)
        
        # Step 10: Apply final normalization if not norm_first
        if not self.norm_first and self.residual:
            updated_features = self._apply_norm_pair(updated_features, norm_idx=1)
        
        # Step 11: Restore original features for nodes that were not in the mask
        result_features = self._restore_node_mask(original_features, updated_features, node_mask)
        
        return result_features

#########################################################################

class GVPConvLayer(nn.Module):
    '''
    Full graph convolution / message passing layer with 
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward 
    network to node embeddings, and returns updated node embeddings.
    
    To only compute the aggregated messages, see `GVPConv`.
    
    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(
            self, 
            node_dims, 
            edge_dims,
            n_message=3, 
            n_feedforward=2, 
            drop_rate=.1,
            autoregressive=False, 
            activations=(F.silu, torch.sigmoid), 
            vector_gate=True,
            residual=True,
            norm_first=False,
        ):
        
        super(GVPConvLayer, self).__init__()
        self.conv = GVPConv(node_dims, node_dims, edge_dims, n_message,
                           aggr="add" if autoregressive else "mean",
                           activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)
        self.residual = residual
        self.norm_first = norm_first

    def forward(self, x, edge_index, edge_attr,
                autoregressive_x=None, node_mask=None):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`. 
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The current node 
                embeddings `x` will still be the base of the update and the 
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        '''
        
        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)
            
            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )
            
            count = scatter_add(torch.ones_like(dst), dst,
                        dim_size=dh[0].size(0)).clamp(min=1).unsqueeze(-1)
            
            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            if self.norm_first:
                dh = self.conv(self.norm[0](x), edge_index, edge_attr)
            else:
                dh = self.conv(x, edge_index, edge_attr)
        
        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)
        
        if self.norm_first:
            x = tuple_sum(x, self.dropout[0](dh))
            dh = self.ff_func(self.norm[1](x))
            x = tuple_sum(x, self.dropout[1](dh))
        else:
            x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) if self.residual else dh
            dh = self.ff_func(x)
            x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) if self.residual else dh
        
        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x

class GVPConv(MessagePassing):
    '''
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.
    
    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.
    
    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.silu, torch.sigmoid), vector_gate=True):
        super(GVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo)))
            else:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        message = self.propagate(edge_index, 
                    s=x_s, v=x_v.contiguous().view(x_v.shape[0], x_v.shape[1] * 3),
                    edge_attr=edge_attr)
        return _split(message, self.vo) 

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)
  

class MultiGVPConvLayer(nn.Module):
    '''
    GVPConvLayer for handling multiple conformations (encoder-only)
    '''
    def __init__(
            self, 
            node_dims, 
            edge_dims,
            n_message=3, 
            n_feedforward=2, 
            drop_rate=.1,
            activations=(F.silu, torch.sigmoid), 
            vector_gate=True,
            residual=True,
            norm_first=False,
        ):
        super(MultiGVPConvLayer, self).__init__()
        self.conv = MultiGVPConv(node_dims, node_dims, edge_dims, n_message,
                                 aggr="mean", activations=activations, vector_gate=vector_gate)
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims))
        else:
            hid_dims = 4*node_dims[0], 2*node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward-2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)
        self.residual = residual
        self.norm_first = norm_first

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        if self.norm_first:
            dh = self.conv(self.norm[0](x), edge_index, edge_attr)
            x = tuple_sum(x, self.dropout[0](dh))
            dh = self.ff_func(self.norm[1](x))
            x = tuple_sum(x, self.dropout[1](dh))
        else:
            dh = self.conv(x, edge_index, edge_attr)
            x = self.norm[0](tuple_sum(x, self.dropout[0](dh))) if self.residual else dh
            dh = self.ff_func(x)
            x = self.norm[1](tuple_sum(x, self.dropout[1](dh))) if self.residual else dh
        return x

class MultiGVPConv(MessagePassing):
    '''
    GVPConv for handling multiple conformations
    '''
    def __init__(self, in_dims, out_dims, edge_dims,
                 n_layers=3, module_list=None, aggr="mean", 
                 activations=(F.silu, torch.sigmoid), vector_gate=True):
        super(MultiGVPConv, self).__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        
        GVP_ = functools.partial(GVP, 
                activations=activations, vector_gate=vector_gate)
        
        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), 
                        (self.so, self.vo)))
            else:
                module_list.append(
                    GVP_((2*self.si + self.se, 2*self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        '''
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        '''
        x_s, x_v = x
        n_conf = x_s.shape[1]
        
        # x_s: [n_nodes, n_conf, d] -> [n_nodes, n_conf * d]
        x_s = x_s.contiguous().view(x_s.shape[0], x_s.shape[1] * x_s.shape[2])        
        # x_v: [n_nodes, n_conf, d, 3] -> [n_nodes, n_conf * d * 3]
        x_v = x_v.contiguous().view(x_v.shape[0], x_v.shape[1] * x_v.shape[2] * 3)
        
        message = self.propagate(edge_index, s=x_s, v=x_v, edge_attr=edge_attr)
        
        return _split_multi(message, self.so, self.vo, n_conf)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        # [n_nodes, n_conf * d] -> [n_nodes, n_conf, d]
        s_i = s_i.view(s_i.shape[0], s_i.shape[1]//self.si, self.si)
        s_j = s_j.view(s_j.shape[0], s_j.shape[1]//self.si, self.si)
        # [n_nodes, n_conf * d * 3] -> [n_nodes, n_conf, d, 3]
        v_i = v_i.view(v_i.shape[0], v_i.shape[1]//(self.vi * 3), self.vi, 3)
        v_j = v_j.view(v_j.shape[0], v_j.shape[1]//(self.vi * 3), self.vi, 3)

        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge_multi(*message)

#########################################################################

class GVP(nn.Module):
    '''
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.silu, torch.sigmoid), vector_gate=True):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)    
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo: 
                v = self.wv(vh) 
                v = torch.transpose(v, -1, -2)
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s
    
#########################################################################

class _VDropout(nn.Module):
    '''
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    '''
    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        '''
        :param x: `torch.Tensor` corresponding to vector channels
        '''
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device))
        x = mask.unsqueeze(-1) * x / (1 - self.drop_rate)
        return x

class Dropout(nn.Module):
    '''
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)

class LayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims):
        super(LayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn

def tuple_sum(*args):
    '''
    Sums any number of tuples (s, V) elementwise.
    '''
    return tuple(map(sum, zip(*args)))

def tuple_cat(*args, dim=-1):
    '''
    Concatenates any number of tuples (s, V) elementwise.
    
    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    '''
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)

def tuple_index(x, idx):
    '''
    Indexes into a tuple (s, V) along the first dimension.
    
    :param idx: any object which can be used to index into a `torch.Tensor`
    '''
    return x[0][idx], x[1][idx]

def randn(n, dims, device="cpu"):
    '''
    Returns random tuples (s, V) drawn elementwise from a normal distribution.
    
    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)
    
    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    '''
    return torch.randn(n, dims[0], device=device), \
            torch.randn(n, dims[1], 3, device=device)

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

def _split(x, nv):
    '''
    Splits a merged representation of (s, V) back into a tuple. 
    Should be used only with `_merge(s, V)` and only if the tuple 
    representation cannot be used.
    
    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    '''
    s = x[..., :-3 * nv]
    v = x[..., -3 * nv:].contiguous().view(x.shape[0], nv, 3)
    return s, v

def _merge(s, v):
    '''
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    '''
    v = v.contiguous().view(v.shape[0], v.shape[1] * 3)
    return torch.cat([s, v], -1)

def _split_multi(x, ns, nv, n_conf=5):
    '''
    _split for multiple conformers
    '''
    s = x[..., :-3 * nv * n_conf].contiguous().view(x.shape[0], n_conf, ns)
    v = x[..., -3 * nv * n_conf:].contiguous().view(x.shape[0], n_conf, nv, 3)
    return s, v

def _merge_multi(s, v):
    '''
    _merge for multiple conformers
    '''
    # s: [n_nodes, n_conf, d] -> [n_nodes, n_conf * d]
    s = s.contiguous().view(s.shape[0], s.shape[1] * s.shape[2])
    # v: [n_nodes, n_conf, d, 3] -> [n_nodes, n_conf * d * 3]
    v = v.contiguous().view(v.shape[0], v.shape[1] * v.shape[2] * 3)
    return torch.cat([s, v], -1)
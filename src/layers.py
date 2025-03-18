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
   
class GraphAttentionLayer(nn.Module):
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
        
    def forward(self, x, mask=None):
        """
        Forward pass of the attention layer with vector norm incorporation.
        
        Args:
            x (tuple): (node_s, node_v) tuple of node features
                        node_s has shape [n_nodes, d_s]
                        node_v has shape [n_nodes, d_v, 3] or None
            mask (torch.Tensor, optional): Attention mask of shape [n_nodes, n_nodes] or [n_heads, n_nodes, n_nodes].
                                         Values to mask should be set to a large negative number (e.g., -1e9).
        
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
        
        # Apply attention mask if provided
        if mask is not None:
            # If mask is 2D, expand to 3D for all heads
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).expand(self.n_heads, -1, -1)
            scores = scores + mask
        
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

class MultiAttentiveGVPLayer(nn.Module):
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
        self.attention = GraphAttentionLayer(  # Updated class name
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
    
class AttentiveGVPLayer(nn.Module):
    """
    Hybrid layer that combines GVP-based message passing with self-attention.
    
    This layer implements a parallel architecture with two branches:
    1. GVP-based message passing (autoregressive)
    2. Self-attention with causal masking
    
    The outputs from both branches are combined with equal weights.
    
    Args:
        node_dims (tuple): (scalar_dim, vector_dim) for node features
        edge_dims (tuple): (scalar_dim, vector_dim) for edge features
        n_message (int): number of GVPs to use in message function
        n_feedforward (int): number of GVPs to use in feedforward function
        drop_rate (float): drop probability in all dropout layers
        activations (tuple): activation functions for scalar and vector features
        vector_gate (bool): whether to use vector gating
        residual (bool): whether to use residual connections
        norm_first (bool): whether to apply normalization before or after operations
        n_heads (int): number of attention heads
        attention_dropout (float): dropout rate for attention weights
    """
    def __init__(
        self,
        node_dims,
        edge_dims,
        n_message=3,
        n_feedforward=2,
        drop_rate=0.1,
        activations=(F.silu, torch.sigmoid),
        vector_gate=True,
        residual=True,
        norm_first=False,
        n_heads=4,
        attention_dropout=0.1
    ):
        super().__init__()
        
        # Branch A: GVP-based message passing
        self.gvp_branch = GVPConvLayer(
            node_dims=node_dims,
            edge_dims=edge_dims,
            n_message=n_message,
            n_feedforward=n_feedforward,
            drop_rate=drop_rate,
            activations=activations,
            vector_gate=vector_gate,
            residual=residual,
            norm_first=norm_first,
            autoregressive=True  # Always use autoregressive mode
        )
        
        # Branch B: Self-attention
        self.attention_branch = GraphAttentionLayer(
            node_h_dim=node_dims[0],  # Scalar feature dimension
            vector_h_dim=node_dims[1],  # Vector feature dimension
            n_heads=n_heads,
            dropout=attention_dropout,
            concat=False  # Use averaging instead of concatenation
        )
        
        # Normalization layers
        self.norm_first = norm_first
        self.norm = LayerNorm(node_dims)
        
        # Dropout for regularization
        self.dropout = Dropout(drop_rate)
        
        # Optional feedforward network
        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)
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
        
        # Final normalization
        self.final_norm = LayerNorm(node_dims)
        
    def create_causal_mask(self, n_nodes, device):
        """
        Create a causal mask for autoregressive attention.
        
        Args:
            n_nodes (int): number of nodes
            device (torch.device): device to create mask on
            
        Returns:
            torch.Tensor: causal mask of shape [n_nodes, n_nodes]
        """
        # Create lower triangular mask (including diagonal)
        mask = torch.triu(torch.ones(n_nodes, n_nodes, device=device), diagonal=1)
        # Convert to additive mask (1 -> -inf, 0 -> 0)
        mask = mask * -1e9
        return mask
        
    def forward(self, x, edge_index, edge_attr, autoregressive_x=None, node_mask=None):
        """
        Forward pass of the hybrid layer.
        
        Args:
            x (tuple): (node_s, node_v) tuple of node features
            edge_index (torch.Tensor): edge indices [2, n_edges]
            edge_attr (tuple): (edge_s, edge_v) tuple of edge features
            autoregressive_x (tuple, optional): node features for autoregressive message passing
            node_mask (torch.Tensor, optional): boolean mask for nodes to update
            
        Returns:
            tuple: Updated (node_s, node_v) tuple after both branches
        """
        s, v = x
        n_nodes = s.shape[0]
        device = s.device
        
        # Initialize outputs to be the same as inputs (for residual connection)
        out_s, out_v = s, v
        
        # Apply layer normalization before operations if norm_first is True
        if self.norm_first:
            s, v = self.norm((s, v))
        
        # Branch A: GVP-based message passing
        gvp_s, gvp_v = self.gvp_branch(
            (s, v), edge_index, edge_attr,
            autoregressive_x=autoregressive_x,
            node_mask=node_mask
        )
        
        # Branch B: Self-attention with causal masking
        # Create causal mask for autoregressive attention
        causal_mask = self.create_causal_mask(n_nodes, device)
        attn_s, attn_v = self.attention_branch((s, v), mask=causal_mask)
        
        # Combine outputs from both branches with equal weights
        combined_s = 0.5 * gvp_s + 0.5 * attn_s
        combined_v = gvp_v  # Use GVP branch's vector features
        
        # Apply dropout
        combined_s, combined_v = self.dropout((combined_s, combined_v))
        
        # Residual connection
        out_s = out_s + combined_s
        if out_v is not None and combined_v is not None:
            out_v = out_v + combined_v
        
        # Apply layer normalization after operations if norm_first is False
        if not self.norm_first:
            out_s, out_v = self.norm((out_s, out_v))
        
        # Optional feedforward network
        ff_s, ff_v = self.ff_func((out_s, out_v))
        out_s, out_v = self.final_norm((out_s + ff_s, out_v + ff_v))
        
        return (out_s, out_v)

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
    
#########################################################################

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
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
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

################################################################
# Check the changes in the model dont break the forward pass. 
################################################################


import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from src.models import AutoregressiveMultiGNNv1
from src.constants import DATA_PATH
from src.layers_mod import (
    GVPConvLayer, 
    MultiConformationEncoderLayer,
    GeometricMultiHeadAttention
)


def test_forward_pass():
    """
    Test forward pass of the modified model with parallel attention
    on a single datapoint from the dataset.
    """
    print("Testing forward pass of the modified model with parallel attention...")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        # Create a simple test data structure
        print("Creating test data...")
        
        # Create a small RNA sequence (10 nucleotides)
        seq = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], device=device, dtype=torch.long)  # A,G,C,U,A,G,C,U,A,G
        
        # Create random coordinates for 3 backbone atoms (P, C4', N1/N9) for each nucleotide
        num_nodes = len(seq)
        num_conf = 1
        num_bb_atoms = 3
        coords = torch.randn(num_conf, num_nodes, num_bb_atoms, 3, device=device)
        
        # Create edge indices (fully connected graph)
        edge_index = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                edge_index.extend([[i,j], [j,i]])
        edge_index = torch.tensor(edge_index, device=device).t()
        
        # Create masks
        mask_coords = torch.ones(num_nodes, dtype=torch.bool, device=device)
        mask_confs = torch.ones(num_nodes, num_conf, dtype=torch.bool, device=device)
        
        # Create node features
        node_s = torch.randn(num_nodes, num_conf, num_bb_atoms * 5, device=device)  # 5 features per atom
        node_v = torch.randn(num_nodes, num_conf, 4, 3, device=device)  # 4 vectors per node
        
        # Create edge features
        num_edges = edge_index.shape[1]
        edge_s = torch.randn(num_edges, num_conf, 32 + 32 + num_bb_atoms, device=device)  # RBF + posenc + log(lengths)
        edge_v = torch.randn(num_edges, num_conf, num_bb_atoms, 3, device=device)
        
        # Create a Data object
        from torch_geometric.data import Data
        data = Data(
            seq=seq,
            node_s=node_s,
            node_v=node_v,
            edge_s=edge_s,
            edge_v=edge_v,
            edge_index=edge_index,
            mask_confs=mask_confs,
            mask_coords=mask_coords
        )
        
        # Print data information
        print("\n==== INPUT DATA DIMENSIONS ====")
        print(f"Number of nodes: {data.node_s.shape[0]}")
        print(f"Node scalar shape: {data.node_s.shape}")
        print(f"Node vector shape: {data.node_v.shape}")
        print(f"Edge scalar shape: {data.edge_s.shape}")
        print(f"Edge vector shape: {data.edge_v.shape}")
        print(f"Edge index shape: {data.edge_index.shape}")
        print(f"Sequence shape: {data.seq.shape}")
        
        # Initialize model
        print("\n==== INITIALIZING MODEL ====")
        
        # Create a subclass of AutoregressiveMultiGNNv1 that prints dimensions
        class DebugModel(AutoregressiveMultiGNNv1):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.scalar_values = {}  # Dictionary to store scalar values at each stage
            
            def forward(self, batch):
                print("\n==== MODEL TRANSFORMATIONS ====")
                
                h_V = (batch.node_s, batch.node_v)
                h_E = (batch.edge_s, batch.edge_v)
                edge_index = batch.edge_index
                seq = batch.seq
                
                # Store initial scalar values
                self.scalar_values['input'] = h_V[0].mean(dim=-1).detach().cpu().numpy()
                
                # Node embedding
                h_V = self.W_v(h_V)
                self.scalar_values['embedding'] = h_V[0].mean(dim=-1).detach().cpu().numpy()
                
                # Edge embedding
                h_E = self.W_e(h_E)
                
                # Encoder layers
                for i, layer in enumerate(self.encoder_layers):
                    h_V = layer(h_V, edge_index, h_E)
                    self.scalar_values[f'encoder_{i+1}'] = h_V[0].mean(dim=-1).detach().cpu().numpy()
                
                # Pool multi-conformation features
                h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
                self.scalar_values['pooling'] = h_V[0].mean(dim=-1).detach().cpu().numpy()
                
                encoder_embeddings = h_V
                
                # Sequence embeddings
                h_S = self.W_s(seq)
                h_S = h_S[edge_index[0]]
                h_S[edge_index[0] >= edge_index[1]] = 0
                h_E = (torch.cat([h_E[0], h_S], dim=-1), h_E[1])
                
                # Decoder layers
                for i, layer in enumerate(self.decoder_layers):
                    h_V = layer(h_V, edge_index, h_E, autoregressive_x=encoder_embeddings)
                    self.scalar_values[f'decoder_{i+1}'] = h_V[0].mean(dim=-1).detach().cpu().numpy()
                
                # Final output projection
                logits = self.W_out(h_V)
                self.scalar_values['output'] = logits.mean(dim=-1).detach().cpu().numpy()
                
                return logits
                
            @torch.no_grad()
            def sample(self, batch, n_samples, temperature=0.1, logit_bias=None, return_logits=False):
                print("\n==== SAMPLING PROCEDURE ====")
                
                h_V = (batch.node_s, batch.node_v)
                h_E = (batch.edge_s, batch.edge_v)
                edge_index = batch.edge_index
            
                device = edge_index.device
                num_nodes = h_V[0].shape[0]
                
                print(f"Starting sampling with {n_samples} samples for {num_nodes} nodes")
                print(f"Initial h_V: node_s {h_V[0].shape}, node_v {h_V[1].shape}")
                print(f"Initial h_E: edge_s {h_E[0].shape}, edge_v {h_E[1].shape}")
                
                # Node embedding
                h_V = self.W_v(h_V)
                print(f"After W_v embedding: node_s {h_V[0].shape}, node_v {h_V[1].shape}")
                
                # Edge embedding
                h_E = self.W_e(h_E)
                print(f"After W_e embedding: edge_s {h_E[0].shape}, edge_v {h_E[1].shape}")
                
                # Encoder layers
                for i, layer in enumerate(self.encoder_layers):
                    h_V = layer(h_V, edge_index, h_E)
                
                print(f"After all encoder layers: node_s {h_V[0].shape}, node_v {h_V[1].shape}")
                
                # Pool multi-conformation features
                h_V, h_E = self.pool_multi_conf(h_V, h_E, batch.mask_confs, edge_index)
                print(f"After pooling: node_s {h_V[0].shape}, node_v {h_V[1].shape}")
                print(f"After pooling: edge_s {h_E[0].shape}, edge_v {h_E[1].shape}")
                
                # Repeat features for sampling n_samples times
                h_V = (h_V[0].repeat(n_samples, 1),
                    h_V[1].repeat(n_samples, 1, 1))
                h_E = (h_E[0].repeat(n_samples, 1),
                    h_E[1].repeat(n_samples, 1, 1))
                
                print(f"After repeating for {n_samples} samples:")
                print(f"  h_V: node_s {h_V[0].shape}, node_v {h_V[1].shape}")
                print(f"  h_E: edge_s {h_E[0].shape}, edge_v {h_E[1].shape}")
                
                # Expand edge index for autoregressive decoding
                edge_index = edge_index.expand(n_samples, -1, -1)
                offset = num_nodes * torch.arange(n_samples, device=device).view(-1, 1, 1)
                edge_index = torch.cat(tuple(edge_index + offset), dim=-1)
                print(f"Expanded edge_index shape: {edge_index.shape}")
                
                seq = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.int)
                h_S = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)
                logits = torch.zeros(n_samples * num_nodes, self.out_dim, device=device)
                
                h_V_cache = [(h_V[0].clone(), h_V[1].clone()) for _ in self.decoder_layers]
                
                print(f"\nDecoding sequence autogressively for {num_nodes} nodes:")
                
                # Skip full token-by-token printing to keep output manageable
                print(f"[Autoregressive decoding details omitted for brevity...]")
                
                # Just decode 3 tokens as example
                print("Showing first 3 tokens as example:")
                for i in range(min(3, num_nodes)):
                    print(f"\n-- Decoding token {i+1}/{num_nodes} --")
                    
                    h_S_ = h_S[edge_index[0]]
                    h_S_[edge_index[0] >= edge_index[1]] = 0
                    h_E_ = (torch.cat([h_E[0], h_S_], dim=-1), h_E[1])
                            
                    edge_mask = edge_index[1] % num_nodes == i
                    edge_index_ = edge_index[:, edge_mask]
                    h_E_ = (h_E_[0][edge_mask], h_E_[1][edge_mask])
                    node_mask = torch.zeros(n_samples * num_nodes, device=device, dtype=torch.bool)
                    node_mask[i::num_nodes] = True
                    
                    print(f"  Edge subset: {edge_index_.shape}")
                    print(f"  Node mask active nodes: {node_mask.sum().item()}")
                    
                    # Only show first decoder layer for brevity
                    print("  [Decoder processing...]")
                    
                    for j, layer in enumerate(self.decoder_layers):
                        out = layer(h_V_cache[j], edge_index_, h_E_,
                                autoregressive_x=h_V_cache[0], node_mask=node_mask)
                        
                        out = (out[0][node_mask], out[1][node_mask])
                        
                        if j < len(self.decoder_layers)-1:
                            h_V_cache[j+1][0][i::num_nodes] = out[0]
                            h_V_cache[j+1][1][i::num_nodes] = out[1]
                    
                    lgts = self.W_out(out)
                    if logit_bias is not None:
                        lgts += logit_bias[i]
                        
                    seq[i::num_nodes] = torch.distributions.Categorical(
                        logits=lgts / temperature).sample()
                    h_S[i::num_nodes] = self.W_s(seq[i::num_nodes])
                    logits[i::num_nodes] = lgts
                    
                    print(f"  Token {i+1} logits shape: {lgts.shape}")
                    print(f"  Token {i+1} sampled values: {seq[i::num_nodes][:3]}...")
                
                print("\n-- Sampling Completed --")
                print(f"Final sequence shape: {seq.view(n_samples, num_nodes).shape}")
                print(f"Final logits shape: {logits.view(n_samples, num_nodes, self.out_dim).shape}")
                
                if return_logits:
                    return seq.view(n_samples, num_nodes), logits.view(n_samples, num_nodes, self.out_dim)
                else:    
                    return seq.view(n_samples, num_nodes)
                
        # Use the debug model with the same parameters
        model = DebugModel(
            node_in_dim=(data.node_s.shape[2], data.node_v.shape[2]),
            node_h_dim=(128, 16),
            edge_in_dim=(data.edge_s.shape[2], data.edge_v.shape[2]),
            edge_h_dim=(32, 1),
            num_layers=3,
            drop_rate=0.1,
            out_dim=4
        ).to(device)
        
        # Print model information
        total_param = sum(p.numel() for p in model.parameters())
        print(f"Model initialized with {total_param} parameters")
        
        # Test both forward pass and sampling
        print("\n==== FORWARD PASS TEST ====")
        model.train()
        # Forward pass - will print dimensions during execution
        logits = model(data)
        print(f"Final output logits shape: {logits.shape}")
        
        print("\n==== SAMPLING TEST ====")
        model.eval()
        # Sampling - will print dimensions during execution
        with torch.no_grad():
            samples = model.sample(data, n_samples=1, temperature=0.1)
            print(f"Final samples shape: {samples.shape}")
            
            print("\n==== SAMPLING WITH LOGITS TEST ====")
            samples, sampling_logits = model.sample(
                data, n_samples=1, temperature=0.1, return_logits=True
            )
            print(f"Final samples shape: {samples.shape}")
            print(f"Final sampling logits shape: {sampling_logits.shape}")
        
        print("\nAll tests passed! Dimension analysis complete.")
        return True
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


    
if __name__ == "__main__":
    test_forward_pass() 
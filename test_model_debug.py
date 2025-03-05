import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from src.data.dataset import RNADesignDataset
from src.models import AutoregressiveMultiGNNv1
from src.layers import MultiGVPConvLayer, GraphAttentionLayer
from src.constants import DATA_PATH


def debug_test_model():
    """
    Detailed test script to debug the modified model with parallel attention
    by checking intermediate representations and components.
    """
    print("Starting detailed debug test for the model with parallel attention...")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        # Load a single datapoint from the dataset
        print("Loading data...")
        data_list = list(torch.load(os.path.join(DATA_PATH, "processed.pt")).values())
        
        # Take just the first datapoint
        single_data = data_list[0]
        
        # Create dataset with a single sample
        dataset = RNADesignDataset(
            data_list=[single_data],
            split="test",
            radius=0.0,
            top_k=32,
            num_rbf=32,
            num_posenc=32,
            max_num_conformers=1,
            noise_scale=0.1
        )
        
        # Get the single datapoint
        data = dataset[0]
        
        # Move data to device
        data = data.to(device)
        
        # Print data information
        print(f"Data: {data}")
        print(f"Number of nodes: {data.node_s.shape[0]}")
        print(f"Node scalar shape: {data.node_s.shape}")
        print(f"Node vector shape: {data.node_v.shape}")
        print(f"Edge scalar shape: {data.edge_s.shape}")
        print(f"Edge vector shape: {data.edge_v.shape}")
        print(f"Edge index shape: {data.edge_index.shape}")
        print(f"Sequence shape: {data.seq.shape}")
        
        # Test GraphAttentionLayer separately first
        print("\nTesting GraphAttentionLayer component...")
        node_h_dim = (128, 16)
        attention_layer = GraphAttentionLayer(node_h_dim=node_h_dim, dropout=0.1).to(device)
        
        # Create dummy input to test attention layer
        batch_size = data.node_s.shape[0]
        dummy_scalar = torch.randn(batch_size, node_h_dim[0], device=device)
        dummy_vector = torch.randn(batch_size, node_h_dim[1], 3, device=device)
        dummy_input = (dummy_scalar, dummy_vector)
        
        # Test attention layer forward pass
        print("Testing attention layer forward pass...")
        attention_out = attention_layer(dummy_input, data.edge_index)
        print(f"Attention output scalar shape: {attention_out[0].shape}")
        print(f"Attention output vector shape: {attention_out[1].shape}")
        
        # Test MultiGVPConvLayer with attention
        print("\nTesting MultiGVPConvLayer with attention...")
        multi_layer = MultiGVPConvLayer(
            node_dims=node_h_dim,
            edge_dims=(data.edge_s.shape[2], data.edge_v.shape[2]),
            drop_rate=0.1
        ).to(device)
        
        # Test multi-layer forward pass
        print("Testing multi-layer forward pass...")
        multi_out = multi_layer(dummy_input, data.edge_index, (data.edge_s[:, 0], data.edge_v[:, 0]))
        print(f"MultiGVPConvLayer output scalar shape: {multi_out[0].shape}")
        print(f"MultiGVPConvLayer output vector shape: {multi_out[1].shape}")
        
        # Initialize full model
        print("\nInitializing full model...")
        model = AutoregressiveMultiGNNv1(
            node_in_dim=(data.node_s.shape[2], data.node_v.shape[2]),
            node_h_dim=node_h_dim,
            edge_in_dim=(data.edge_s.shape[2], data.edge_v.shape[2]),
            edge_h_dim=(32, 1),
            num_layers=3,
            drop_rate=0.1,
            out_dim=4
        ).to(device)
        
        # Print model information
        total_param = 0
        for param in model.parameters():
            total_param += np.prod(list(param.data.size()))
        print(f"\nMODEL\n    Total parameters: {total_param}")
        
        # Test model's encoder layers specifically
        print("\nTesting model's encoder layers...")
        h_V = (data.node_s, data.node_v)
        h_E = (data.edge_s, data.edge_v)
        edge_index = data.edge_index
        
        # Apply input embedding
        h_V = model.W_v(h_V)
        h_E = model.W_e(h_E)
        
        print(f"After embedding, h_V[0] shape: {h_V[0].shape}")
        print(f"After embedding, h_V[1] shape: {h_V[1].shape}")
        
        # Apply each encoder layer individually and check output
        for i, layer in enumerate(model.encoder_layers):
            h_V = layer(h_V, edge_index, h_E)
            print(f"After encoder layer {i+1}, h_V[0] shape: {h_V[0].shape}")
            print(f"After encoder layer {i+1}, h_V[1] shape: {h_V[1].shape}")
        
        # Test full forward pass
        print("\nTesting full forward pass...")
        model.train()
        logits = model(data)
        print(f"Logits shape: {logits.shape}")
        
        # Test sampling
        print("\nTesting sampling...")
        model.eval()
        with torch.no_grad():
            samples = model.sample(data, n_samples=2, temperature=0.1)
            print(f"Samples shape: {samples.shape}")
            print(f"Samples: {samples}")
        
        print("\nDetailed debug test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during debug testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    debug_test_model() 
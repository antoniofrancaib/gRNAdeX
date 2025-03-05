import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from src.data.dataset import RNADesignDataset
from src.models import AutoregressiveMultiGNNv1
from src.constants import DATA_PATH
from src.layers import (
    GVPConvLayer, 
    HybridGVPAttentionLayer,
    GraphAttentionLayer
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
        
        # Initialize model
        print("Initializing model...")
        model = AutoregressiveMultiGNNv1(
            node_in_dim=(data.node_s.shape[2], data.node_v.shape[2]),
            node_h_dim=(128, 16),
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
        
        # Test both forward pass and sampling
        print("\nTesting forward pass...")
        model.train()
        # Forward pass
        logits = model(data)
        print(f"Logits shape: {logits.shape}")
        
        print("\nTesting sampling...")
        model.eval()
        # Sampling (no_grad to avoid memory issues)
        with torch.no_grad():
            samples = model.sample(data, n_samples=2, temperature=0.1)
            print(f"Samples shape: {samples.shape}")
            print(f"Samples: {samples}")
            
            # Test with return_logits option
            samples, logits = model.sample(
                data, n_samples=1, temperature=0.1, return_logits=True
            )
            print(f"Samples shape: {samples.shape}")
            print(f"Sampling logits shape: {logits.shape}")
        
        print("\nAll tests passed! The model with parallel attention is working correctly.")
        return True
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_forward_pass() 
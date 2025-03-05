#!/usr/bin/env python3
"""
Quick validation script to verify that the new AutoregressiveMultiGNNv1 
with HybridGVPAttentionLayer trains correctly.

This runs minimal training (1 epoch, limited batches) to verify:
1. Data loading works
2. Model initialization works with attention parameters
3. Forward passes complete successfully
4. Backward passes and gradient updates work
5. Memory usage is reasonable
"""

import os
import torch
import argparse
import numpy as np
import time
from tqdm import tqdm

from src.models import AutoregressiveMultiGNNv1
from src.data.dataset import RNADesignDataset, BatchSampler
from torch_geometric.loader import DataLoader
from src.constants import DATA_PATH

def setup_minimal_config():
    """Create a minimal configuration for testing."""
    class Config:
        pass
    
    config = Config()
    
    # Data parameters
    config.radius = 10.0
    config.top_k = 30
    config.num_rbf = 16
    config.num_posenc = 16
    config.max_num_conformers = 5
    config.noise_scale = 0.0
    config.max_nodes_batch = 1000
    config.max_nodes_sample = 1000
    config.num_workers = 2
    
    # Model parameters
    config.node_in_dim = (31, 32)  # Will be overridden by actual data
    config.node_h_dim = (128, 16)
    config.edge_in_dim = (32, 1)  # Will be overridden by actual data
    config.edge_h_dim = (32, 1)
    config.num_layers = 3
    config.drop_rate = 0.1
    config.out_dim = 4  # RNA has 4 nucleotides
    
    # New attention parameters
    config.attention_heads = 4
    config.attention_dropout = 0.1
    
    # Training parameters
    config.lr = 0.001
    config.label_smoothing = 0.1
    
    return config

def get_limited_data(config, max_samples=10):
    """Load a limited dataset for quick testing."""
    print(f"Loading data from {DATA_PATH}...")
    data_dict = torch.load(os.path.join(DATA_PATH, "processed.pt"))
    data_list = list(data_dict.values())[:max_samples]
    
    print(f"Creating dataset with {len(data_list)} samples...")
    dataset = RNADesignDataset(
        data_list=data_list,
        split="train",
        radius=config.radius,
        top_k=config.top_k,
        num_rbf=config.num_rbf,
        num_posenc=config.num_posenc,
        max_num_conformers=config.max_num_conformers,
        noise_scale=config.noise_scale
    )
    
    # Update config with actual dimensions from data
    first_data = dataset[0]
    config.node_in_dim = (first_data.node_s.shape[2], first_data.node_v.shape[2])
    config.edge_in_dim = (first_data.edge_s.shape[2], first_data.edge_v.shape[2])
    
    print(f"Actual input dimensions:")
    print(f"  Node: {config.node_in_dim}")
    print(f"  Edge: {config.edge_in_dim}")
    
    loader = DataLoader(
        dataset,
        num_workers=config.num_workers,
        batch_sampler=BatchSampler(
            node_counts=dataset.node_counts,
            max_nodes_batch=config.max_nodes_batch,
            max_nodes_sample=config.max_nodes_sample,
            shuffle=True,
        ),
        pin_memory=True
    )
    
    return loader

def test_model_setup(config):
    """Initialize model and check basic functionality."""
    print("\n" + "="*50)
    print("INITIALIZING MODEL WITH ATTENTION")
    print("="*50)
    
    model = AutoregressiveMultiGNNv1(
        node_in_dim=config.node_in_dim,
        node_h_dim=config.node_h_dim,
        edge_in_dim=config.edge_in_dim,
        edge_h_dim=config.edge_h_dim,
        num_layers=config.num_layers,
        drop_rate=config.drop_rate,
        out_dim=config.out_dim,
        attention_heads=config.attention_heads,
        attention_dropout=config.attention_dropout
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params:,} parameters")
    
    # Verify encoder layers have attention
    hybrid_layers = 0
    for name, module in model.named_modules():
        if "HybridGVPAttentionLayer" in str(type(module)):
            hybrid_layers += 1
    print(f"Found {hybrid_layers} HybridGVPAttentionLayer instances")
    
    if hybrid_layers != config.num_layers:
        print(f"WARNING: Expected {config.num_layers} hybrid layers but found {hybrid_layers}")
    
    return model

def run_training_check(model, loader, device, num_batches=3, num_epochs=1):
    """Run minimal training to verify everything works."""
    print("\n" + "="*50)
    print("RUNNING TRAINING CHECK")
    print("="*50)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        
        # Track metrics
        batch_times = []
        forward_times = []
        backward_times = []
        loss_values = []
        memory_usage = []
        
        for i, batch in enumerate(tqdm(loader)):
            if i >= num_batches:
                break
                
            start_time = time.time()
            batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            forward_start = time.time()
            logits = model(batch)
            forward_end = time.time()
            
            # Loss computation
            loss = loss_fn(logits, batch.seq)
            loss_values.append(loss.item())
            
            # Backward pass
            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_end = time.time()
            
            # Timing and memory stats
            batch_time = time.time() - start_time
            forward_time = forward_end - forward_start
            backward_time = backward_end - backward_start
            
            batch_times.append(batch_time)
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            
            if device.type == "cuda":
                memory_usage.append(torch.cuda.max_memory_allocated(device) / 1024**2)
                torch.cuda.reset_peak_memory_stats(device)
            
            # Detailed batch stats
            print(f"\nBatch {i+1} Stats:")
            print(f"  Input shape: nodes={batch.node_s.shape}, edges={batch.edge_index.shape}")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Timing: Total={batch_time:.2f}s, Forward={forward_time:.2f}s, Backward={backward_time:.2f}s")
            
            if device.type == "cuda":
                print(f"  GPU Memory: {memory_usage[-1]:.1f} MB")
            
            # Check for NaN values in gradients
            has_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"  WARNING: NaN gradient in {name}")
                    has_nan = True
            
            if not has_nan:
                print("  Gradients: OK (no NaN values)")
        
        # Epoch summary
        print("\nEpoch Summary:")
        print(f"  Avg loss: {np.mean(loss_values):.4f}")
        print(f"  Avg batch time: {np.mean(batch_times):.2f}s")
        print(f"  Avg forward time: {np.mean(forward_times):.2f}s")
        print(f"  Avg backward time: {np.mean(backward_times):.2f}s")
        
        if device.type == "cuda" and memory_usage:
            print(f"  Max GPU memory: {np.max(memory_usage):.1f} MB")
    
    print("\n" + "="*50)
    print("TRAINING CHECK COMPLETED SUCCESSFULLY")
    print("="*50)
    print("The model with attention initialized and trained without errors!")
    print("You can now proceed with full training.")

def main():
    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Create configuration
    config = setup_minimal_config()
    
    # Load minimal dataset
    loader = get_limited_data(config, max_samples=10)
    
    # Initialize and check model
    model = test_model_setup(config)
    
    # Run minimal training
    run_training_check(model, loader, device, num_batches=3, num_epochs=1)

if __name__ == "__main__":
    main() 
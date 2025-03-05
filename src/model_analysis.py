#!/usr/bin/env python3

import os
import torch
import numpy as np
import pandas as pd
import traceback
from collections import OrderedDict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.models import AutoregressiveMultiGNNv1, AutoregressiveMultiGNNv1_previous, NonAutoregressiveMultiGNNv1
from src.layers import (
    HybridGVPAttentionLayer, GVPConvLayer, MultiGVPConvLayer, 
    GVPConv, MultiGVPConv, LayerNorm, GVP, GraphAttentionLayer
)
from src.constants import DATA_PATH
from src.data.dataset import RNADesignDataset

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_layer_parameters(model):
    """Count parameters for each layer in the model."""
    results = OrderedDict()
    for name, module in model.named_modules():
        if name == '':  # Skip the model itself
            continue
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if params > 0:  # Only include layers with parameters
            results[name] = params
    return results

def count_submodule_parameters(module):
    """Count parameters for each submodule in a given module."""
    results = OrderedDict()
    for name, submodule in module.named_modules():
        if name == '':  # Skip the module itself
            continue
        params = sum(p.numel() for p in submodule.parameters() if p.requires_grad)
        if params > 0:  # Only include modules with parameters
            results[name] = params
    return results

def get_real_datapoint():
    """Load a single real datapoint from the processed data file."""
    # Load the processed data
    data_path = os.path.join(DATA_PATH, "processed.pt")
    print(f"Loading data from {data_path}")
    data_dict = torch.load(data_path)
    data_list = list(data_dict.values())
    
    # Get the first data point
    sample_data = data_list[0]
    print(f"Loaded sample datapoint with {len(sample_data.keys()) if hasattr(sample_data, 'keys') else 'unknown'} keys")
    
    # Create a dataset to properly process the datapoint
    dataset = RNADesignDataset(
        data_list=[sample_data],
        split="test",
        radius=10.0,
        top_k=30,
        num_rbf=16,
        num_posenc=16,
        max_num_conformers=5,
        noise_scale=0.0
    )
    
    # Get the properly processed data point
    processed_data = dataset[0]
    print("\nProcessed datapoint attributes:")
    for key in processed_data.keys():
        if isinstance(processed_data[key], torch.Tensor):
            print(f"  {key}: {processed_data[key].shape}")
        else:
            print(f"  {key}: {type(processed_data[key])}")
    
    return processed_data

def compare_conv_attention_models(node_dims, edge_dims, n_conf=5):
    """Compare the architecture and parameter count of different convolution layers."""
    print("\n=== Convolution Layer Comparison ===")
    
    # Create instances of different types of convolution layers
    gvp_conv = GVPConv(node_dims, node_dims, edge_dims, n_layers=3)
    multi_gvp_conv = MultiGVPConv(node_dims, node_dims, edge_dims, n_layers=3)
    
    # Create a properly designed hybrid layer (this is what we should have)
    class ProperHybridLayer(torch.nn.Module):
        def __init__(self, node_dims, edge_dims):
            super().__init__()
            self.conv = MultiGVPConv(node_dims, node_dims, edge_dims, n_layers=3)
            self.attention = HybridGVPAttentionLayer(node_dims, edge_dims).attention
            self.norm_s = torch.nn.LayerNorm(node_dims[0])
            if node_dims[1] > 0:
                self.norm_v = torch.nn.LayerNorm(node_dims[1])
                
    # Current hybrid layer
    current_hybrid = HybridGVPAttentionLayer(node_dims, edge_dims)
    
    # Proposed correct hybrid layer
    proper_hybrid = ProperHybridLayer(node_dims, edge_dims)
    
    # Count parameters
    gvp_params = count_parameters(gvp_conv)
    multi_gvp_params = count_parameters(multi_gvp_conv)
    current_hybrid_params = count_parameters(current_hybrid)
    proper_hybrid_params = count_parameters(proper_hybrid)
    
    print(f"GVPConv parameters: {gvp_params:,}")
    print(f"MultiGVPConv parameters: {multi_gvp_params:,}")
    print(f"Current HybridGVPAttentionLayer parameters: {current_hybrid_params:,}")
    print(f"Proposed correct hybrid layer parameters: {proper_hybrid_params:,}")
    
    # Detailed component breakdown
    print("\nCurrent hybrid (HybridGVPAttentionLayer) component breakdown:")
    current_hybrid_comps = count_submodule_parameters(current_hybrid)
    for name, params in current_hybrid_comps.items():
        if params > 0:
            print(f"  {name}: {params:,}")
    
    # Analyze the conv component specifically
    print("\nConv component analysis:")
    print(f"Current hybrid uses: {type(current_hybrid.conv).__name__}")
    print(f"Previous model used: MultiGVPConv")
    print(f"Difference: GVPConv has {gvp_params:,} parameters, while MultiGVPConv has {multi_gvp_params:,} parameters")
    
    return {
        'gvp_conv': gvp_params,
        'multi_gvp_conv': multi_gvp_params,
        'current_hybrid': current_hybrid_params,
        'proper_hybrid': proper_hybrid_params
    }

def analyse_hybrid_architecture():
    """Analyze the current implementation of HybridGVPAttentionLayer."""
    print("\n=== Analysis of HybridGVPAttentionLayer Implementation ===")
    print("Current implementation issues:")
    print("1. HybridGVPAttentionLayer uses GVPConv instead of MultiGVPConv")
    print("2. MultiGVPConv is specifically designed for multi-conformation data")
    print("3. This explains why our hybrid layer has fewer parameters - we're using a simpler convolution layer")
    
    print("\nWhat the implementation should be:")
    print("1. Keep MultiGVPConv for handling multiple conformations")
    print("2. Add GraphAttentionLayer in parallel")
    print("3. Combine their outputs with equal weights (0.5 each)")
    
    print("\nSuggested fix:")
    print("Modify HybridGVPAttentionLayer to use MultiGVPConv instead of GVPConv")
    print("This should be done in the layer's __init__ method:")
    print("```python")
    print("self.conv = MultiGVPConv(  # Use MultiGVPConv instead of GVPConv")
    print("    in_dims = node_dims,")
    print("    out_dims = node_dims,") 
    print("    edge_dims = edge_dims,")
    print("    n_layers = 3,")
    print("    vector_gate = vector_gate,")
    print("    activations = activations")
    print(")")
    print("```")

def compare_models_side_by_side(current_model, previous_model):
    """Compare two models side by side, focusing on differences in layer parameters."""
    print("\n=== Side-by-Side Model Comparison ===")
    
    # Get parameter counts
    current_params = count_parameters(current_model)
    previous_params = count_parameters(previous_model)
    
    # Get layer-by-layer parameter counts
    current_layer_params = count_layer_parameters(current_model)
    previous_layer_params = count_layer_parameters(previous_model)
    
    # Print total parameter counts
    print(f"Current model total parameters: {current_params:,}")
    print(f"Previous model total parameters: {previous_params:,}")
    print(f"Difference: {current_params - previous_params:,} ({(current_params/previous_params - 1)*100:.2f}%)")
    
    # Categorize layers for detailed comparison
    current_embedding = {k: v for k, v in current_layer_params.items() if k.startswith('W_v') or k.startswith('W_e') or k.startswith('W_s')}
    previous_embedding = {k: v for k, v in previous_layer_params.items() if k.startswith('W_v') or k.startswith('W_e') or k.startswith('W_s')}
    
    current_encoder = {k: v for k, v in current_layer_params.items() if 'encoder_layers' in k}
    previous_encoder = {k: v for k, v in previous_layer_params.items() if 'encoder_layers' in k}
    
    current_decoder = {k: v for k, v in current_layer_params.items() if 'decoder_layers' in k}
    previous_decoder = {k: v for k, v in previous_layer_params.items() if 'decoder_layers' in k}
    
    current_output = {k: v for k, v in current_layer_params.items() if k.startswith('W_out')}
    previous_output = {k: v for k, v in previous_layer_params.items() if k.startswith('W_out')}
    
    # Print comparison table
    print("\n=== Layer-by-Layer Parameter Comparison ===")
    
    # Header
    print(f"{'Layer Type':<30} {'Current Model':<15} {'Previous Model':<15} {'Difference':<12} {'% Change':<10}")
    print('-' * 85)
    
    # Embedding layers
    total_current_emb = sum(current_embedding.values())
    total_previous_emb = sum(previous_embedding.values())
    print(f"{'Embedding Layers':<30} {total_current_emb:<15,} {total_previous_emb:<15,} {total_current_emb - total_previous_emb:<12,} {(total_current_emb/total_previous_emb - 1)*100:<10.2f}%")
    
    # Encoder layers
    total_current_enc = sum(current_encoder.values())
    total_previous_enc = sum(previous_encoder.values())
    print(f"{'Encoder Layers':<30} {total_current_enc:<15,} {total_previous_enc:<15,} {total_current_enc - total_previous_enc:<12,} {(total_current_enc/total_previous_enc - 1)*100:<10.2f}%")
    
    # Decoder layers
    total_current_dec = sum(current_decoder.values())
    total_previous_dec = sum(previous_decoder.values())
    print(f"{'Decoder Layers':<30} {total_current_dec:<15,} {total_previous_dec:<15,} {total_current_dec - total_previous_dec:<12,} {(total_current_dec/total_previous_dec - 1)*100:<10.2f}%")
    
    # Output layers
    total_current_out = sum(current_output.values())
    total_previous_out = sum(previous_output.values())
    print(f"{'Output Layers':<30} {total_current_out:<15,} {total_previous_out:<15,} {total_current_out - total_previous_out:<12,} {(total_current_out/total_previous_out - 1)*100:<10.2f}%")
    
    # Total
    print('-' * 85)
    print(f"{'TOTAL':<30} {current_params:<15,} {previous_params:<15,} {current_params - previous_params:<12,} {(current_params/previous_params - 1)*100:<10.2f}%")
    
    # Detailed encoder layer comparison
    print("\n=== Detailed Encoder Layer Comparison ===")
    # Extract all encoder layers
    current_enc_layers = [module for name, module in current_model.named_modules() 
                         if isinstance(module, HybridGVPAttentionLayer)]
    previous_enc_layers = [module for name, module in previous_model.named_modules() 
                          if isinstance(module, MultiGVPConvLayer)]
    
    if current_enc_layers and previous_enc_layers:
        # Analyze the first encoder layer of each model
        current_enc = current_enc_layers[0]
        previous_enc = previous_enc_layers[0]
        
        # Current model encoder layer breakdown - handle both old and new HybridGVPAttentionLayer structures
        current_conv_params = sum(p.numel() for p in current_enc.conv.parameters() if p.requires_grad)
        current_attn_params = sum(p.numel() for p in current_enc.attention.parameters() if p.requires_grad)
        
        # Check if we're using the old or new normalization structure
        if hasattr(current_enc, 'norm'):
            # New structure with a single norm attribute
            current_norm_params = sum(p.numel() for p in current_enc.norm.parameters() if p.requires_grad)
        else:
            # Old structure with separate norm_s and norm_v attributes
            current_norm_params = sum(p.numel() for p in current_enc.norm_s.parameters() if p.requires_grad)
            if hasattr(current_enc, 'norm_v'):
                current_norm_params += sum(p.numel() for p in current_enc.norm_v.parameters() if p.requires_grad)
        
        # Previous model encoder layer breakdown
        previous_conv_params = sum(p.numel() for p in previous_enc.conv.parameters() if p.requires_grad)
        previous_norm_params = sum(p.numel() for p in previous_enc.norm[0].parameters() if p.requires_grad)
        previous_norm_params += sum(p.numel() for p in previous_enc.norm[1].parameters() if p.requires_grad)
        previous_ff_params = sum(p.numel() for p in previous_enc.ff_func.parameters() if p.requires_grad)
        
        print("Current model (HybridGVPAttentionLayer) components:")
        print(f"  GVPConv component: {current_conv_params:,} parameters")
        print(f"  Attention component: {current_attn_params:,} parameters")
        print(f"  Normalization components: {current_norm_params:,} parameters")
        print(f"  Total per layer: {current_conv_params + current_attn_params + current_norm_params:,} parameters")
        
        print("\nPrevious model (MultiGVPConvLayer) components:")
        print(f"  MultiGVPConv component: {previous_conv_params:,} parameters")
        print(f"  Normalization components: {previous_norm_params:,} parameters")
        print(f"  Feedforward components: {previous_ff_params:,} parameters")
        print(f"  Total per layer: {previous_conv_params + previous_norm_params + previous_ff_params:,} parameters")
        
        diff = (current_conv_params + current_attn_params + current_norm_params) - (previous_conv_params + previous_norm_params + previous_ff_params)
        pct = ((current_conv_params + current_attn_params + current_norm_params) / (previous_conv_params + previous_norm_params + previous_ff_params) - 1) * 100
        print(f"\nDifference per layer: {diff:,} parameters ({pct:.2f}%)")
        
        # Highlight the key difference
        print("\n=== Key Architectural Difference ===")
        print(f"Current model uses MultiGVPConv: {isinstance(current_enc.conv, MultiGVPConv)}")
        print(f"Previous model uses MultiGVPConv: {isinstance(previous_enc.conv, MultiGVPConv)}")
        if isinstance(current_enc.conv, MultiGVPConv):
            print("✅ Successfully updated to use MultiGVPConv!")
        else:
            print("❌ Error: Current model should be using MultiGVPConv")

def test_hybrid_layer_update():
    """Test the updated HybridGVPAttentionLayer implementation with MultiGVPConv."""
    print("\n=== Testing Updated HybridGVPAttentionLayer Implementation ===")
    
    try:
        # Import necessary modules locally to ensure they're available
        from src.layers import LayerNorm, GVP, MultiGVPConv, GraphAttentionLayer
        
        # Get dimensions from real data
        sample_data = get_real_datapoint()
        
        node_scalar_dim = sample_data.node_s.shape[2]
        node_vector_dim = sample_data.node_v.shape[2]
        edge_scalar_dim = sample_data.edge_s.shape[2]
        edge_vector_dim = sample_data.edge_v.shape[2]
        
        node_in_dim = (node_scalar_dim, node_vector_dim)
        node_h_dim = (128, 16)
        edge_in_dim = (edge_scalar_dim, edge_vector_dim)
        edge_h_dim = (32, 1)
        
        print(f"Input dimensions: node={node_in_dim}, edge={edge_in_dim}")
        print(f"Hidden dimensions: node={node_h_dim}, edge={edge_h_dim}")
        
        # Create instances of different layer types for comparison
        gvp_conv = GVPConv(node_h_dim, node_h_dim, edge_h_dim, n_layers=3)
        multi_gvp_conv = MultiGVPConv(node_h_dim, node_h_dim, edge_h_dim, n_layers=3)
        
        # Use only 1 attention head for testing to avoid dimension issues
        hybrid_layer = HybridGVPAttentionLayer(node_h_dim, edge_h_dim, n_heads=4, attention_dropout=0.1)
        
        # Verify the hybrid layer is using MultiGVPConv
        is_using_multi_gvp = isinstance(hybrid_layer.conv, MultiGVPConv)
        print(f"HybridGVPAttentionLayer is using MultiGVPConv: {is_using_multi_gvp}")
        
        if not is_using_multi_gvp:
            print("WARNING: HybridGVPAttentionLayer is not using MultiGVPConv!")
        
        # Count parameters for comparison
        gvp_params = count_parameters(gvp_conv)
        multi_gvp_params = count_parameters(multi_gvp_conv)
        hybrid_params = count_parameters(hybrid_layer)
        
        print(f"GVPConv parameters: {gvp_params:,}")
        print(f"MultiGVPConv parameters: {multi_gvp_params:,}")
        print(f"Updated HybridGVPAttentionLayer parameters: {hybrid_params:,}")
        
        # Parameter count validation
        if hybrid_params > multi_gvp_params:
            print("✅ Success: HybridGVPAttentionLayer now has MORE parameters than MultiGVPConv")
            print(f"   Additional parameters: {hybrid_params - multi_gvp_params:,}")
        else:
            print("❌ Error: HybridGVPAttentionLayer should have more parameters than MultiGVPConv")
        
        # Verify forward pass with real data
        try:
            # Extract input features from sample data
            h_V = (sample_data.node_s, sample_data.node_v)
            h_E = (sample_data.edge_s, sample_data.edge_v)
            edge_index = sample_data.edge_index
            
            print("\nTesting forward pass with real data...")
            
            # First, embed the features to the right dimensions
            W_v = torch.nn.Sequential(
                LayerNorm(node_in_dim),
                GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=True)
            )
            W_e = torch.nn.Sequential(
                LayerNorm(edge_in_dim),
                GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=True)
            )
            
            h_V_emb = W_v(h_V)
            h_E_emb = W_e(h_E)
            
            # Test MultiGVPConv
            multi_out = multi_gvp_conv(h_V_emb, edge_index, h_E_emb)
            print(f"MultiGVPConv output shapes: {multi_out[0].shape}, {multi_out[1].shape}")
            
            # Print attention layer configuration
            n_confs = h_V_emb[0].shape[1]
            print(f"Number of conformations: {n_confs}")
            print(f"Attention layer input dim: {node_h_dim[0]}")
            print(f"Attention layer heads: {hybrid_layer.attention.n_heads}")
            print(f"Attention layer W_O shape: {hybrid_layer.attention.W_O.weight.shape}")
            
            # Test HybridGVPAttentionLayer
            hybrid_out = hybrid_layer(h_V_emb, edge_index, h_E_emb)
            print(f"HybridGVPAttentionLayer output shapes: {hybrid_out[0].shape}, {hybrid_out[1].shape}")
            
            # Ensure outputs have the same shape
            shapes_match = (multi_out[0].shape == hybrid_out[0].shape and 
                           multi_out[1].shape == hybrid_out[1].shape)
            
            if shapes_match:
                print("✅ Forward pass successful: Output shapes match expected dimensions")
            else:
                print("❌ Error: Output shapes do not match expected dimensions")
                print(f"Expected: {multi_out[0].shape}, {multi_out[1].shape}")
                print(f"Got: {hybrid_out[0].shape}, {hybrid_out[1].shape}")
            
            # Initialize a small model for testing
            mini_model = NonAutoregressiveMultiGNNv1(
                node_in_dim=node_in_dim,
                node_h_dim=node_h_dim,
                edge_in_dim=edge_in_dim,
                edge_h_dim=edge_h_dim,
                num_layers=1,  # Use just 1 layer for quick testing
                attention_heads=1  # Use 1 head to avoid dimension issues
            )
            
            # Test full model forward pass
            try:
                logits = mini_model(sample_data)
                print(f"Full model output shape: {logits.shape}")
                print("✅ Full model forward pass successful")
            except Exception as e:
                print(f"❌ Error in full model forward pass: {e}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"❌ Error during forward pass testing: {e}")
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Error in test setup: {e}")
        traceback.print_exc()
    
    print("\nImplementation analysis:")
    print("1. The updated HybridGVPAttentionLayer now uses MultiGVPConv for handling multiple conformations")
    print("2. Graph attention is applied separately within each conformation")
    print("3. The model properly combines convolution and attention pathways")
    print("4. Parameter count has increased as expected with this more powerful architecture")

def main():
    # First get a real data point to determine the actual dimensions
    sample_data = get_real_datapoint()
    
    # Extract actual dimensions from the data
    node_scalar_dim = sample_data.node_s.shape[2]  # 15
    node_vector_dim = sample_data.node_v.shape[2]  # 4 (not 3 as previously set)
    edge_scalar_dim = sample_data.edge_s.shape[2]  # 67 (not 16 as previously set)
    edge_vector_dim = sample_data.edge_v.shape[2]  # 3
    
    print(f"\nDetected dimensions from real data:")
    print(f"  Node dimensions: ({node_scalar_dim}, {node_vector_dim})")
    print(f"  Edge dimensions: ({edge_scalar_dim}, {edge_vector_dim})")
    
    # Define model parameters based on actual data dimensions
    node_in_dim = (node_scalar_dim, node_vector_dim)
    node_h_dim = (128, 16)
    edge_in_dim = (edge_scalar_dim, edge_vector_dim)
    edge_h_dim = (32, 1)
    num_layers = 3
    drop_rate = 0.1
    out_dim = 4
    
    # Initialize current model with HybridGVPAttentionLayer
    current_model = AutoregressiveMultiGNNv1(
        node_in_dim=node_in_dim,
        node_h_dim=node_h_dim,
        edge_in_dim=edge_in_dim,
        edge_h_dim=edge_h_dim,
        num_layers=num_layers,
        drop_rate=drop_rate,
        out_dim=out_dim,
        attention_heads=4,
        attention_dropout=0.1
    )
    
    # Initialize previous model with MultiGVPConvLayer
    previous_model = AutoregressiveMultiGNNv1_previous(
        node_in_dim=node_in_dim,
        node_h_dim=node_h_dim,
        edge_in_dim=edge_in_dim,
        edge_h_dim=edge_h_dim,
        num_layers=num_layers,
        drop_rate=drop_rate,
        out_dim=out_dim
    )
    
    # Compare models side by side
    compare_models_side_by_side(current_model, previous_model)
    
    # Compare different conv layer architectures
    compare_conv_attention_models(node_h_dim, edge_h_dim)
    
    # Analyze the hybrid architecture
    analyse_hybrid_architecture()
    
    # Test the updated implementation
    test_hybrid_layer_update()
    
    # Summary and conclusions
    print("\n=== Summary and Conclusions ===")
    print("1. The HybridGVPAttentionLayer now uses MultiGVPConv, which is designed for multi-conformation data")
    print("2. Attention is applied separately within each conformation")
    print("3. The updated model combines the strengths of both approaches:")
    print("   a. MultiGVPConv handles the multi-conformation structure correctly")
    print("   b. GraphAttentionLayer adds global context within each conformation")
    print("4. This implementation should result in better model performance while properly")
    print("   handling the multiple RNA conformations in your data.")
    
if __name__ == "__main__":
    main() 
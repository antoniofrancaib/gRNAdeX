#!/usr/bin/env python3

import os
import torch
import numpy as np
from collections import OrderedDict
from torch_geometric.data import Data

from src.models import AutoregressiveMultiGNNv1, AutoregressiveMultiGNNv1_previous
from src.layers import HybridGVPAttentionLayer, LayerNorm, GVP, MultiGVPConvLayer
from src.constants import DATA_PATH
from src.data.dataset import RNADesignDataset

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ModelInspector:
    """
    Tracks and displays the transformations of data through the model.
    """
    def __init__(self, model, model_name="Model", verbose=True, last_nucleotide_only=False):
        self.model = model
        self.model_name = model_name
        self.hooks = []
        self.activations = OrderedDict()
        self.verbose = verbose
        self.last_nucleotide_only = last_nucleotide_only
        
        # Register hooks for all encoder layers
        for name, module in model.named_modules():
            if isinstance(module, (HybridGVPAttentionLayer, MultiGVPConvLayer)):
                self.register_hooks(module, name)
            elif 'W_v' in name or 'W_e' in name or 'W_s' in name or 'decoder_layers' in name or 'W_out' in name:
                self.register_hooks(module, name)
    
    def register_hooks(self, module, name):
        """Register forward hooks on modules to capture inputs and outputs."""
        def hook_fn(module, input, output, name=name):
            self.activations[f"{name}_input"] = input
            self.activations[f"{name}_output"] = output
            
        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def print_activations(self):
        """Print the stored activations with clear formatting."""
        if not self.verbose:
            return
            
        print("\n" + "="*80)
        print(f"{self.model_name.upper()} TRANSFORMATIONS VISUALIZATION")
        print("="*80)
        
        # First print the input to the model
        print("\nModel Input:")
        self._print_data_info(self.data_input)
        
        # Then print embedding layer transformations
        print("\nEmbedding Layers:")
        for name in self.activations:
            if any(x in name for x in ['W_v', 'W_e', 'W_s']) and 'input' in name:
                module_name = name.replace('_input', '')
                print(f"\n{module_name} Input:")
                self._print_tensor_info(self.activations[name], print_stats=not self.last_nucleotide_only)
                print(f"\n{module_name} Output:")
                self._print_tensor_info(self.activations[module_name + '_output'], print_stats=not self.last_nucleotide_only)
        
        # Print encoder layer transformations
        print("\nEncoder Layers:")
        encoder_layers = [name for name in self.activations if 'encoder_layers' in name and 'input' in name]
        encoder_layers.sort()  # Sort to ensure proper order
        
        for i, name in enumerate(encoder_layers):
            module_name = name.replace('_input', '')
            print(f"\nEncoder Layer {i+1}:")
            print("Input:")
            self._print_tensor_info(self.activations[name], print_stats=not self.last_nucleotide_only)
            print("\nOutput:")
            self._print_tensor_info(self.activations[module_name + '_output'], print_stats=not self.last_nucleotide_only)
        
        # Print decoder layer transformations if applicable
        if not self.last_nucleotide_only:
            decoder_layers = [name for name in self.activations if 'decoder_layers' in name and 'input' in name]
            decoder_layers.sort()
            
            if decoder_layers:
                print("\nDecoder Layers:")
                for i, name in enumerate(decoder_layers):
                    module_name = name.replace('_input', '')
                    print(f"\nDecoder Layer {i+1}:")
                    print("Input:")
                    self._print_tensor_info(self.activations[name], print_stats=True)
                    print("\nOutput:")
                    self._print_tensor_info(self.activations[module_name + '_output'], print_stats=True)
        else:
            # For last nucleotide, only print the last decoder layer
            decoder_layers = [name for name in self.activations if 'decoder_layers' in name and 'input' in name]
            decoder_layers.sort()
            
            if decoder_layers:
                last_layer_name = decoder_layers[-1]
                module_name = last_layer_name.replace('_input', '')
                print("\nLast Decoder Layer (for last nucleotide):")
                print("Input:")
                self._print_tensor_info(self.activations[last_layer_name], print_stats=True)
                print("\nOutput:")
                self._print_tensor_info(self.activations[module_name + '_output'], print_stats=True)
        
        # Print output layer transformation
        if not self.last_nucleotide_only:
            output_layers = [name for name in self.activations if 'W_out' in name and 'input' in name]
            if output_layers:
                print("\nOutput Layer:")
                for name in output_layers:
                    module_name = name.replace('_input', '')
                    print("\nInput:")
                    self._print_tensor_info(self.activations[name], print_stats=True)
                    print("\nOutput:")
                    self._print_tensor_info(self.activations[module_name + '_output'], print_stats=True)
        else:
            # For last nucleotide, print only the output layer
            output_layers = [name for name in self.activations if 'W_out' in name and 'input' in name]
            if output_layers:
                last_output_name = output_layers[-1]
                module_name = last_output_name.replace('_input', '')
                print("\nOutput Layer (for last nucleotide):")
                print("Input:")
                self._print_tensor_info(self.activations[last_output_name], print_stats=True)
                print("\nOutput:")
                self._print_tensor_info(self.activations[module_name + '_output'], print_stats=True)
        
        # Print final model output
        print("\nFinal Model Output (Logits):")
        self._print_tensor_info(self.model_output, print_stats=True)
        
        if hasattr(self, 'predicted_sequence'):
            print("\nPredicted Sequence:")
            print(self.predicted_sequence)
        
        # Print parameter count
        param_count = count_parameters(self.model)
        print(f"\nTotal Parameters: {param_count:,}")
    
    def _print_data_info(self, data):
        """Print information about a PyG Data object."""
        print(f"node_s = {data.node_s.shape}")
        print(f"node_v = {data.node_v.shape}")
        if hasattr(data, 'edge_s'):
            print(f"edge_s = {data.edge_s.shape}")
        if hasattr(data, 'edge_v'):
            print(f"edge_v = {data.edge_v.shape}")
        print(f"edge_index = {data.edge_index.shape}")
    
    def _print_tensor_info(self, tensor, print_stats=True):
        """Print information about tensor or tuple of tensors."""
        if isinstance(tensor, tuple):
            for i, t in enumerate(tensor):
                if isinstance(t, torch.Tensor):
                    print(f"tensor[{i}] shape = {t.shape}")
                    if print_stats and t.dtype in [torch.float16, torch.float32, torch.float64]:
                        print(f"  mean = {t.mean().item():.4f}, std = {t.std().item():.4f}")
                        print(f"  min = {t.min().item():.4f}, max = {t.max().item():.4f}")
                    elif print_stats:
                        print(f"  dtype = {t.dtype} (statistics skipped for non-float tensors)")
                else:
                    print(f"tensor[{i}] = {type(t)}")
        elif isinstance(tensor, torch.Tensor):
            print(f"shape = {tensor.shape}")
            # Only print statistics for floating point tensors
            if print_stats and tensor.dtype in [torch.float16, torch.float32, torch.float64]:
                print(f"mean = {tensor.mean().item():.4f}, std = {tensor.std().item():.4f}")
                print(f"min = {tensor.min().item():.4f}, max = {tensor.max().item():.4f}")
            elif print_stats:
                print(f"dtype = {tensor.dtype} (statistics skipped for non-float tensors)")
        else:
            print(f"type = {type(tensor)}")
    
    def run_and_analyze(self, data):
        """Run the model with the provided data and analyze the transformations."""
        self.data_input = data
        with torch.no_grad():
            self.model_output = self.model(data)
            
            # Get predicted sequence if applicable
            if isinstance(self.model_output, torch.Tensor) and len(self.model_output.shape) > 1:
                # Assuming the output is logits [batch, seq_len, num_classes]
                predictions = torch.argmax(self.model_output, dim=-1)
                # Convert to sequence if applicable
                self.predicted_sequence = predictions.tolist()
        
        # Print the captured activations
        self.print_activations()
        
        # Clean up
        self.remove_hooks()
        
        return self.model_output, self.predicted_sequence if hasattr(self, 'predicted_sequence') else None

def get_real_datapoint():
    """Load a single real datapoint from the processed data file."""
    # Load the processed data
    data_path = os.path.join(DATA_PATH, "processed.pt")
    print(f"Loading data from {data_path}")
    data_dict = torch.load(data_path)
    data_list = list(data_dict.values())
    
    # Get the first data point
    sample_data = data_list[0]
    
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
    return processed_data

def compare_model_outputs(output1, output2, seq1, seq2):
    """Compare the outputs and predictions of two models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Check if the logits are similar
    if isinstance(output1, torch.Tensor) and isinstance(output2, torch.Tensor):
        logits_diff = torch.abs(output1 - output2).mean().item()
        logits_max_diff = torch.abs(output1 - output2).max().item()
        print(f"\nLogits Difference:")
        print(f"Mean Absolute Difference: {logits_diff:.6f}")
        print(f"Max Absolute Difference: {logits_max_diff:.6f}")
    
    # Check if the predicted sequences match
    sequences_match = False
    if seq1 is not None and seq2 is not None:
        sequences_match = seq1 == seq2
        print(f"\nPredicted Sequences Match: {sequences_match}")
        
        if not sequences_match:
            # Show where they differ
            if isinstance(seq1, list) and isinstance(seq2, list):
                if len(seq1) == len(seq2):
                    diff_positions = [i for i in range(len(seq1)) if seq1[i] != seq2[i]]
                    print(f"Sequences differ at positions: {diff_positions}")
                    for pos in diff_positions[:5]:  # Show first 5 differences
                        print(f"Position {pos}: Current model = {seq1[pos]}, Previous model = {seq2[pos]}")
                else:
                    print(f"Sequence lengths differ: Current = {len(seq1)}, Previous = {len(seq2)}")
    
    return sequences_match

def main():
    # First get a real data point to determine the actual dimensions
    sample_data = get_real_datapoint()
    
    # Extract actual dimensions from the data
    node_scalar_dim = sample_data.node_s.shape[2]
    node_vector_dim = sample_data.node_v.shape[2]
    edge_scalar_dim = sample_data.edge_s.shape[2]
    edge_vector_dim = sample_data.edge_v.shape[2]
    
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
    
    # Initialize previous model
    previous_model = AutoregressiveMultiGNNv1_previous(
        node_in_dim=node_in_dim,
        node_h_dim=node_h_dim,
        edge_in_dim=edge_in_dim,
        edge_h_dim=edge_h_dim,
        num_layers=num_layers,
        drop_rate=drop_rate,
        out_dim=out_dim
    )
    
    # Print parameter counts
    current_params = count_parameters(current_model)
    previous_params = count_parameters(previous_model)
    
    print("\n" + "="*80)
    print("MODEL PARAMETER COMPARISON")
    print("="*80)
    print(f"Current model parameters: {current_params:,}")
    print(f"Previous model parameters: {previous_params:,}")
    print(f"Difference: {current_params - previous_params:,} parameters ({(current_params/previous_params - 1)*100:.2f}%)")
    
    # Initialize model inspectors - only show detailed output for current model
    current_inspector = ModelInspector(current_model, "Current Model (with HybridGVPAttentionLayer)", verbose=True, last_nucleotide_only=True)
    previous_inspector = ModelInspector(previous_model, "Previous Model (with MultiGVPConvLayer)", verbose=False)
    
    # Run analysis for current model first with detailed output
    print("\nAnalyzing Current Model...")
    current_output, current_seq = current_inspector.run_and_analyze(sample_data)
    
    # Then run previous model quietly and just get the predictions
    print("\nAnalyzing Previous Model (silently)...")
    previous_output, previous_seq = previous_inspector.run_and_analyze(sample_data)
    print("\nPrevious Model Predicted Sequence:")
    print(previous_seq)
    
    # Compare model outputs and predictions
    sequences_match = compare_model_outputs(current_output, previous_output, current_seq, previous_seq)
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Current Model (with HybridGVPAttentionLayer): {current_params:,} parameters")
    print(f"Previous Model (with MultiGVPConvLayer): {previous_params:,} parameters")
    print(f"Parameter Increase: {current_params - previous_params:,} ({(current_params/previous_params - 1)*100:.2f}%)")
    print(f"Sequences Match: {sequences_match}")
    print("\nNote: If sequences match, the models are functionally equivalent despite architectural differences.")
    print("If sequences differ, the architectural changes are affecting the model's behavior.")

if __name__ == "__main__":
    main() 
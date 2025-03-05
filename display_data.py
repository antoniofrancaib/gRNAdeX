import os
import math
import torch
import torch.nn as nn
from src.models import AutoregressiveMultiGNNv1

def product_of_linear_norms(module):
    """Recursively multiply the operator norms of all nn.Linear layers within a module."""
    prod = 1.0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            norm = torch.linalg.norm(m.weight.data, ord=2).item()
            prod *= norm
    return prod

def compute_NN_contribution(model):
    # Compute contribution from the initial node embedding:
    init_node_norm = product_of_linear_norms(model.W_v)
    # Compute contribution from each encoder layer (message passing + feedforward)
    encoder_norm = 1.0
    for layer in model.encoder_layers:
        layer_norm = product_of_linear_norms(layer)
        encoder_norm *= layer_norm
    # Multiply initial embedding and encoder layers:
    NN_contribution = init_node_norm * encoder_norm
    return NN_contribution

def main():
    # Instantiate the model
    model = AutoregressiveMultiGNNv1()
    
    # Print all linear layers with their operator norms.
    print("Printing all linear transformations and their operator norms:")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            norm = torch.linalg.norm(module.weight.data, ord=2).item()
            print(f"{name}: shape {module.weight.shape}, operator norm = {norm:.4f}")
    
    # Compute the product of operator norms from the initial embedding and encoder layers.
    NN_contrib = compute_NN_contribution(model)
    print("\nNN contribution (initial embedding * encoder layers) ≈", NN_contrib)
    
    # Assume top_k neighbors; get k from your configuration (e.g., 16 or 32)
    top_k = 16  # change this as needed
    A_norm = math.sqrt(top_k)
    print("Adjacency norm estimate (sqrt(top_k)) =", A_norm)
    
    L_encoder = NN_contrib * A_norm
    print("Estimated overall L_encoder ≈", L_encoder)
    
if __name__ == "__main__":
    main()

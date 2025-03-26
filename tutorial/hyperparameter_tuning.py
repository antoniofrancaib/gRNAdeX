# Create a script to tune hyperparameters for the gRNAde model using the test_demodata.py script

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import argparse

# Define functions
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()  

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# perform hyperparameter tuning using grid search
def grid_search(args):
    # load the default config
    config = load_config(args.config)   
    
    # define the hyperparameters to tune
    hyperparameters = {
        'model_type': ['ARv1', 'ARv2'],
        'num_layers': [2, 3, 4],
        'drop_rate': [0.1, 0.2, 0.3],
        'out_dim': [128, 256, 512],
    }   

    # perform grid search
    for model_type in hyperparameters['model_type']:
        for num_layers in hyperparameters['num_layers']:
            for drop_rate in hyperparameters['drop_rate']:
                for out_dim in hyperparameters['out_dim']:
                    # update the config
                    config.model_type = model_type
                    config.num_layers = num_layers
                    config.drop_rate = drop_rate
                    config.out_dim = out_dim

                    # run the test_demodata.py script
                    os.system(f"python test_demodata.py --config {config} --model_type {model_type} --num_layers {num_layers} --drop_rate {drop_rate} --out_dim {out_dim}")

# Main function
if __name__ == "__main__":
    args = parse_args()
    print('Parsed arguments')

    # perform grid search
    print('Performing grid search on the parameters')
    grid_search(args)
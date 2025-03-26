# Create a script to tune hyperparameters for the gRNAde model using the test_demodata.py script

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import argparse
import yaml
import json
from test_demodata import run_benchmark, instantiate_gRNAde, parse_args as parse_test_args

# Define functions
def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for gRNAde model')
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    
    # Add arguments for each hyperparameter
    parser.add_argument("--model_type", type=str, nargs='+', default=["ARv1", "ARv2"],
                      help='List of model types to test')
    parser.add_argument("--temperature", type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.5],
                      help='List of temperature values to test')
    parser.add_argument("--max_temperature", type=float, nargs='+', default=[0.5],
                      help='List of max temperature values to test')
    parser.add_argument("--temperature_factor", type=float, nargs='+', default=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                      help='List of temperature factor values to test')
    parser.add_argument("--beam_width", type=int, nargs='+', default=[1],
                      help='List of beam width values to test')
    parser.add_argument("--beam_branch", type=int, nargs='+', default=[1],
                      help='List of beam branch values to test')
    parser.add_argument("--sampling_strategy", type=str, nargs='+', default=["categorical", "top_p", "top_k", "min_p"],
                      help='List of sampling strategies to test')
    parser.add_argument("--sampling_value", type=float, nargs='+', default=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 2.0, 3.0],
                      help='List of sampling values to test')
    
    # Add argument for output directory
    parser.add_argument("--output_dir", type=str, default="tutorial/tests/hyperparameters",
                      help='Directory to save results')
    
    return parser.parse_args()  

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# perform hyperparameter tuning using grid search
def grid_search(args):
    # define the hyperparameters to tune
    hyperparameters = {
        'model_type': args.model_type,
        'sampling_strategy': args.sampling_strategy,
        'sampling_value': args.sampling_value,
        'beam_width': args.beam_width,
        'beam_branch': args.beam_branch,
        'temperature': args.temperature,
        'max_temperature': args.max_temperature,
        'temperature_factor': args.temperature_factor,        
    }   

    # Print hyperparameters being tested
    print("\nHyperparameters to test:")
    for param, values in hyperparameters.items():
        print(f"{param}: {values}")

    # Initialize empty dataframes to store results
    all_benchmark_results = pd.DataFrame()
    all_summary_results = pd.DataFrame()

    # perform grid search
    for model_type in hyperparameters['model_type']:
        for temp in hyperparameters['temperature']:
            for max_temp in hyperparameters['max_temperature']:
                for temp_factor in hyperparameters['temperature_factor']:
                    for beam_w in hyperparameters['beam_width']:
                        for beam_b in hyperparameters['beam_branch']:
                            for sampling in hyperparameters['sampling_strategy']:
                                for sampling_value in hyperparameters['sampling_value']:
                                        # Set sampling parameters based on strategy
                                        if sampling == 'top_p' and sampling_value > 1.0:
                                            df, df_sample = run_benchmark_with_params(
                                                model_type, temp, max_temp, temp_factor,
                                                beam_w, beam_b, sampling, sampling_value,
                                                all_benchmark_results=all_benchmark_results,
                                                all_summary_results=all_summary_results
                                            )
                                            all_benchmark_results = pd.concat([all_benchmark_results, df], ignore_index=True)
                                            all_summary_results = pd.concat([all_summary_results, df_sample], ignore_index=True)
                                        elif sampling == 'min_p' and sampling_value < 0.5:
                                            df, df_sample = run_benchmark_with_params(
                                                model_type, temp, max_temp, temp_factor,
                                                beam_w, beam_b, sampling, sampling_value,
                                                all_benchmark_results=all_benchmark_results,
                                                all_summary_results=all_summary_results
                                            )
                                            all_benchmark_results = pd.concat([all_benchmark_results, df], ignore_index=True)
                                            all_summary_results = pd.concat([all_summary_results, df_sample], ignore_index=True)
                                        elif sampling == 'top_k' and int(sampling_value) > 1:
                                            sampling_value = int(sampling_value)
                                            df, df_sample = run_benchmark_with_params(
                                                model_type, temp, max_temp, temp_factor,
                                                beam_w, beam_b, sampling, sampling_value,
                                                all_benchmark_results=all_benchmark_results,
                                                all_summary_results=all_summary_results
                                            )
                                            all_benchmark_results = pd.concat([all_benchmark_results, df], ignore_index=True)
                                            all_summary_results = pd.concat([all_summary_results, df_sample], ignore_index=True)
    # Save final results
    os.makedirs(args.output_dir, exist_ok=True)

    benchmark_file = os.path.join(args.output_dir, "benchmark_hyperparams.csv")
    summary_file = os.path.join(args.output_dir, "summary_hyperparams.csv")
    params_file = os.path.join(args.output_dir, "hyperparameters.json")

    all_benchmark_results.to_csv(benchmark_file, index=False)
    all_summary_results.to_csv(summary_file, index=False)
    
    # Save hyperparameters used
    with open(params_file, 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    print(f"\nFinal results saved to:")
    print(f"Benchmark results: {benchmark_file}")
    print(f"Summary results: {summary_file}")
    print(f"Hyperparameters used: {params_file}")

def run_benchmark_with_params(model_type, temperature, max_temperature, temperature_factor,
                            beam_width, beam_branch, sampling_strategy, sampling_value,
                            all_benchmark_results=None, all_summary_results=None):
    print(f"\nRunning benchmark with parameters:")
    print(f"model_type: {model_type}")
    print(f"temperature: {temperature}")
    print(f"max_temperature: {max_temperature}")
    print(f"temperature_factor: {temperature_factor}")
    print(f"beam_width: {beam_width}")
    print(f"beam_branch: {beam_branch}")
    print(f"sampling_strategy: {sampling_strategy}")
    print(f"sampling_value: {sampling_value}")

    # Create test arguments with current hyperparameters
    test_args = parse_test_args()
    test_args.model_type = model_type
    test_args.temperature = temperature
    test_args.max_temperature = max_temperature
    test_args.temperature_factor = temperature_factor
    test_args.beam_width = beam_width
    test_args.beam_branch = beam_branch
    test_args.sampling_strategy = sampling_strategy
    test_args.sampling_value = sampling_value

    # Instantiate gRNAde with current parameters
    grnade_module = instantiate_gRNAde(test_args)

    # Run benchmark and get results
    df, df_sample = run_benchmark(test_args, grnade_module)

    # Add hyperparameters to results
    df['model_type'] = model_type
    df['sampling_strategy'] = sampling_strategy
    df['sampling_value'] = sampling_value
    df['beam_width'] = beam_width
    df['beam_branch'] = beam_branch
    df_sample['temperature'] = temperature
    df_sample['max_temperature'] = max_temperature
    df_sample['temperature_factor'] = temperature_factor

    return df, df_sample

# Main function
if __name__ == "__main__":
    args = parse_args()
    print('Parsed arguments')

    # perform grid search
    print('Performing grid search on the parameters')
    grid_search(args)
# Import libraries and set up the environment

import sys
sys.path.append('../')

import dotenv
dotenv.load_dotenv("../.env")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Try to import directly from the absolute path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import constants after setting up paths
try:
    from src.constants import PROJECT_PATH
except ImportError:
    # Fallback if import still fails
    PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Import the gRNAde module
from gRNAde import gRNAde

def create_test_dir():
    if not os.path.exists(os.path.join(PROJECT_PATH, "tutorial/tests")):
        os.makedirs(os.path.join(PROJECT_PATH, "tutorial/tests"))

def create_output_filepath(args):
    # Create output filepath if not provided
    if args.output_filepath is None:
        # Get base name of input (either file or directory)
        if os.path.isdir(args.pdb_input):
            base_name = os.path.basename(os.path.normpath(args.pdb_input))
        else:
            base_name = os.path.basename(args.pdb_input).split('.')[0]
        
        # Create descriptive filename with sampling parameters
        sampling_value = ""
        if args.sampling_strategy == "min_p":
            sampling_value = f"minp{args.min_p_sampling}"
        elif args.sampling_strategy == "top_k":
            sampling_value = f"topk{args.top_k_sampling}"
        elif args.sampling_strategy == "top_p":
            sampling_value = f"topp{args.top_p_sampling}"
        
        output_filepath = f"{base_name}_conf{args.max_num_conformers}_{args.sampling_strategy}_{sampling_value}_beam{args.beam_width}x{args.beam_branch}.fasta"
        output_filepath = os.path.join(PROJECT_PATH, "tutorial/tests", output_filepath)
    else:
        output_filepath = os.path.join(PROJECT_PATH, "tutorial/tests", args.output_filepath)
    return output_filepath

# Choose the appropriate design function based on input type
def choose_design_function(args):
    if os.path.isdir(args.pdb_input):
        if args.max_num_conformers <= 1:
            print('Error: max_num_conformers must be greater than 1 when pdb_input is a directory')
            import sys
            sys.exit(1)
        
        # Use directory-based design
        design_function = gRNAde_module.design_from_directory
        sampling_params['directory_filepath'] = args.pdb_input
    else:
        # Use single PDB file design
        design_function = gRNAde_module.design_from_pdb_file
        sampling_params['pdb_filepath'] = os.path.join(PROJECT_PATH, args.pdb_input)
    return design_function, sampling_params

# 6UGG 1 A, 6UGI 1 A
# add argparse to specify test directory
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="tutorial/tests")
    parser.add_argument("--pdb_input", type=str, default="tutorial/demo_data/4FE5_1_B.pdb")
    parser.add_argument("--output_filepath", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling_strategy", type=str, default="categorical")
    parser.add_argument("--top_k_sampling", type=int, default=2)
    parser.add_argument("--top_p_sampling", type=float, default=0.9)
    parser.add_argument("--min_p_sampling", type=float, default=0.05)
    parser.add_argument("--beam_width", type=int, default=2)
    parser.add_argument("--beam_branch", type=int, default=6)
    parser.add_argument("--split", type=str, default="das")
    parser.add_argument("--max_num_conformers", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    return args

def write_output_file(output_filepath, args, sequences):
    with open(output_filepath, 'w') as f:
        # Write configuration header as a comment
        f.write(f"# Configuration:\n")
        f.write(f"# model_split: {args.split}\n")
        f.write(f"# max_num_conformers: {args.max_num_conformers}\n")
        f.write(f"# sampling_strategy: {args.sampling_strategy}\n")
        f.write(f"# temperature: {args.temperature}\n")
        f.write(f"# seed: {args.seed}\n")
        if args.sampling_strategy == "top_k":
            f.write(f"# top_k_sampling: {args.top_k_sampling}\n")
        elif args.sampling_strategy == "top_p":
            f.write(f"# top_p_sampling: {args.top_p_sampling}\n")
        elif args.sampling_strategy == "min_p":
            f.write(f"# min_p_sampling: {args.min_p_sampling}\n")
        f.write(f"# beam_width: {args.beam_width}\n")
        f.write(f"# beam_branch: {args.beam_branch}\n")
        f.write(f"# n_samples: {args.n_samples}\n\n")
        
        # Write sequences
        for seq in sequences:
            f.write(seq.format('fasta'))


if __name__ == "__main__":
    # Create test directory if it doesn't exist
    create_test_dir()
    print('Created test directory')

    # Parse arguments
    args = parse_args()
    print('Parsed arguments')
    # Create an instance of gRNAde
    gRNAde_module = gRNAde(split=args.split, max_num_conformers=args.max_num_conformers, gpu_id=args.gpu_id)
    print('Created gRNAde instance')

    # Prepare common sampling parameters
    sampling_params = {
        'n_samples': args.n_samples,
        'temperature': args.temperature,
        'seed': args.seed,
        'sampling_strategy': args.sampling_strategy,
        'top_k_sampling': args.top_k_sampling,
        'top_p_sampling': args.top_p_sampling,
        'min_p_sampling': args.min_p_sampling,
        'beam_width': args.beam_width,
        'beam_branch': args.beam_branch
    }

    # Choose the appropriate design function based on input type
    design_function, sampling_params = choose_design_function(args)

    # Create output filepath
    output_filepath = create_output_filepath(args)
    print(f"Output filepath: {output_filepath}")
    
    # Perform sampling with the selected function and parameters
    sequences, samples, perplexity, recovery_sample, sc_score = design_function(**sampling_params)
    print('Performed sampling')
    
    # Print the sequences
    for seq in sequences:
        print(seq.format('fasta'))
    
    # create output filepath if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Write configuration and sequences to output file
    print(f"Writing sequences to {output_filepath}")
    write_output_file(output_filepath, args, sequences)
    print(f"Sequences and configuration written to {output_filepath}")
# Import libraries and set up the environment

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import dotenv
dotenv.load_dotenv("../.env")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from src.constants import PROJECT_PATH, DATA_PATH, RNA_CORR

# Import the gRNAde module
from gRNAde import gRNAde

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='RNA sequence generator using gRNAde')
    parser.add_argument('--run_type', dest='run_type', default='single', type=str, 
                        help='Type of run: "single" or "multi" (default: single)')
    parser.add_argument('--input_path', dest='input_path', type=str, default=None,
                        help='Path to input PDB file or directory')
    parser.add_argument('--output_path', dest='output_path', type=str, default=None,
                        help='Path to output FASTA file')
    parser.add_argument('--max_num_conformers', dest='max_num_conformers', type=int, default=1,
                        help='Maximum number of conformers (default: 1)')
    parser.add_argument('--split', dest='split', default='das', type=str,
                        help='Split type (default: das)')
    parser.add_argument('--n_samples', dest='n_samples', default=16, type=int,
                        help='Number of samples (default: 16)')
    parser.add_argument('--temperature', dest='temperature', default=1.0, type=float,
                        help='Temperature for sampling (default: 1.0)')
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help='Random seed (default: 0)')
    parser.add_argument('--gpu_id', dest='gpu_id', default=0, type=int,
                        help='GPU ID (default: 0)')
    parser.add_argument('--avoid_sequences', dest='avoid_sequences', nargs='+', default=None, 
                        help='List of sequences to avoid during design')
    parser.add_argument('--filename', dest='filename', type=str, default=None,
                        help='PDB filename without extension (for single mode)')
    parser.add_argument('--directory', dest='directory', type=str, default=None,
                        help='Directory containing PDB files (for multi mode)')
    parser.add_argument('--use_nullomers', dest='use_nullomers', default=True, type=bool,
                        help='Use nullomers (default: True)')
    args = parser.parse_args()

    # Print all arguments in a formatted way
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    if args.avoid_sequences:
        print(f"Avoiding sequences: {args.avoid_sequences}")

    # Create an instance of gRNAde with appropriate parameters
    gRNAde_module = gRNAde(
        split=args.split, 
        max_num_conformers=args.max_num_conformers if args.run_type.lower() == 'multi' else 1, 
        gpu_id=args.gpu_id
    )

    if args.run_type.lower() == 'multi':
        # Set default directory path if not provided
        # Priority: input_path > directory > default path
        directory_filepath = args.input_path or args.directory or os.path.join(PROJECT_PATH, args.directory)
        output_filepath = args.output_path or os.path.join(PROJECT_PATH, "results/outputs/"+args.directory+"_multistate.fasta")
        
        # Design example usage for multi-state
        sequences, samples, perplexity, recovery_sample, sc_score = gRNAde_module.design_from_directory(
            directory_filepath=directory_filepath,
            output_filepath=output_filepath,
            n_samples=args.n_samples,
            temperature=args.temperature,
            seed=args.seed,
            avoid_sequences=args.avoid_sequences
        )
    
    else:
        # Single-state design
        filename = args.filename
        
        # Use input_path if provided, otherwise use default
        # Priority: input_path > filename-based path > default
        if args.input_path:
            pdb_filepath = args.input_path
        elif filename:
            pdb_filepath = os.path.join(PROJECT_PATH, f"results/demo_data/{filename}.pdb")
        else:
            # Default to a sample file if neither input_path nor filename is provided
            pdb_filepath = os.path.join(PROJECT_PATH, "results/demo_data/5T2A_1_D.pdb")
            filename = "5T2A_1_D"  # Set default filename for output
            
        output_filepath = args.output_path or os.path.join(PROJECT_PATH, f"results/outputs/{filename or 'output'}.fasta")
        
        sequences, samples, perplexity, recovery_sample, sc_score = gRNAde_module.design_from_pdb_file(
            pdb_filepath=pdb_filepath,
            output_filepath=output_filepath,
            n_samples=args.n_samples,
            temperature=args.temperature,
            seed=args.seed,
            avoid_sequences=args.avoid_sequences,
            use_nullomers=args.use_nullomers
        )

    # Print sequence in FASTA format
    for i, seq in enumerate(sequences):
        print(f"Sequence {i+1}:")
        print(seq.format('fasta'))
        
    # Print additional metrics
    print(f"\nPerplexity: {perplexity}")
    print(f"Self-consistency score: {sc_score}")
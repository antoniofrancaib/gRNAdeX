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

from src.constants_mod import PROJECT_PATH, DATA_PATH

# Import the gRNAde module
from gRNAde_mod import gRNAde

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_type', dest='run_type', default='single', type=str, required=True)
    #parser.add_argument('--input_path', dest='input_path', type=str, required=True)
    #parser.add_argument('--output_path', dest='output_path', type=str, required=True)
    #parser.add_argument('--max_num_conformers', dest='max_num_conformers', type=int, required=True)
    #parser.add_argument('--split', dest='split', default='das', type=str)
    #parser.add_argument('--n_samples', dest='n_samples', default=16, type=int)
    #parser.add_argument('--temperature', dest='temperature', default=1.0, type=float)
    #parser.add_argument('--seed', dest='seed', default=0, type=int)
    #parser.add_argument('--gpu_id', dest='gpu_id', default=0, type=int)
    
    args, unknown = parser.parse_known_args()

    # Print all arguments in a formatted way
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    if args.run_type.lower() == 'multi':
        # Create an instance of multi-state gRNAde
        gRNAde_module = gRNAde(split='das', max_num_conformers=3, gpu_id=0)

        # Design example usage
        sequences, samples, perplexity, recovery_sample, sc_score = gRNAde_module.design_from_directory(
            directory_filepath = os.path.join(PROJECT_PATH, "tutorial/demo_data/adenine_riboswitch/"),
            output_filepath = os.path.join(PROJECT_PATH, "tutorial/outputs/demo_output_multistate.fasta"),
            n_samples = 1,
            temperature = 1.0,
            seed = 0
        )
    
    else:
        # Create an instance of gRNAde
        gRNAde_module = gRNAde(split='das', max_num_conformers=1, gpu_id=0)

        # Single-state design example usage
        sequences, samples, perplexity, recovery_sample, sc_score = gRNAde_module.design_from_pdb_file(
            pdb_filepath = os.path.join(PROJECT_PATH, "tutorial/demo_data/4FE5_1_B.pdb"),
            output_filepath = os.path.join(PROJECT_PATH, "tutorial/outputs/demo_output.fasta"),
            n_samples = 16,
            temperature = 1.0,
            seed = 0
        )

    # Print sequence in FASTA format
    for seq in sequences:
        print(seq.format('fasta'))
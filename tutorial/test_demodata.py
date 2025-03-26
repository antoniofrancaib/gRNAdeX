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

def create_test_dir(model_type):
    if not os.path.exists(os.path.join(PROJECT_PATH, "tutorial/tests", model_type)):
        os.makedirs(os.path.join(PROJECT_PATH, "tutorial/tests", model_type))

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
            sampling_value = f"{args.min_p_sampling}"
        elif args.sampling_strategy == "top_k":
            sampling_value = f"{args.top_k_sampling}"
        elif args.sampling_strategy == "top_p":
            sampling_value = f"{args.top_p_sampling}"
        
        output_filepath = f"{base_name}_conf{args.max_num_conformers}_{args.sampling_strategy}_{sampling_value}_beam{args.beam_width}x{args.beam_branch}_temp{args.temperature}_seed{args.seed}_max_temp{args.max_temperature}_temp_factor{args.temperature_factor}.fasta"
        output_filepath = os.path.join(PROJECT_PATH, "tutorial/tests", args.model_type, output_filepath)
    else:
        output_filepath = os.path.join(PROJECT_PATH, "tutorial/tests", args.model_type, args.output_filepath)
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
        f.write(f"# n_samples: {args.n_samples}\n")
        f.write(f"# max_temperature: {args.max_temperature}\n")
        f.write(f"# temperature_factor: {args.temperature_factor}\n\n")
        
        # Write sequences
        for seq in sequences:
            f.write(seq.format('fasta'))

def create_benchmark_output_filepath(args):
    # Create descriptive filename with sampling parameters
    sampling_value = ""
    if args.sampling_strategy == "min_p":
        sampling_value = f"minp{args.min_p_sampling}"
    elif args.sampling_strategy == "top_k":
        sampling_value = f"topk{args.top_k_sampling}"
    elif args.sampling_strategy == "top_p":
        sampling_value = f"topp{args.top_p_sampling}"
    
    output_filepath = f"benchmark_conf{args.max_num_conformers}_{args.sampling_strategy}_{sampling_value}_temp{args.temperature}_beam{args.beam_width}x{args.beam_branch}_seed{args.seed}_max_temp{args.max_temperature}_temp_factor{args.temperature_factor}.csv"
    return os.path.join(PROJECT_PATH, "tutorial/tests", args.model_type, output_filepath)

def run_benchmark(args, gRNAde_module):
    # Metadata and recoveries from Das et al.
    demo_data_id = ["1CSL","1ET4","1F27","1L2X","1LNT","1Q9A","4FE5","1X9C","1XPE","2GCS","2GDI","2OEU","2R8S","354D"]

    demo_data_info = [
        "RRE high affinity site",
        "Vitamin B12 binding RNA aptamer",
        "Biotin-binding RNA pseudoknot",
        "Viral RNA pseudoknot",
        "RNA internal loop of SRP",
        "Sarcin/ricin domain from E.coli 23S rRNA",
        "Guanine riboswitch aptamer",
        "All-RNA hairpin ribozyme",
        "HIV-1 B RNA dimerization initiation site",
        "Pre-cleavage state of glmS ribozyme",
        "Thiamine pyrophosphate-specific riboswitch",
        "Junctionless hairpin ribozyme",
        "Tetrahymena ribozyme P4-P6 domain",
        "Loop E from E. coli 5S rRNA",
    ]

    vienna_recovery = [0.2566288,0.25852272,0.2951389,0.2448347,0.325,0.27455357,0.2916667,0.26201922,0.2709652,0.24888393,0.25,0.23317307,0.26923078,0.28035715]
    farna_recovery = [0.20,0.34,0.36,0.45,0.27,0.40,0.28,0.31,0.24,0.26,0.38,0.30,0.36,0.35]
    rosetta_recovery = [0.44,0.44,0.37,0.48,0.53,0.41,0.36,0.50,0.40,0.44,0.48,0.37,0.53,0.55]

    # Evaluate gRNAde on the Das et al. data
    grnade_recovery = []
    grnade_perplexity = []
    grnade_sc_score = []
    
    print(f"Running benchmark on Das et al. dataset")
    
    for pdb_filepath in os.listdir(os.path.join(PROJECT_PATH, "tutorial/demo_data/")):
        if pdb_filepath.endswith(".pdb"):
            print(f"Processing {pdb_filepath}...")
            sequences, samples, perplexity, recovery_sample, sc_score = gRNAde_module.design_from_pdb_file(
                pdb_filepath=os.path.join(PROJECT_PATH, f"tutorial/demo_data/{pdb_filepath}"),
                n_samples=args.n_samples,
                temperature=args.temperature,
                seed=args.seed
            )
            grnade_recovery.append(np.mean(recovery_sample))
            grnade_perplexity.append(np.mean(perplexity))
            grnade_sc_score.append(np.mean(sc_score))
            
            # Write sequences to a file for this PDB
            pdb_output_filepath = os.path.join(PROJECT_PATH, "tutorial/tests", args.model_type, 
                                              f"{pdb_filepath.split('.')[0]}_conf{args.max_num_conformers}_{args.sampling_strategy}_temp{args.temperature}_beam{args.beam_width}x{args.beam_branch}_seed{args.seed}.fasta")
            os.makedirs(os.path.dirname(pdb_output_filepath), exist_ok=True)
            write_output_file(pdb_output_filepath, args, sequences)
            print(f"  Sequences written to {pdb_output_filepath}")
        else:
            print(f"Warning: {pdb_filepath} not found, skipping...")
            # Add placeholder values to maintain alignment with other data

    # Collate results as dataframes for plotting
    df = pd.DataFrame.from_dict({
        "idx": list(range(len(farna_recovery))),
        "pdb_id": demo_data_id,
        "description": demo_data_info,
        "vienna_recovery": np.array(vienna_recovery),
        "farna_recovery": np.array(farna_recovery),
        "rosetta_recovery": np.array(rosetta_recovery),
        "grnade_recovery": np.array(grnade_recovery),
        "grnade_perplexity": np.array(grnade_perplexity),
        "grnade_sc_score": np.array(grnade_sc_score),
    })
    
    df_sample = pd.DataFrame.from_dict({
        "idx": list(range(len(farna_recovery) * 4)),
        "mean_recovery": np.array(vienna_recovery + farna_recovery + rosetta_recovery + grnade_recovery),
        "model_name": ["ViennaRNA\n(2D only)"] * len(vienna_recovery) + ["FARNA"] * len(farna_recovery) + ["Rosetta"] * len(rosetta_recovery) + ["gRNAde"] * len(grnade_recovery),
    })
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(PROJECT_PATH, "tutorial/outputs"), exist_ok=True)
    
    # Create benchmark output filepath
    benchmark_output_filepath = create_benchmark_output_filepath(args)
    df.to_csv(benchmark_output_filepath)
    print(f"\nBenchmark results saved to {benchmark_output_filepath}")

    # Print mean recovery by model
    print("\nMean Recovery by Model:")
    print(df_sample.groupby("model_name").mean())
    return df, df_sample

# 6UGG 1 A, 6UGI 1 A
# add argparse to specify test directory
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="tutorial/tests")
    parser.add_argument("--pdb_input", type=str, default="tutorial/demo_data/4FE5_1_B.pdb")
    parser.add_argument("--output_filepath", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_temperature", type=float, default=0.5)
    parser.add_argument("--temperature_factor", type=float, default=0.01)
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
    parser.add_argument("--run_benchmark", default=False, help="Run benchmark on Das et al. dataset")
    parser.add_argument("--model_type", type=str, default="ARv2")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    print('Parsed arguments')

    # Create test directory if it doesn't exist
    create_test_dir(args.model_type)
    print('Created test directory')

    # Create an instance of gRNAde
    gRNAde_module = gRNAde(
        split=args.split,
        max_num_conformers=args.max_num_conformers,
        gpu_id=args.gpu_id,
        sampling_strategy=args.sampling_strategy,
        top_k_sampling=args.top_k_sampling,
        top_p_sampling=args.top_p_sampling,
        min_p_sampling=args.min_p_sampling,
        beam_width=args.beam_width,
        beam_branch=args.beam_branch,
        temperature=args.temperature,
        max_temperature=args.max_temperature,
        temperature_factor=args.temperature_factor,
        model_type=args.model_type
    )
    print('Created gRNAde instance of model type', args.model_type)

    # Check if benchmark should be run
    if args.run_benchmark:
        df, df_sample = run_benchmark(args, gRNAde_module)
        print("Benchmark completed")
    else:
        # Prepare common sampling parameters
        sampling_params = {
            'n_samples': args.n_samples,
            'temperature': args.temperature,
            'seed': args.seed
        }

        # Choose the appropriate design function based on input type
        design_function, sampling_params = choose_design_function(args)

        # Create output filepath
        output_filepath = create_output_filepath(args)
        print(f"Output filepath: {output_filepath}")
        
        # Perform sampling with the selected function and parameters
        sequences, samples, perplexity, recovery_sample, sc_score = design_function(**sampling_params)
        print('Performed sampling')
        
        # create output filepath if it doesn't exist
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        # Write configuration and sequences to output file
        print(f"Writing sequences to {output_filepath}")
        write_output_file(output_filepath, args, sequences)
        print(f"Sequences and configuration written to {output_filepath}")
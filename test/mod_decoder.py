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

from src.constants import PROJECT_PATH, DATA_PATH

# Import the gRNAde module
from gRNAde_mod_decoding import gRNAde

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

for seq in sequences:
    print(seq.format('fasta'))
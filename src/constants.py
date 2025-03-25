import os
import wandb

PROJECT_PATH = os.environ.get("PROJECT_PATH")

DATA_PATH = os.environ.get("DATA_PATH")

X3DNA_PATH = os.environ.get("X3DNA")

ETERNAFOLD_PATH = os.environ.get("ETERNAFOLD")


# Value to fill missing coordinate entries when reading PDB files
FILL_VALUE = 1e-5


# Small epsilon value added to distances to avoid division by zero
DISTANCE_EPS = 0.001


# List of possible atoms in RNA nucleotides
RNA_ATOMS = [
    'P', "C5'", "O5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'",
    'N1', 
    'C2', 
    'O2', 'N2',
    'N3', 
    'C4', 'O4', 'N4',
    'C5', 
    'C6', 
    'O6', 'N6', 
    'N7', 
    'C8', 
    'N9',
    'OP1', 'OP2',
]


# List of possible RNA nucleotides
RNA_NUCLEOTIDES = [
    'A', 
    'G', 
    'C', 
    'U',
    # might have to add N back in later when sampling
    #'N'  # placeholder for missing/unknown nucleotides
]


# List of purine nucleotides
PURINES = ["A", "G"]


# List of pyrimidine nucleotides
PYRIMIDINES = ["C", "U"]


# 
LETTER_TO_NUM = dict(zip(
    RNA_NUCLEOTIDES, 
    list(range(len(RNA_NUCLEOTIDES)))
))


#
NUM_TO_LETTER = {v:k for k, v in LETTER_TO_NUM.items()}


#
DOTBRACKET_TO_NUM = {
    '.': 0,
    '(': 1,
    ')': 2
}


# 3D self-consistency score thresholds for desingability/validity
RMSD_THRESHOLD = 2.0
TM_THRESHOLD = 0.45
GDT_THRESHOLD = 0.50

# Sampling strategies for decoding
SAMPLING_STRATEGY = "min_p"
TOP_K = 2
TOP_P = 0.9
MIN_P = 0.05

# Define beam width and beam branch
BEAM_WIDTH = 2
BEAM_BRANCH = 4

RNA_CORR = {'5T2A_1_D': '5S_rRNA'}

# Records to keep in PDB files
RECORDS_TO_KEEP = ['ATOM']

# Protein residues to be aware of in PDB files
PROT_RESIDUES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", 
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", 
    "THR", "TRP", "TYR", "VAL",  # Standard Amino Acids

    "SEC", "PYL",  # Non-standard Amino Acids

    "PHS", "PHT", "PHY",  # Post-Translational Modifications (PTMs)
    "MEK", "MER", "ACK", "HYP"  # Modified Amino Acids
]

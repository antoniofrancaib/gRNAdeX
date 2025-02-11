#!/bin/bash
#SBATCH -J gRNAde_processing
#SBATCH -A YOUR_ACCOUNT_NAME  # e.g. MLMI-jaf98-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8     # Match num_workers in config
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00         # Extended for 14k PDBs
#SBATCH --partition=icelake
#SBATCH --output=logs/process_%j.log
#SBATCH --error=logs/process_%j.err

### 1. Critical HPC Configuration ###
module purge
module load rhel8/default-icl
module load python/3.8 gcc/11  # Verify with 'module avail'

### 2. Environment Setup ###
export PROJECT_PATH="/rds/user/jaf98/hpc-work/geometric-rna-design"
export X3DNA="$PROJECT_PATH/tools/x3dna-v2.4"
export ETERNAFOLD="$PROJECT_PATH/tools/EternaFold"
export PATH="$X3DNA/bin:$PATH"

# Fix Lua/Slurm plugin issue
unset SLURM_PLUGINS

### 3. Conda Environment ###
source ~/.bashrc
mamba activate rna
pip install python-dotenv wandb  # Ensure critical packages

### 4. WandB Configuration ###
export WANDB_DIR="$PROJECT_PATH/wandb"
mkdir -p $WANDB_DIR
chmod 700 $WANDB_DIR

### 5. Execution Command ###
cd $PROJECT_PATH
python data/process_data.py \
    --config configs/data_config.yaml \
    --tags hpc_csd3 \
    --expt_name rna_processing_v1 \

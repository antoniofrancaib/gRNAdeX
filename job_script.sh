#!/bin/bash
#SBATCH -J forward_pass
#SBATCH -A MLMI-jaf98-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                        
#SBATCH --cpus-per-task=4                  
#SBATCH --time=01:00:00
#SBATCH --output=logs/forward_pass_%j.out
#SBATCH --error=logs/forward_pass_%j.err
#SBATCH -p ampere 

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# --- Environment setup ---
module purge
module load rhel8/default-amp

# Source your bash profile (if it loads conda/venv settings)
source /home/$USER/.bashrc
source /home/jaf98//miniforge3/bin/activate
mamba activate rna

# --- Set project and data paths ---
export PROJECT_PATH="/home/jaf98/rds/hpc-work/geometric-rna-design"
export DATA_PATH="$PROJECT_PATH/data"

cd "$PROJECT_PATH" || exit 1
echo "Current directory: $(pwd)"

# --- Run the forward pass ---
python -u main.py --config configs/eval.yaml --evaluate True

echo "Job finished at: $(date)"
#!/bin/bash
#!
#! Example SLURM job script for GPU usage on CSD3
#! Last updated: 16/02/2025
#!

####### SBATCH directives begin here ###############################
#SBATCH -J gRNAde_eval_gpu              # Job name
#SBATCH -A MLMI-jaf98-SL2-GPU          # Example account for GPU jobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # e.g. 4 CPUs (adjust as needed)
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --time=10:00:00
#SBATCH --mail-type=NONE
#SBATCH --output=logs/forward_pass_%j.out
#SBATCH --error=logs/forward_pass_%j.err
#SBATCH -p ampere                      # Example GPU partition name on CSD3
####### SBATCH directives end here ###############################

# Application options and environment
options="--config configs/eval.yaml --evaluate True --tags hpc_csd3 --expt_name rna_eval_filt_baseline"

# Environment setup
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp      # Adjust to your HPC environment

source /home/$USER/.bashrc
source /home/jaf98//miniforge3/bin/activate
mamba activate rna             # The environment with PyTorch+PyG & your deps

# Python script to run
application="python -u main.py"

# Work directory
workdir="$SLURM_SUBMIT_DIR"

cd "$workdir"
echo -e "Changed directory to $(pwd).\n"
JOBID=$SLURM_JOB_ID

CMD="$application $options > logs/out.eval_gpu 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: $(date)"
echo "Running on master node: $(hostname)"
echo "Current directory: $(pwd)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

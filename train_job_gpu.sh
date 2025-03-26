#!/bin/bash
#!
#! Example SLURM job script for GPU training on CSD3
#! Last updated: 20/02/2025
#!

####### SBATCH directives begin here ###############################
#SBATCH -J gRNAde_train_gpu             # Job name for training
#SBATCH -A MLMI-jaf98-SL2-GPU          # GPU job account
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # Adjust CPU count as needed
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --time=15:00:00                # Adjust training time as needed
#SBATCH --mail-type=NONE
#SBATCH --output=logs/train_job_%j.out
#SBATCH --error=logs/train_job_%j.err
#SBATCH -p ampere                      # GPU partition name on CSD3
####### SBATCH directives end here ###############################

# Application options and environment
options="--config configs/default.yaml --expt_name all_data_hybrid_multi_tensor_pool --tags hybrid_enc,multi_tensor_pool, hybrid_enc, all_data, train"

# Environment setup
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp      

source /home/$USER/.bashrc
source /home/jaf98/miniforge3/bin/activate
mamba activate rna             

# Python script to run (training)
application="python -u main.py"

# Work directory
workdir="$SLURM_SUBMIT_DIR"

cd "$workdir"
echo -e "Changed directory to $(pwd).\n"
JOBID=$SLURM_JOB_ID

CMD="$application $options > logs/out.train_job 2>&1"

echo -e "JobID: $JOBID\n======"
echo "Time: $(date)"
echo "Running on master node: $(hostname)"
echo "Current directory: $(pwd)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

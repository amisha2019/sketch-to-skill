#!/bin/bash
#SBATCH --job-name=Reach_seed1
#SBATCH --output=/fs/nexus-projects/Sketch_VLM_RL/amishab/pofd/slurm_logs/%x.%j.out
#SBATCH --time=2:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --account=scavenger

# Load necessary modules and set environment paths
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate ibrl
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

# Change directory to the folder where train_rl_mw.py is located
cd /fs/nexus-scratch/amishab/Teacher_student_RLsketch/mw_main

# Run the training script with the specified parameters
srun bash -c "python train_rl_mw_pofd.py --config_path /fs/nexus-scratch/amishab/Teacher_student_RLsketch/release/cfgs/metaworld/ibrl_basic.yaml --bc_policy Reach --seed 1"
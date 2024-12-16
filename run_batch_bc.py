import subprocess
import os
from datetime import datetime

# Define the root directory for saving scripts and logs
root_dir = "/fs/nexus-projects/Sketch_VLM_RL/amishab/BC_sketch_RL/BC_demo3_gen_10"
os.makedirs(f'{root_dir}/slurm_scripts', exist_ok=True)
os.makedirs(f'{root_dir}/slurm_logs', exist_ok=True)

# Define the path to the subfolder where train_bc_mw.py is located
train_script_path = "/fs/nexus-scratch/amishab/Teacher_student_RLsketch/mw_main"

# Define environments and seeds
# environments = ["ButtonPress", "Reach", "ReachWall", "ButtonPressTopdownWall"]
environments = ["Assembly"]
# environments = ["BoxClose"]
seeds = [0]

# SLURM job parameters
partition = "scavenger"
qos = "scavenger"
time = "00:30:00"
memory = "32gb"
gres = "gpu:1"
account = "scavenger"

for env in environments:
    for seed in seeds:
        job_name = f"{env}_seed{seed}"
        job_dir = f"{root_dir}/{job_name}"
        # os.makedirs(job_dir, exist_ok=True)

        # Construct the command to run the Python script with arguments
        python_command = (
            f"python train_bc_orig.py "
            f"--dataset.path {env} "
            f"--dataset.num_data 30 "
            f"--num_epoch 5 "
            f"--seed {seed}"
        )

        # Create the SLURM job script content
        job_script_content = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={root_dir}/slurm_logs/%x.%j.out
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={memory}
#SBATCH --gres={gres}
#SBATCH --account={account}

# Load necessary modules and set environment paths
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate ibrl
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

# Change directory to the folder where train_bc_metaworld.py is located
cd {train_script_path}

# Run the training script with the specified parameters
srun bash -c "{python_command}"
'''

        # Write the job script to a file
        job_script_path = f'{root_dir}/slurm_scripts/submit_job__{job_name}.sh'
        with open(job_script_path, 'w') as job_script_file:
            job_script_file.write(job_script_content)

        # Submit the job using sbatch
        subprocess.run(['sbatch', job_script_path])

        # Print the job submission info
        print(f'Job submitted for environment: {env}, Seed: {seed}')
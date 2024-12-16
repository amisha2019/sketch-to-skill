import subprocess
import os
import numpy as np
import itertools

# Directory setup
root_dir = "/fs/nexus-projects/Sketch_VLM_RL/amishab/PID_experiments"
os.makedirs(f'{root_dir}/slurm_scripts', exist_ok=True)
os.makedirs(f'{root_dir}/slurm_logs', exist_ok=True)
def generate_pid_combinations():
    Kp_range = np.linspace(10, 20, 2)  # Example range for Kp
    Ki_range = np.linspace(0.001, 0.01, 5)  # Example range for Ki
    Kd_range = np.linspace(0.001, 0.01, 5)  # Example range for Kd

    return itertools.product(Kp_range, Ki_range, Kd_range)
def submit_jobs_for_pid_parameters(pid_combinations):
    for Kp, Ki, Kd in pid_combinations:
        job_name = f"PID_{Kp}_{Ki}_{Kd}"
        job_dir = f"{root_dir}/{job_name}"
        os.makedirs(job_dir, exist_ok=True)

        job_script_content = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={root_dir}/slurm_logs/%x.%j.out
#SBATCH --time=00:15:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH --account=scavenger

# Environment setup
# Load necessary modules and set environment paths
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate ibrl
export PYTHONPATH=$PWD:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_PY_MUJOCO_PATH=~/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin

# Change directory to the script location
cd /fs/nexus-scratch/amishab/Teacher_student_RLsketch/teacher_model/fit_traj

# Run Python script
srun python datasetv2.py --Kp {Kp} --Ki {Ki} --Kd {Kd}
'''

        job_script_path = f'{root_dir}/slurm_scripts/{job_name}.sh'
        with open(job_script_path, 'w') as file:
            file.write(job_script_content)

        # Submit the job
        subprocess.run(['sbatch', job_script_path])
        print(f'Job submitted for PID params: Kp={Kp}, Ki={Ki}, Kd={Kd}')

pid_combinations = generate_pid_combinations()
submit_jobs_for_pid_parameters(pid_combinations)

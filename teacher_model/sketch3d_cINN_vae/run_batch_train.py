import subprocess
import os
from itertools import product

# Define the root directory for saving scripts and logs
root_dir = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/{os.environ.get('USER')}/VAE_AugNew"
os.makedirs(f'{root_dir}/slurm_scripts', exist_ok=True)
os.makedirs(f'{root_dir}/slurm_logs', exist_ok=True)

# Define the parameter grid for ablation
parameters = {
    "num_epochs": [200],
    "scheduler": ["cosine", "onecycle"],
    "lr": [1e-3],
    # "gamma": [0.5],
    "bs": [256],
    "M_N": [0.0001],  #0.00025
}

param_names = list(parameters.keys())
param_values = [v for v in parameters.values()]
combinations = list(product(*param_values))

# Iterate over parameter combinations
jobs_num = 0
for combo in combinations:
    param_dict = {key: value for key, value in zip(param_names, combo)}

    # Construct a unique job name based on the parameters
    job_name = f"train_bs{param_dict['bs']}_{param_dict['scheduler']}_lr{param_dict['lr']}_kld{param_dict['M_N']}"

    # Construct the command to run the Python script with arguments
    python_command = (
        f"python train_vae.py --num_epochs {param_dict['num_epochs']} "
        f"--scheduler {param_dict['scheduler']} "
        f"--lr {param_dict['lr']} "
        f"--bs {param_dict['bs']} "
        f"--M_N {param_dict['M_N']} "
    )

    # SLURM job parameters based on your srun command
    # account = "scavenger"
    # partition = "scavenger"
    # qos = "scavenger"
    account = "cml-scavenger"
    partition = "cml-scavenger"
    qos = "cml-scavenger"
    time = "8:00:00"
    memory = "64gb"
    gres = "gpu:1"

    # Create the SLURM job script content
    job_script_content = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={root_dir}/slurm_logs/%x.%j.out
#SBATCH --time={time}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={memory}
#SBATCH --gres={gres}

# Load necessary modules and activate the Python environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate ibrl

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
    result = ", ".join([f"{name}: {value}" for name, value in zip(param_names, combo)])
    print(f'Job submitted for parameters: {result}')

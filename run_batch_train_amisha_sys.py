import subprocess
import os
from itertools import product
from datetime import datetime

# Define the root directory for saving scripts and logs
root_dir = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/asingh/training_results_vae"
os.makedirs(f'{root_dir}/slurm_scripts', exist_ok=True)
os.makedirs(f'{root_dir}/slurm_logs', exist_ok=True)

# Define the path to the subfolder where train_vae.py is located
train_script_path = "/fs/nexus-scratch/amishab/Teacher_student_RLsketch/teacher_model/sketch3d_cINN_vae"

# Parameter grid for training
parameters = {
    "batch_size": [128, 256],  # Batch sizes to experiment with
    "num_samples": [200000],  # Number of samples for training
    "lr": [1e-4, 5e-4],  # Learning rates
    "latent_dim": [128, 256],  # Latent dimensions for the VAE model
    "kld_weight": [0.1, 0.01, 0.0001, 0.00001],  # KLD weights
    "epochs": [500, 800]  # Number of training epochs
}

param_names = list(parameters.keys())
param_values = [v for v in parameters.values()]
combinations = list(product(*param_values))

# Iterate over parameter combinations
for combo in combinations:
    param_dict = {key: value for key, value in zip(param_names, combo)}

    # Construct a unique job name and token based on the parameters
    job_name = f"train_lr{param_dict['lr']}_latent{param_dict['latent_dim']}_kld{param_dict['kld_weight']}_epochs{param_dict['epochs']}"
    
    # Unique directory for saving the results for each parameter set
    unique_token = f"{job_name}_time{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    job_dir = f"{root_dir}/{unique_token}"
    os.makedirs(job_dir, exist_ok=True)

    # Construct the command to run the Python script with arguments
    python_command = (
        f"python train_vae.py "
        f"--lr {param_dict['lr']} "
        f"--latent_dim {param_dict['latent_dim']} "
        f"--kld_weight {param_dict['kld_weight']} "
        f"--epochs {param_dict['epochs']} "
        f"--train"
    )

    # SLURM job parameters
    partition = "tron"
    qos = "medium"
    time = "20:10:00"
    memory = "64gb"
    gres = "gpu:1"

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

# Load necessary modules and activate the Python environment
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate ibrl

# Change directory to the folder where train_vae.py is located
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
    result = ", ".join([f"{name}: {value}" for name, value in zip(param_names, combo)])
    print(f'Job submitted for parameters: {result}')

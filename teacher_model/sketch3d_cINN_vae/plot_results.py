import os
import numpy as np
import matplotlib.pyplot as plt

def plot_losses(exp_dict):
    """
    Plot training and validation losses for multiple experiments.

    Args:
    exp_dict (dict): Dictionary with experiment names as keys and folder paths as values.
    """
    # Get all loss types
    loss_types = set()
    for folder in exp_dict.values():
        train_loss_file = os.path.join(folder, 'train_losses.npy')
        if os.path.exists(train_loss_file):
            train_losses = np.load(train_loss_file, allow_pickle=True).item()
            loss_types.update(train_losses.keys())

    # Create subplots for each loss type
    n_plots = len(loss_types)
    fig, axs = plt.subplots(n_plots, 1, figsize=(6, 3*n_plots), sharex=True)
    if n_plots == 1:
        axs = [axs]

    colors = plt.cm.rainbow(np.linspace(0, 1, len(exp_dict)))

    for (exp_name, folder), color in zip(exp_dict.items(), colors):
        train_loss_file = os.path.join(folder, 'train_losses.npy')
        val_loss_file = os.path.join(folder, 'val_losses.npy')

        if os.path.exists(train_loss_file) and os.path.exists(val_loss_file):
            train_losses = np.load(train_loss_file, allow_pickle=True).item()
            val_losses = np.load(val_loss_file, allow_pickle=True).item()

            epochs = range(1, len(train_losses['loss']) + 1)

            for i, loss_type in enumerate(loss_types):
                if loss_type in train_losses and loss_type in val_losses:
                    label_train = f'{exp_name} - Train' if i == 0 else None
                    label_val = f'{exp_name} - Val' if i == 0 else None
                    axs[i].semilogy(epochs, np.mean(train_losses[loss_type], axis=1), 
                                    color=color, linestyle='-', label=label_train)
                    axs[i].semilogy(epochs, np.mean(val_losses[loss_type], axis=1), 
                                    color=color, linestyle='--', label=label_val)
                axs[i].set_title(f'{loss_type} Loss')
                axs[i].set_ylabel('Loss (log scale)')
                axs[i].grid(True, which="both", ls="-", alpha=0.5)
                axs[i].set_yscale('log')
                # Set y-axis upper limit to 10^-2
                # axs[i].set_ylim(top=1e-2)
        else:
            print(f"Loss files not found for experiment: {exp_name}")

    axs[-1].set_xlabel('Epochs')
    plt.tight_layout()
    
    # Move legend to the right of the figure
    fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Adjust layout to make room for the legend
    plt.subplots_adjust(right=0.85)
    
    plt.savefig('eval_vae_results/vae_loss_comparison.png', bbox_inches='tight')
    plt.close()

# Example usage:

# "Exp1": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_3D/vae_2024-09-19_03-35-27_lr1e-3_gamma0.5",
# "Exp2": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_3D/vae_2024-09-19_13-56-26_lr1e-3_gamma0.5_bs512_aug",
# "Exp3": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_3D/vae_2024-09-19_14-49-37_lr1e-3_gamma0.5_bs512_aug",
# "Exp4": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_3D/vae_2024-09-19_15-45-30_lr1e-3_gamma0.5_bs1024_aug"

if __name__ == "__main__":
    # exp_dict = {
    #     "Exp1": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-41-33_ep200_onecycle_lr0.001_bs256_kld0.0001_aug",
    #     # "Exp2": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-41-33_ep200_onecycle_lr0.001_bs256_kld0.00025_aug",
    #     # "Exp3": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-41-34_ep200_onecycle_lr0.001_bs256_kld0.0005_aug",
    #     # "Exp4": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-41-34_ep200_onecycle_lr0.001_bs512_kld0.0001_aug",
    #     # "Exp5": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-41-34_ep200_onecycle_lr0.001_bs512_kld0.0005_aug",
    #     # "Exp6": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-41-34_ep200_onecycle_lr0.001_bs512_kld0.00025_aug",

    #     "Exp7": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-41-35_ep200_onecycle_lr0.002_bs256_kld0.0001_aug",
    #     # "Exp8": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-41-35_ep200_onecycle_lr0.002_bs256_kld0.00025_aug",
    #     # "Exp9": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-41-35_ep200_onecycle_lr0.002_bs512_kld0.0005_aug",
    #     # "Exp10": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-41-35_ep200_onecycle_lr0.002_bs512_kld0.00025_aug",
    #     # "Exp11": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-41-36_ep200_onecycle_lr0.002_bs256_kld0.0005_aug",
    #     # "Exp12": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-41-36_ep200_onecycle_lr0.002_bs512_kld0.0001_aug",
        
    #     # "Exp13": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-42-23_ep200_onecycle_lr0.0005_bs256_kld0.0001_aug",
    #     # "Exp14": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-42-23_ep200_onecycle_lr0.0005_bs256_kld0.00025_aug",
    #     # "Exp15": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-43-00_ep200_onecycle_lr0.0005_bs256_kld0.0005_aug",
    #     # "Exp16": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-43-00_ep200_onecycle_lr0.0005_bs512_kld0.0001_aug",
    #     # "Exp17": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-43-00_ep200_onecycle_lr0.0005_bs512_kld0.0005_aug",
    #     # "Exp18": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-43-00_ep200_onecycle_lr0.0005_bs512_kld0.00025_aug",
    # }

    exp_dict = {
        # "Exp1": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-04_ep200_cosine_lr0.001_bs256_kld0.001_aug",
        "Exp2": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-04_ep200_cosine_lr0.001_bs256_kld0.0001_aug",
        # "Exp3": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-09_ep200_cosine_lr0.001_bs256_kld0.01_aug",
        # "Exp4": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-09_ep200_cosine_lr0.001_bs512_kld0.01_aug",
        # "Exp5": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-09_ep200_cosine_lr0.001_bs512_kld0.001_aug",
        # "Exp6": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-09_ep200_cosine_lr0.001_bs512_kld0.0001_aug",
        # "Exp7": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-10_ep200_cosine_lr0.001_bs1024_kld0.01_aug",
        # "Exp8": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-13_ep200_cosine_lr0.001_bs1024_kld0.001_aug",
        # "Exp9": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-13_ep200_cosine_lr0.001_bs1024_kld0.0001_aug",

        # "Exp10": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-18_ep200_cosine_lr0.0005_bs256_kld0.01_aug",
        # "Exp11": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-18_ep200_cosine_lr0.0005_bs256_kld0.001_aug",
        # "Exp12": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-21_ep200_cosine_lr0.0005_bs256_kld0.0001_aug",
        # "Exp13": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-16_ep200_cosine_lr0.0005_bs512_kld0.001_aug",
        # "Exp14": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-16_ep200_cosine_lr0.0005_bs512_kld0.0001_aug",
        # "Exp15": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-27_ep200_cosine_lr0.0005_bs512_kld0.01_aug",
        # "Exp16": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-27_ep200_cosine_lr0.0005_bs1024_kld0.01_aug",
        # "Exp17": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-27_ep200_cosine_lr0.0005_bs1024_kld0.001_aug",
        # "Exp18": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-27_ep200_cosine_lr0.0005_bs1024_kld0.0001_aug",
        

        # "Exp19": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-25_ep200_cosine_lr0.0001_bs256_kld0.01_aug",
        # "Exp20": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-25_ep200_cosine_lr0.0001_bs256_kld0.001_aug",
        # "Exp21": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-25_ep200_cosine_lr0.0001_bs256_kld0.0001_aug",
        # "Exp22": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-37_ep200_cosine_lr0.0001_bs512_kld0.0001_aug",
        # "Exp23": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-38_ep200_cosine_lr0.0001_bs512_kld0.01_aug",
        # "Exp24": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-38_ep200_cosine_lr0.0001_bs512_kld0.001_aug",
        # "Exp25": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-38_ep200_cosine_lr0.0001_bs1024_kld0.01_aug",
        # "Exp26": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-38_ep200_cosine_lr0.0001_bs1024_kld0.001_aug",
        # "Exp27": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_21-21-38_ep200_cosine_lr0.0001_bs1024_kld0.0001_aug",
    }
    plot_losses(exp_dict)

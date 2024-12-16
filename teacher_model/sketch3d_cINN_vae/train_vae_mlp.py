import os
import datetime
import torch
from model import VAE_MLP
from data import get_dataloader
import shutil
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR


def train(model, optimizer, train_loader, val_loader, num_epochs, root_dir, logging_file):
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    model.train()

    train_losses = {}
    val_losses = {}
    best_val_loss = np.inf

    for epoch in range(num_epochs):

        # Training Loop
        model.train()
        for i, batch in enumerate(train_loader):
            sketch1, sketch2, traj_gt, params_gt, fitted_traj = batch
            sketch1, sketch2, traj_gt, params_gt, fitted_traj = sketch1.cuda(), sketch2.cuda(), traj_gt.cuda(), params_gt.cuda(), fitted_traj.cuda()

            optimizer.zero_grad()
            recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
            loss = model.loss_function(recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params, params_gt, fitted_traj, M_N=0.0001)

            for key in loss:
                if key not in train_losses:
                    train_losses[key] = np.zeros((num_epochs, len(train_loader)))
                train_losses[key][epoch, i] = loss[key].item()
            
            loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters, 10.)
            optimizer.step()

        # Validation Loop
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                sketch1, sketch2, traj_gt, params_gt, fitted_traj = batch
                sketch1, sketch2, traj_gt, params_gt, fitted_traj = sketch1.cuda(), sketch2.cuda(), traj_gt.cuda(), params_gt.cuda(), fitted_traj.cuda()

                recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
                val_loss_values = model.loss_function(recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params, params_gt, fitted_traj, M_N=0.0001)

                for key in val_loss_values:
                    if key not in val_losses:
                        val_losses[key] = np.zeros((num_epochs, len(val_loader)))
                    val_losses[key][epoch, i] = val_loss_values[key].item()

        if val_losses["loss"][epoch].mean() < best_val_loss:
            best_val_loss = val_losses["loss"][epoch].mean()
            torch.save(model.state_dict(), f'{root_dir}/bspline_model_best.pth')

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'{root_dir}/bspline_model_epoch_{epoch}.pth')

        # Calculate and print out the average training and validation loss
        message = [f"{key}: {np.mean(train_losses[key][epoch]):.4f}" for key in train_losses]
        message = ", ".join(message)
        print(f"Epoch {epoch+1}/{num_epochs}, train, {message}")
        with open(logging_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}, train, {message}\n")

        message = [f"{key}: {np.mean(val_losses[key][epoch]):.4f}" for key in val_losses]
        message = ", ".join(message)
        print(f"Epoch {epoch+1}/{num_epochs}, val, {message}\n")
        with open(logging_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}, val, {message}\n\n")

        # Step the scheduler based on validation loss
        scheduler.step()

    # Save the training and validation losses
    np.save(os.path.join(root_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(root_dir, 'val_losses.npy'), val_losses)


def evaluate_model(model, test_loader, root_dir, logging_file):
    model.eval()
    test_loss = {}

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            sketch1, sketch2, traj_gt, params_gt, fitted_traj = batch
            sketch1, sketch2, traj_gt, params_gt, fitted_traj = sketch1.cuda(), sketch2.cuda(), traj_gt.cuda(), params_gt.cuda(), fitted_traj.cuda()

            recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
            loss = model.loss_function(recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params, params_gt, fitted_traj, M_N=0.0001)

            for key in loss:
                if key not in test_loss:
                    test_loss[key] = np.zeros(len(test_loader))
                test_loss[key][i] = loss[key].item()

    message = [f"{key}: {np.mean(test_loss[key]):.4f}" for key in test_loss]
    message = ", ".join(message)
    print(f"Test Loss: {message}")
    with open(logging_file, 'a') as f:
        f.write(f"Test Loss: {message}\n")

    # Save the test losses
    np.save(os.path.join(root_dir, 'test_losses.npy'), test_loss)    


def visualize_evaluation(model, test_loader, name):
    model.eval()
    with torch.no_grad():
        sketches1, sketches2, reconstructed_sketches1, reconstructed_sketches2, trajectories, fitted_trajectories, generated_trajectories = [], [], [], [], [], [], []
        for i, batch in enumerate(test_loader):
            sketch1, sketch2, traj, params_gt, fitted_traj = batch
            sketch1, sketch2, traj, params_gt, fitted_traj = sketch1.cuda(), sketch2.cuda(), traj.cuda(), params_gt.cuda(), fitted_traj.cuda()
            recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
            generated_trajectory = model.generate_trajectory(params)

            sketches1.append(sketch1[0].cpu().numpy().transpose(1, 2, 0))
            sketches2.append(sketch2[0].cpu().numpy().transpose(1, 2, 0))
            reconstructed_sketches1.append(recons1[0].cpu().numpy().transpose(1, 2, 0))
            reconstructed_sketches2.append(recons2[0].cpu().numpy().transpose(1, 2, 0))
            trajectories.append(traj[0].cpu().numpy())
            fitted_trajectories.append(fitted_traj[0].cpu().numpy())
            generated_trajectories.append(generated_trajectory[0].cpu().numpy())

            if i == 5:  # Visualize only the first 5 batches
                break

        # Visualize the first sample in the batch
        visualize_trajectory(
            sketches1,
            sketches2,
            reconstructed_sketches1,
            reconstructed_sketches2,
            trajectories,
            fitted_trajectories,
            generated_trajectories,
            img_name=f"{name}_generated_trajectory"
        )


def visualize_trajectory(sketches1, sketches2, reconstructed_sketches1, reconstructed_sketches2, trajectories, fitted_trajectories, generated_trajectories, img_name):
    num_samples = len(sketches1)
    fig, axes = plt.subplots(figsize=(6 * num_samples, 36))

    # Create a colormap
    cmap = plt.get_cmap('viridis')

    for i in range(num_samples):
        ax1 = fig.add_subplot(6, num_samples, i + 1)
        ax1.imshow(np.flipud(unnormalize_sketch(sketches1[i].squeeze())))
        ax1.set_title(f"Sample {i + 1}: Sketch 1")
        ax1.set_aspect('equal', 'box')

        ax2 = fig.add_subplot(6, num_samples, num_samples + i + 1)
        ax2.imshow(np.flipud(unnormalize_sketch(sketches2[i].squeeze())))
        ax2.set_title(f"Sample {i + 1}: Sketch 2")
        ax2.set_aspect('equal', 'box')

        ax3 = fig.add_subplot(6, num_samples, 2 * num_samples + i + 1)
        ax3.imshow(np.flipud(unnormalize_sketch(reconstructed_sketches1[i].squeeze())))
        ax3.set_title(f"Sample {i + 1}: Reconstructed Sketch 1")
        ax3.set_aspect('equal', 'box')

        ax4 = fig.add_subplot(6, num_samples, 3 * num_samples + i + 1)
        ax4.imshow(np.flipud(unnormalize_sketch(reconstructed_sketches2[i].squeeze())))
        ax4.set_title(f"Sample {i + 1}: Reconstructed Sketch 2")
        ax4.set_aspect('equal', 'box')

        ax5 = fig.add_subplot(6, num_samples, 4 * num_samples + i + 1, projection='3d')
        ax5.scatter(trajectories[i][:, 0], trajectories[i][:, 1], trajectories[i][:, 2], c=np.arange(len(trajectories[i])), cmap=cmap, alpha=0.3)
        ax5.plot(fitted_trajectories[i][:, 0], fitted_trajectories[i][:, 1], fitted_trajectories[i][:, 2], 'r-', linewidth=2)
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_zlabel('Z')
        ax5.set_title(f"Sample {i + 1}: Fitted Trajectory")

        ax6 = fig.add_subplot(6, num_samples, 5 * num_samples + i + 1, projection='3d')
        ax6.plot(generated_trajectories[i][:, 0], generated_trajectories[i][:, 1], generated_trajectories[i][:, 2])
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')
        ax6.set_title(f"Sample {i + 1}: Generated Trajectory")

    plt.tight_layout()
    plt.savefig(f'eval_results/{img_name}.png')


def unnormalize_sketch(sketch):
    data_mean = 0.0037
    data_std = 0.0472
    data_max = 22.0
    return sketch * data_max * data_std + data_mean


if __name__ == "__main__":
    unique_token = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    unique_name = f"vae_mlp_{unique_token}_lr1e-3_equalWeight4Loss_Euclidean"
    root_dir = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/{os.environ.get('USER')}/sketch_3D/{unique_name}"
    os.makedirs(root_dir, exist_ok=True)
    shutil.copy(__file__, root_dir)

    num_control_points = 20
    img_size = 64
    model = VAE_MLP(img_size=img_size,
                    in_channels=3,
                    latent_dim=256,
                    num_control_points=num_control_points,
                    degree=3).cuda()
    pretrained_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    new_params = list(model.mlp.parameters())
    optimizer = torch.optim.Adam([
        {'params': pretrained_params, 'lr': 1e-4},  # Lower learning rate for pretrained parameters
        {'params': new_params, 'lr': 1e-3}  # Higher learning rate for new parameters
    ])

    pretrained_vae_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE/vae_2024-09-19_23-41-33_ep200_onecycle_lr0.001_bs256_kld0.0001_aug/models/vae_model_best.pth"
    model.load_pretrained_vae(pretrained_vae_path)
    # model.freeze_vae()
    
    # DataLoader for training, validation, and test sets
    train_loader, val_loader, test_loader = get_dataloader(batch_size=256, num_samples=None, img_size=img_size, num_control_points=num_control_points)

    # Train the model
    logging_file = f'{root_dir}/bspline_model_{img_size}_logging.txt'
    train(model, optimizer, train_loader, val_loader, num_epochs=200, root_dir=root_dir, logging_file=logging_file)

    # Save the trained model
    torch.save(model.state_dict(), f'{root_dir}/bspline_model_{img_size}.pth')

    # Evaluate on the test set
    evaluate_model(model, test_loader, root_dir, logging_file)

    # Visualize results
    visualize_evaluation(model, test_loader, unique_name)

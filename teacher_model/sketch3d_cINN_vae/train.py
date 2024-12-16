import os
import datetime
import torch
from model import VAE_CINN
from data import get_dataloader
import shutil
import numpy as np

# TODO:
# 1. log all the losses to a npz file, and plot them [done]
# 2. modify the save folder name, let the folder name contain the hyperparameters (or added modifications)
# 3. maybe important hyperparameters: num_control_points, M_N/kld weight(from beta-vae), lr, batch_size, num_epochs
# 4. add a measure metric of the learned parmaters, e.g. chardol loss of the fitted trajectories w.r.t the ground truth trajectories


def train(model, optimizer, dataloader, num_epochs, logging_text, logging_epochLoss_npz, logging_allLoss_npz):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)
    model.train()
 
    epoch_loss = {
        "loss": [],
        "Reconstruction_Loss_1": [],
        "KLD_1": [],
        "Reconstruction_Loss_2": [],
        "KLD_2": [],
        "CINN_Loss": []
    }
    all_loss = {
        "loss": [],
        "Reconstruction_Loss_1": [],
        "KLD_1": [],
        "Reconstruction_Loss_2": [],
        "KLD_2": [],
        "CINN_Loss": []
    }

    for epoch in range(num_epochs):
        total_loss = {
            "loss": 0,
            "Reconstruction_Loss_1": 0,
            "KLD_1": 0,
            "Reconstruction_Loss_2": 0,
            "KLD_2": 0,
            "CINN_Loss": 0
        }
        # initilize a np array for each value in all_loss
        for key in all_loss:
            all_loss[key].append([])
        # initilize a np array to store the loss values
        for batch in dataloader:
            sketch1, sketch2, traj, params, fitted_traj = batch
            sketch1, sketch2, traj, params, fitted_traj = sketch1.cuda(), sketch2.cuda(), traj.cuda(), params.cuda(), fitted_traj.cuda()
            
            optimizer.zero_grad()
            # z, log_jac_det = model(sketch, params)
            recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, z_cinn, log_jac_det = model(sketch1, sketch2, params)
            
            # Corrected loss calculation
            # nll = torch.mean(z**2) / 2 - torch.mean(log_jac_det) / (model.n_dim_total)
            loss = model.loss_function(recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, z_cinn, log_jac_det, M_N=0.00025)
            for key in total_loss:
                total_loss[key] += loss[key].item()
                # append this batch loss to the all_loss
                all_loss[key][-1].append(loss[key].item())
            
            loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters, 10.)
            optimizer.step()
                
        for key in loss:
            epoch_loss[key].append(total_loss[key] / len(dataloader))

        print(f"Epoch {epoch+1}/{num_epochs}, Average loss: {epoch_loss['loss'][-1]:.4f}, Reconstruction Loss 1: {epoch_loss['Reconstruction_Loss_1'][-1]:.4f}, KLD 1: {epoch_loss['KLD_1'][-1]:.4f}, Reconstruction Loss 2: {epoch_loss['Reconstruction_Loss_2'][-1]:.4f}, KLD 2: {epoch_loss['KLD_2'][-1]:.4f}, CINN Loss: {epoch_loss['CINN_Loss'][-1]:.4f}")
        with open(logging_text, 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}, Average loss: {epoch_loss['loss'][-1]:.4f}, Reconstruction Loss 1: {epoch_loss['Reconstruction_Loss_1'][-1]:.4f}, KLD 1: {epoch_loss['KLD_1'][-1]:.4f}, Reconstruction Loss 2: {epoch_loss['Reconstruction_Loss_2'][-1]:.4f}, KLD 2: {epoch_loss['KLD_2'][-1]:.4f}, CINN Loss: {epoch_loss['CINN_Loss'][-1]:.4f}\n")

        scheduler.step()

    # save the epoch loss to a npz file, and save the all loss to a npz file
    np.savez(logging_epochLoss_npz, **epoch_loss)
    np.savez(logging_allLoss_npz, **all_loss)

if __name__ == "__main__":
    unique_token = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    root_dir = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/{os.environ.get('USER')}/sketch_3D/vae_cinn_{unique_token}"
    os.makedirs(root_dir, exist_ok=True)
    # Copy this file to the root_dir, and rename it to the unique_token
    shutil.copy(__file__, root_dir)

    num_control_points = 20
    img_size = 64
    model = VAE_CINN(img_size=img_size,
                    in_channels=3,
                    latent_dim=128,
                    condition_dim=64,
                    num_control_points=num_control_points,
                    degree=3,).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # num_samples = 21000 maximum
    dataloader = get_dataloader(batch_size=256, num_samples=None, img_size=img_size, num_control_points=num_control_points)
    logging_text = f'{root_dir}/bspline_cinn_model_{img_size}.txt'
    logging_epochLoss_npz = f'{root_dir}/bspline_cinn_model_{img_size}_epochLoss.npz'
    logging_allLoss_npz = f'{root_dir}/bspline_cinn_model_{img_size}_allLoss.npz'

    train(model, optimizer, dataloader, num_epochs=100, logging_text=logging_text, logging_epochLoss_npz=logging_epochLoss_npz, logging_allLoss_npz=logging_allLoss_npz)
    
    torch.save(model.state_dict(), f'{root_dir}/bspline_cinn_model_{img_size}.pth')
    
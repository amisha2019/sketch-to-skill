import os
import datetime
import torch
import argparse
import numpy as np
from model import VAE_CINN
from data import get_dataloader
import shutil

def train(model, optimizer, dataloader, num_epochs, logging_file, loss_file, M_N):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)
    model.train()
 
    all_loss = {
        "loss": [],
        "Reconstruction_Loss": [],
        "KLD": [],
        "CINN_Loss": []
    }
    for epoch in range(num_epochs):
        total_loss = {
            "loss": 0,
            "Reconstruction_Loss": 0,
            "KLD": 0,
            "CINN_Loss": 0
        }
        for batch in dataloader:
            sketch, traj, params, fitted_traj = batch
            sketch, traj, params, fitted_traj = sketch.cuda(), traj.cuda(), params.cuda(), fitted_traj.cuda()
            
            optimizer.zero_grad()
            recons, input, mu, log_var, z_cinn, log_jac_det = model(sketch, params)
            
            # Calculate loss
            loss = model.loss_function(recons, input, mu, log_var, z_cinn, log_jac_det, M_N=M_N)
            for key in total_loss:
                total_loss[key] += loss[key].item()
            
            loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters, 10.0)
            optimizer.step()
                
        for key in loss:
            all_loss[key].append(total_loss[key] / len(dataloader))

        # Print loss details
        print(f"Epoch {epoch+1}/{num_epochs}, Average loss: {all_loss['loss'][-1]:.4f}, "
              f"Reconstruction Loss: {all_loss['Reconstruction_Loss'][-1]:.4f}, "
              f"KLD: {all_loss['KLD'][-1]:.4f}, CINN Loss: {all_loss['CINN_Loss'][-1]:.4f}")
        
        # Write loss details to the log file
        with open(logging_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}, Average loss: {all_loss['loss'][-1]:.4f}, "
                    f"Reconstruction Loss: {all_loss['Reconstruction_Loss'][-1]:.4f}, "
                    f"KLD: {all_loss['KLD'][-1]:.4f}, CINN Loss: {all_loss['CINN_Loss'][-1]:.4f}\n")
        
        # Save the loss values to an .npz file after every epoch
        np.savez(loss_file, **all_loss)
        
        scheduler.step()

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="VAE-CINN Training Script")
    parser.add_argument('--num_control_points', type=int, default=10, help='Number of control points for B-spline')
    parser.add_argument('--img_size', type=int, default=64, help='Size of the input image')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension of the VAE')
    parser.add_argument('--condition_dim', type=int, default=64, help='Condition dimension for CINN')
    # main
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_samples', type=int, default=20000, help='Total number of training samples')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--kld_weight', type=float, default=0.00025, help='Weight for KLD loss')
    ## 
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')

    args = parser.parse_args()

    # Add parameters to the unique token to prevent overwriting when runs start simultaneously
    unique_token = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_bs{args.batch_size}_ns{args.num_samples}_lr{args.learning_rate}_kld{args.kld_weight}_epochs{args.num_epochs}"
    root_dir = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/asingh/{unique_token}"
    os.makedirs(root_dir, exist_ok=True)
    shutil.copy(__file__, root_dir)

    # Model initialization
    model = VAE_CINN(img_size=args.img_size,
                     in_channels=1,
                     latent_dim=args.latent_dim,
                     condition_dim=args.condition_dim,
                     num_control_points=args.num_control_points,
                     degree=3).cuda()

    # Optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # DataLoader setup
    dataloader = get_dataloader(batch_size=args.batch_size, num_samples=args.num_samples,
                                img_size=args.img_size, num_control_points=args.num_control_points)

    # Logging and loss file setup
    log_filename = f'log_bs{args.batch_size}_ns{args.num_samples}_lr{args.learning_rate}_kld{args.kld_weight}_epochs{args.num_epochs}.txt'
    logging_file = f'{root_dir}/{log_filename}'
    
    loss_file = f'{root_dir}/losses_bs{args.batch_size}_ns{args.num_samples}_lr{args.learning_rate}_kld{args.kld_weight}_epochs{args.num_epochs}.npz'

    # Train the model
    train(model, optimizer, dataloader, num_epochs=args.num_epochs, logging_file=logging_file, loss_file=loss_file, M_N=args.kld_weight)
    
    # Save the trained model with parameters in the filename
    model_filename = f'{root_dir}/bspline_cinn_model_bs{args.batch_size}_ns{args.num_samples}_lr{args.learning_rate}_kld{args.kld_weight}_epochs{args.num_epochs}.pth'
    torch.save(model.state_dict(), model_filename)



# import os
# import datetime
# import torch
# import argparse
# import numpy as np
# from model import VAE_CINN
# from data import get_dataloader
# import shutil

# def train(model, optimizer, dataloader, num_epochs, logging_file, loss_file, M_N):
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)
#     model.train()
 
#     all_loss = {
#         "loss": [],
#         "Reconstruction_Loss": [],
#         "KLD": [],
#         "CINN_Loss": []
#     }
#     for epoch in range(num_epochs):
#         total_loss = {
#             "loss": 0,
#             "Reconstruction_Loss": 0,
#             "KLD": 0,
#             "CINN_Loss": 0
#         }
#         for batch in dataloader:
#             sketch, traj, params, fitted_traj = batch
#             sketch, traj, params, fitted_traj = sketch.cuda(), traj.cuda(), params.cuda(), fitted_traj.cuda()
            
#             optimizer.zero_grad()
#             recons, input, mu, log_var, z_cinn, log_jac_det = model(sketch, params)
            
#             # Calculate loss
#             loss = model.loss_function(recons, input, mu, log_var, z_cinn, log_jac_det, M_N=M_N)
#             for key in total_loss:
#                 total_loss[key] += loss[key].item()
            
#             loss["loss"].backward()
#             torch.nn.utils.clip_grad_norm_(model.trainable_parameters, 10.0)
#             optimizer.step()
                
#         for key in loss:
#             all_loss[key].append(total_loss[key] / len(dataloader))

#         # Print loss details
#         print(f"Epoch {epoch+1}/{num_epochs}, Average loss: {all_loss['loss'][-1]:.4f}, "
#               f"Reconstruction Loss: {all_loss['Reconstruction_Loss'][-1]:.4f}, "
#               f"KLD: {all_loss['KLD'][-1]:.4f}, CINN Loss: {all_loss['CINN_Loss'][-1]:.4f}")
        
#         # Write loss details to the log file
#         with open(logging_file, 'a') as f:
#             f.write(f"Epoch {epoch+1}/{num_epochs}, Average loss: {all_loss['loss'][-1]:.4f}, "
#                     f"Reconstruction Loss: {all_loss['Reconstruction_Loss'][-1]:.4f}, "
#                     f"KLD: {all_loss['KLD'][-1]:.4f}, CINN Loss: {all_loss['CINN_Loss'][-1]:.4f}\n")
        
#         # Save the loss values to an .npz file after every epoch
#         np.savez(loss_file, **all_loss)
        
#         scheduler.step()

# if __name__ == "__main__":
#     # Argument parsing
#     parser = argparse.ArgumentParser(description="VAE-CINN Training Script")
#     parser.add_argument('--num_control_points', type=int, default=10, help='Number of control points for B-spline')
#     parser.add_argument('--img_size', type=int, default=64, help='Size of the input image')
#     parser.add_argument('--latent_dim', type=int, default=128, help='Latent dimension of the VAE')
#     parser.add_argument('--condition_dim', type=int, default=64, help='Condition dimension for CINN')
#     # main
#     parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
#     parser.add_argument('--num_samples', type=int, default=20000, help='Total number of training samples')
#     parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
#     parser.add_argument('--kld_weight', type=float, default=0.00025, help='Weight for KLD loss')
#     parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
#     parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained weights')

#     args = parser.parse_args()

#     # Directory setup for logging and saving models
#     unique_token = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_bs{args.batch_size}_ns{args.num_samples}_lr{args.learning_rate}_kld{args.kld_weight}_epochs{args.num_epochs}"
#     root_dir = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/asingh/{unique_token}"
#     os.makedirs(root_dir, exist_ok=True)
#     shutil.copy(__file__, root_dir)

#     # Model initialization
#     model = VAE_CINN(img_size=args.img_size,
#                      in_channels=1,
#                      latent_dim=args.latent_dim,
#                      condition_dim=args.condition_dim,
#                      num_control_points=args.num_control_points,
#                      degree=3).cuda()

#     # Load pretrained weights if provided
#     if args.pretrained_weights is not None:
#         print(f"Loading pretrained weights from {args.pretrained_weights}")
#         pretrained_weights = torch.load(args.pretrained_weights)
#         model.load_state_dict(pretrained_weights, strict=False)  # Allow partial loading if architectures differ

#     # Optimizer setup
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
#     # DataLoader setup
#     dataloader = get_dataloader(batch_size=args.batch_size, num_samples=args.num_samples,
#                                 img_size=args.img_size, num_control_points=args.num_control_points)

#     # Logging and loss file setup
#     log_filename = f'log_bs{args.batch_size}_ns{args.num_samples}_lr{args.learning_rate}_kld{args.kld_weight}_epochs{args.num_epochs}.txt'
#     logging_file = f'{root_dir}/{log_filename}'
    
#     loss_file = f'{root_dir}/losses_bs{args.batch_size}_ns{args.num_samples}_lr{args.learning_rate}_kld{args.kld_weight}_epochs{args.num_epochs}.npz'

#     # Train the model
#     train(model, optimizer, dataloader, num_epochs=args.num_epochs, logging_file=logging_file, loss_file=loss_file, M_N=args.kld_weight)
    
#     # Save the trained model with parameters in the filename
#     model_filename = f'{root_dir}/bspline_cinn_model_bs{args.batch_size}_ns{args.num_samples}_lr{args.learning_rate}_kld{args.kld_weight}_epochs{args.num_epochs}.pth'
#     torch.save(model.state_dict(), model_filename)

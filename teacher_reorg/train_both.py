import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import os
import matplotlib.pyplot as plt
from utils.logging import Logger
from data_func.dataloader_helper import rescale


def kl_annealing(epoch, num_epochs, start=0.0, end=1.0, warmup_epochs=10):
    if epoch < warmup_epochs:
        return start + (end - start) * (epoch / warmup_epochs)
    else:
        return end
    
    
def train_vae_mlp(model, optimizer, train_loader, val_loader, sketch_train_loader, args, logger: Logger):
    if args.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, 
                               max_lr=args.lr,
                               steps_per_epoch=len(train_loader),
                               epochs=args.num_epochs,
                               pct_start=0.3)  # 30% of training for the upward phase
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    eval_img_dir = os.path.join(args.root_dir, 'eval_imgs')
    os.makedirs(eval_img_dir, exist_ok=True)

    best_val_loss = np.inf
    for epoch in range(args.num_epochs):
        # Calculate KLD weight for this epoch
        kld_weight = kl_annealing(epoch, args.num_epochs, 1e-10, args.M_N, args.num_epochs) if args.kld_anneal else args.M_N
        logger.log_to_console(f"KLD weight: {kld_weight}")

        # Train the vae model
        mean_vae_loss_kld, mean_vae_loss_recon = [], []
        for i, batch in enumerate(sketch_train_loader):
            sketch = batch.cuda()
            optimizer.zero_grad()
            recons, input, mu, log_var = model.forward_sketch(sketch)
            loss = model.loss_function_sketch(recons, input, mu, log_var, M_N=kld_weight)
            loss["loss"].backward()
            optimizer.step()
            mean_vae_loss_kld.append(loss["KLD"].item())
            mean_vae_loss_recon.append(loss["Reconstruction_Loss"].item())
        logger.log_to_console(f"Mean VAE loss: Reconstruction: {np.mean(mean_vae_loss_recon)}, KLD: {np.mean(mean_vae_loss_kld)}")

        # Start training
        for i, batch in enumerate(train_loader):
            sketch1, sketch2, traj_gt, params_gt, fitted_traj = batch
            sketch1, sketch2, traj_gt, params_gt, fitted_traj = sketch1.cuda(), sketch2.cuda(), traj_gt.cuda(), params_gt.cuda(), fitted_traj.cuda()

            optimizer.zero_grad()
            recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
            loss = model.loss_function(recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params, params_gt, traj_gt, M_N=kld_weight)
            loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters, 10.)
            optimizer.step()

            logger.log_loss(epoch, loss, 'train')

            if args.scheduler == 'onecycle':
                scheduler.step()
                logger.log_lr(epoch, scheduler.get_last_lr()[0])
        
        if args.scheduler == 'cosine':
            scheduler.step()
            logger.log_lr(epoch, scheduler.get_last_lr()[0])

        # Validation Step
        val_loss = evaluate_model(model, val_loader, args, 'val', logger, epoch)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{args.root_dir}/models/vae_mlp_model_best.pth')

        # Save the model at every 10 epochs
        if epoch % 10 == 0:
            # visualize some samples every 10 epochs
            visualize_evaluation(model, val_loader, f'val_{epoch}', eval_img_dir, eval_num=1)
            torch.save(model.state_dict(), f'{args.root_dir}/models/vae_mlp_model_epoch_{epoch}.pth')

        # Calculate and print out the average training and validation loss
        logger.log_epoch_loss_to_file(epoch)

    # Save the training and validation losses
    logger.log_losses_to_npz()

    # Save the trained model
    torch.save(model.state_dict(), f'{args.root_dir}/models/vae_mlp_model_final.pth')


def evaluate_model(model, data_loader, args, eval_mode, logger=None, epoch=None):
    # eval_mode: 'val', 'test', 'hand_draw'
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            sketch1, sketch2, traj_gt, params_gt, fitted_traj = batch
            sketch1, sketch2, traj_gt, params_gt, fitted_traj = sketch1.cuda(), sketch2.cuda(), traj_gt.cuda(), params_gt.cuda(), fitted_traj.cuda()

            recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
            loss = model.loss_function(recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params, params_gt, traj_gt, M_N=args.M_N)
            losses.append(loss["loss"].item())

            if logger is not None:
                epoch = i if epoch is None else epoch
                logger.log_loss(epoch, loss, eval_mode)
        
    if logger is not None and (eval_mode == 'test' or eval_mode == 'hand_draw'):
        logger.log_test_loss_to_file(eval_mode)

    return np.mean(losses)


def visualize_evaluation(model, test_loader, name, root_dir, eval_num=5):
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            sketch1, sketch2, traj, params_gt, fitted_traj = batch
            sketch1, sketch2, traj, params_gt, fitted_traj = sketch1.cuda(), sketch2.cuda(), traj.cuda(), params_gt.cuda(), fitted_traj.cuda()
            recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
            generated_trajectory = model.generate_trajectory(params)

            # Randomly select 10 samples from the batch
            num_samples = min(10, sketch1.size(0))
            indices = torch.randperm(sketch1.size(0))[:num_samples]
            sel_sketches1 = sketch1[indices].cpu().numpy().transpose(0, 2, 3, 1)
            sel_sketches2 = sketch2[indices].cpu().numpy().transpose(0, 2, 3, 1)
            sel_recons1 = recons1[indices].cpu().numpy().transpose(0, 2, 3, 1)
            sel_recons2 = recons2[indices].cpu().numpy().transpose(0, 2, 3, 1)
            sel_trajs = traj[indices].cpu().numpy()
            sel_fitted_trajs = fitted_traj[indices].cpu().numpy()
            sel_generated_trajs = generated_trajectory[indices].cpu().numpy()

            visualize_trajectory(sel_sketches1, sel_sketches2, sel_recons1, sel_recons2, sel_trajs, sel_fitted_trajs, sel_generated_trajs, f"generated_trajectory_{name}_{i}", root_dir)

            # Interpolate between pairs of samples
            sel_mu1 = mu1[indices]
            sel_mu2 = mu2[indices]
            num_samples = min(len(sel_mu1) // 2, 2)
            for j in range(num_samples):
                interpolated_mu_1 = []
                interpolated_mu_2 = []
                for alpha in np.linspace(0, 1, 10):
                    interpolated_mu_1.append((1 - alpha) * sel_mu1[j] + alpha * sel_mu1[-j])
                    interpolated_mu_2.append((1 - alpha) * sel_mu2[j] + alpha * sel_mu2[-j])
                interpolated_mu_1 = torch.stack(interpolated_mu_1, dim=0).cuda()
                interpolated_mu_2 = torch.stack(interpolated_mu_2, dim=0).cuda()
                decoded_sketch1 = model.decoder(interpolated_mu_1).permute(0, 2, 3, 1).cpu().numpy()
                decoded_sketch2 = model.decoder(interpolated_mu_2).permute(0, 2, 3, 1).cpu().numpy()
                decoded_params = model.mlp(torch.cat((interpolated_mu_1, interpolated_mu_2), dim=1))
                generated_trajs = model.generate_trajectory(decoded_params).cpu().numpy()

                traj_gt_start = sel_trajs[j]
                traj_gt_end = sel_trajs[-j]
                sketch_gt_1 = [sel_sketches1[j], sel_sketches1[-j]]
                sketch_gt_2 = [sel_sketches2[j], sel_sketches2[-j]]
                
                visualize_and_save_interpolated_trajectories(traj_gt_start, traj_gt_end, generated_trajs, sketch_gt_1, sketch_gt_2, decoded_sketch1, decoded_sketch2, f"interpolated_trajectory_{name}_{i}_{j}", root_dir)
            
            if i == eval_num - 1:
                break


def visualize_inference(model, data_loader, name, root_dir, eval_num=5):
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            sketch1, sketch2 = batch
            sketch1, sketch2 = sketch1.cuda(), sketch2.cuda()
            recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
            generated_trajectory = model.generate_trajectory(params)

            # Randomly select 10 samples from the batch
            num_samples = min(10, sketch1.size(0))
            indices = torch.randperm(sketch1.size(0))[:num_samples]
            sel_sketches1 = sketch1[indices].cpu().numpy().transpose(0, 2, 3, 1)
            sel_sketches2 = sketch2[indices].cpu().numpy().transpose(0, 2, 3, 1)
            sel_recons1 = recons1[indices].cpu().numpy().transpose(0, 2, 3, 1)
            sel_recons2 = recons2[indices].cpu().numpy().transpose(0, 2, 3, 1)
            sel_generated_trajs = generated_trajectory[indices].cpu().numpy()

            visualize_trajectory(sel_sketches1, sel_sketches2, sel_recons1, sel_recons2, None, None, sel_generated_trajs, f"generated_trajectory_{name}_{i}", root_dir)
            
            # Interpolate between pairs of samples
            sel_mu1 = mu1[indices]
            sel_mu2 = mu2[indices]
            num_samples = min(len(sel_mu1) // 2, 2)
            for j in range(num_samples):
                interpolated_mu_1 = []
                interpolated_mu_2 = []
                for alpha in np.linspace(0, 1, 10):
                    interpolated_mu_1.append((1 - alpha) * sel_mu1[j] + alpha * sel_mu1[-j])
                    interpolated_mu_2.append((1 - alpha) * sel_mu2[j] + alpha * sel_mu2[-j])
                interpolated_mu_1 = torch.stack(interpolated_mu_1, dim=0).cuda()
                interpolated_mu_2 = torch.stack(interpolated_mu_2, dim=0).cuda()
                decoded_sketch1 = model.decoder(interpolated_mu_1).permute(0, 2, 3, 1).cpu().numpy()
                decoded_sketch2 = model.decoder(interpolated_mu_2).permute(0, 2, 3, 1).cpu().numpy()
                decoded_params = model.mlp(torch.cat((interpolated_mu_1, interpolated_mu_2), dim=1))
                generated_trajs = model.generate_trajectory(decoded_params).cpu().numpy()

                sketch_gt_1 = [sel_sketches1[j], sel_sketches1[-j]]
                sketch_gt_2 = [sel_sketches2[j], sel_sketches2[-j]]
                
                visualize_and_save_interpolated_trajectories(None, None, generated_trajs, sketch_gt_1, sketch_gt_2, decoded_sketch1, decoded_sketch2, f"interpolated_trajectory_{name}_{i}_{j}", root_dir)
            
            if i == eval_num - 1:
                break


def visualize_trajectory(sketches1, sketches2, recons1, recons2, trajectories, fitted_trajectories, generated_trajectories, img_name, output_dir):
    if trajectories is None and fitted_trajectories is None:
        rows = 5
    else:
        rows = 6
    num_samples = len(sketches1)
    fig = plt.figure(figsize=(4 * num_samples, 4 * rows))

    # Create a colormap
    cmap = plt.get_cmap('viridis')

    for i in range(num_samples):
        ax1 = fig.add_subplot(rows, num_samples, i + 1)
        ax1.imshow(rescale(sketches1[i].squeeze()))
        ax1.set_title(f"Sample {i + 1}: Sketch 1")
        ax1.set_aspect('equal', 'box')

        ax2 = fig.add_subplot(rows, num_samples, num_samples + i + 1)
        ax2.imshow(rescale(recons1[i].squeeze()))
        ax2.set_title(f"Recon1 MSE: {np.mean((sketches1[i] - recons1[i])**2):.6f}")
        ax2.set_aspect('equal', 'box')

        ax3 = fig.add_subplot(rows, num_samples, 2 * num_samples + i + 1)
        ax3.imshow(rescale(sketches2[i].squeeze()))
        ax3.set_title(f"Sample {i + 1}: Sketch 2")
        ax3.set_aspect('equal', 'box')

        ax4 = fig.add_subplot(rows, num_samples, 3 * num_samples + i + 1)
        ax4.imshow(rescale(recons2[i].squeeze()))
        ax4.set_title(f"Recon2 MSE: {np.mean((sketches2[i] - recons2[i])**2):.6f}")
        ax4.set_aspect('equal', 'box')

        if trajectories is None and fitted_trajectories is None:
            ax5 = fig.add_subplot(rows, num_samples, 4 * num_samples + i + 1, projection='3d')
            ax5.plot(generated_trajectories[i][:, 0], generated_trajectories[i][:, 1], generated_trajectories[i][:, 2])
            ax5.set_xlabel('X')
            ax5.set_ylabel('Y')
            ax5.set_zlabel('Z')
            ax5.set_title(f"Sample {i + 1}: Generated Trajectory")

        else:
            ax5 = fig.add_subplot(rows, num_samples, 4 * num_samples + i + 1, projection='3d')
            ax5.plot(trajectories[i][:, 0], trajectories[i][:, 1], trajectories[i][:, 2], 'b-', alpha=0.5, linewidth=1)
            ax5.scatter(fitted_trajectories[i][:, 0], fitted_trajectories[i][:, 1], fitted_trajectories[i][:, 2], c=np.arange(len(fitted_trajectories[i])), cmap=cmap, s=20)
            ax5.set_xlabel('X')
            ax5.set_ylabel('Y')
            ax5.set_zlabel('Z')
            ax5.set_title(f"Sample {i + 1}: Fitted Trajectory")

            ax6 = fig.add_subplot(rows, num_samples, 5 * num_samples + i + 1, projection='3d')
            ax6.plot(generated_trajectories[i][:, 0], generated_trajectories[i][:, 1], generated_trajectories[i][:, 2])
            ax6.set_xlabel('X')
            ax6.set_ylabel('Y')
            ax6.set_zlabel('Z')
            ax6.set_title(f"Generated Traj MSE: {np.mean((fitted_trajectories[i] - generated_trajectories[i])**2):.6f}")

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{img_name}.png')
    plt.close()


def visualize_and_save_interpolated_trajectories(original_traj1, original_traj2, generated_trajs, sketch_gt_1, sketch_gt_2, decoded_sketch1, decoded_sketch2, img_name, output_dir):
    """
    Save the interpolated 3D trajectories
    """
    rows = 3
    num_interpolations = generated_trajs.shape[0]
    cols = num_interpolations + 2
    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    cmap = plt.get_cmap('viridis')

    # Plot original sketches
    ax = fig.add_subplot(rows, cols, 1)
    ax.imshow(rescale(sketch_gt_1[0]))
    ax.set_title("Original Sketch 1")
    ax.axis('off')

    ax = fig.add_subplot(rows, cols, cols + 1)
    ax.imshow(rescale(sketch_gt_2[0]))
    ax.set_title("Original Sketch 2")
    ax.axis('off')

    # Plot original trajectories
    if original_traj1 is not None:
        ax = fig.add_subplot(rows, cols, 2*cols + 1, projection='3d')
        ax.scatter(original_traj1[:, 0], original_traj1[:, 1], original_traj1[:, 2], c=np.arange(len(original_traj1)), cmap=cmap, s=20)
        ax.plot(original_traj1[:, 0], original_traj1[:, 1], original_traj1[:, 2], label="Original Trajectory 1")
        ax.set_title("Original Trajectories 1")
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Plot interpolated trajectories
    for i in range(num_interpolations):
        # Plot decoded sketches
        ax = fig.add_subplot(rows, cols, i+2)
        ax.imshow(rescale(decoded_sketch1[i]))
        ax.set_title(f"Decoded Sketch 1 - {i+1}")
        ax.axis('off')

        ax = fig.add_subplot(rows, cols, cols+i+2)
        ax.imshow(rescale(decoded_sketch2[i]))
        ax.set_title(f"Decoded Sketch 2 - {i+1}")
        ax.axis('off')

        ax = fig.add_subplot(rows, cols, 2*cols+i+2, projection='3d')
        traj = generated_trajs[i]
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=np.arange(len(traj)), cmap=cmap, s=20)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"Interp {i+1}")
        ax.set_title(f"Interpolated Trajectory {i+1}")
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Show sketch_gt_1 at the end of the first row
    ax = fig.add_subplot(rows, cols, cols)
    ax.imshow(rescale(sketch_gt_1[1]))
    ax.set_title("Original Sketch 1")
    ax.axis('off')

    # Show sketch_gt_2 at the end of the second row
    ax = fig.add_subplot(rows, cols, 2*cols)
    ax.imshow(rescale(sketch_gt_2[1]))
    ax.set_title("Original Sketch 2")
    ax.axis('off')

    # Plot original trajectories
    if original_traj2 is not None:
        ax = fig.add_subplot(rows, cols, 3*cols, projection='3d')
        ax.scatter(original_traj2[:, 0], original_traj2[:, 1], original_traj2[:, 2], c=np.arange(len(original_traj2)), cmap=cmap, s=20)
        ax.plot(original_traj2[:, 0], original_traj2[:, 1], original_traj2[:, 2], label="Original Trajectory 2")
        ax.set_title("Original Trajectories 2")
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{img_name}.png", bbox_inches='tight')  # Increase DPI and use tight bounding box
    plt.close()
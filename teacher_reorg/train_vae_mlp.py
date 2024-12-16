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
    
    
def train_vae_mlp(model, optimizer, train_loader, val_loader, test_loaders, sketch_train_loader, args, logger: Logger):
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
        if not args.disable_vae and args.train_both:
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
            if args.num_sketches == 2:
                recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
                loss = model.loss_function(recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params, params_gt, traj_gt, M_N=kld_weight)
            else:
                recons, sketch, mu, log_var, params = model(sketch1)
                loss = model.loss_function(recons, sketch, mu, log_var, params, params_gt, traj_gt, M_N=kld_weight)
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
            visualize_evaluation(model, val_loader, f'val_ep{epoch}', eval_img_dir, eval_num=1)
            for key in test_loaders.keys():
                visualize_evaluation(model, test_loaders[key], f'test_{key}_ep{epoch}', eval_img_dir, eval_num=1, interpolate=False)
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

            if args.num_sketches == 2:
                recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
                loss = model.loss_function(recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params, params_gt, traj_gt, M_N=args.M_N)
            else:
                recons, sketch, mu, log_var, params = model(sketch1)
                loss = model.loss_function(recons, sketch, mu, log_var, params, params_gt, traj_gt, M_N=args.M_N)
            losses.append(loss["loss"].item())

            if logger is not None:
                epoch = i if epoch is None else epoch
                logger.log_loss(epoch, loss, eval_mode)
        
    if logger is not None and (eval_mode == 'test' or eval_mode == 'hand_draw'):
        logger.log_test_loss_to_file(eval_mode)

    return np.mean(losses)


def visualize_evaluation(model, test_loader, name, root_dir, eval_num=5, gen_num=6, interpolate=True):
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            sketch1, sketch2, traj, params_gt, fitted_traj = batch
            # sss = sketch1.permute(1, 2, 0)
            # sss = (sss - sss.min())/(sss.max()-sss.min()) * 255
            # from PIL import Image
            # Image.fromarray(sss.cpu().numpy().astype(np.uint8)).save("sketch1.png")
            # breakpoint()
            sketch1, sketch2, traj, params_gt, fitted_traj = sketch1.cuda(), sketch2.cuda(), traj.cuda(), params_gt.cuda(), fitted_traj.cuda()
            generated_trajs = []
            for _ in range(gen_num):
                if model.num_sketches == 2:
                    recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
                else:
                    recons1, sketch1, mu1, log_var1, params = model(sketch1)
                generated_trajectory = model.generate_trajectory(params)
                if model.use_traj_rescale:
                    generated_trajectory = model.rescale_traj(generated_trajectory, traj[:, 0, :], traj[:, -1, :])
                generated_trajs.append(generated_trajectory)
            generated_trajs = torch.stack(generated_trajs, dim=0)

            # Randomly select 10 samples from the batch
            num_samples = min(10, sketch1.size(0))
            indices = torch.randperm(sketch1.size(0))[:num_samples]
            if model.num_sketches == 2:
                sel_sketches1 = sketch1[indices].cpu().numpy().transpose(0, 2, 3, 1)
                sel_sketches2 = sketch2[indices].cpu().numpy().transpose(0, 2, 3, 1)
                sel_recons1 = recons1[indices].cpu().numpy().transpose(0, 2, 3, 1)
                sel_recons2 = recons2[indices].cpu().numpy().transpose(0, 2, 3, 1)
            else:
                sel_sketches1 = sketch1[indices].cpu().numpy().transpose(0, 2, 3, 1)
                sel_recons1 = recons1[indices].cpu().numpy().transpose(0, 2, 3, 1)
                sel_sketches2 = None
                sel_recons2 = None
            sel_trajs = traj[indices].cpu().numpy()
            sel_fitted_trajs = fitted_traj[indices].cpu().numpy()
            sel_generated_trajs = generated_trajs[:, indices].cpu().numpy()
            visualize_trajectory(sel_sketches1, sel_sketches2, sel_recons1, sel_recons2, sel_trajs, sel_fitted_trajs, sel_generated_trajs, f"generated_trajectory_{name}_{i}", root_dir)
            
            if interpolate:
                # Interpolate between pairs of samples
                sel_mu1 = mu1[indices]
                sel_mu2 = mu2[indices] if model.num_sketches == 2 else None
                num_samples = min(len(sel_mu1) // 2, 2)
                for j in range(num_samples):
                    traj_gt_start = sel_trajs[j]
                    traj_gt_end = sel_trajs[-j-1]
                    sketch_gt_1 = [sel_sketches1[j], sel_sketches1[-j-1]]
                    sketch_gt_2 = [sel_sketches2[j], sel_sketches2[-j-1]] if model.num_sketches == 2 else None

                    interpolated_mu_1 = []
                    interpolated_mu_2 = []
                    for alpha in np.linspace(0, 1, 10):
                        interpolated_mu_1.append((1 - alpha) * sel_mu1[j] + alpha * sel_mu1[-j-1])
                        if model.num_sketches == 2:
                            interpolated_mu_2.append((1 - alpha) * sel_mu2[j] + alpha * sel_mu2[-j-1])

                    interpolated_mu_1 = torch.stack(interpolated_mu_1, dim=0).cuda()
                    interpolated_mu_2 = torch.stack(interpolated_mu_2, dim=0).cuda() if model.num_sketches == 2 else None
                    decoded_sketch1 = model.decoder(interpolated_mu_1).permute(0, 2, 3, 1).cpu().numpy()
                    decoded_sketch2 = model.decoder(interpolated_mu_2).permute(0, 2, 3, 1).cpu().numpy() if model.num_sketches == 2 else None
                    decoded_params = model.mlp(torch.cat((interpolated_mu_1, interpolated_mu_2), dim=1)) if model.num_sketches == 2 else model.mlp(interpolated_mu_1)
                    generated_trajs = model.generate_trajectory(decoded_params)
                    if model.use_traj_rescale:
                        start = []
                        end = []
                        for alpha in np.linspace(0, 1, 10):
                            start.append((1 - alpha) * traj_gt_start[0, :] + alpha * traj_gt_end[0, :])
                            end.append((1 - alpha) * traj_gt_end[-1, :] + alpha * traj_gt_start[-1, :])
                        start = torch.tensor(np.array(start), dtype=torch.float32).cuda()
                        end = torch.tensor(np.array(end), dtype=torch.float32).cuda()
                        generated_trajs = model.rescale_traj(generated_trajs, start, end)
                    generated_trajs = generated_trajs.cpu().numpy()
                    
                    visualize_and_save_interpolated_trajectories(traj_gt_start, traj_gt_end, generated_trajs, sketch_gt_1, sketch_gt_2, decoded_sketch1, decoded_sketch2, f"interpolated_trajectory_{name}_{i}_{j}", root_dir)
                
            if i == eval_num - 1:
                break


def visualize_inference(model, data_loader, name, root_dir, eval_num=5, gen_num=6):
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            sketch1, sketch2, starts, ends = batch
            sketch1, sketch2 = sketch1.cuda(), sketch2.cuda()
            generated_trajs = []
            for _ in range(gen_num):
                if model.num_sketches == 2:
                    recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
                else:
                    recons1, sketch1, mu1, log_var1, params = model(sketch1)
                generated_trajectory = model.generate_trajectory(params)
                if model.use_traj_rescale and starts is not None and ends is not None:
                    starts, ends = starts.cuda(), ends.cuda()
                    generated_trajectory = model.rescale_traj(generated_trajectory, starts, ends)
                generated_trajs.append(generated_trajectory)
            generated_trajs = torch.stack(generated_trajs, dim=0)

            # Randomly select 10 samples from the batch
            num_samples = min(10, sketch1.size(0))
            indices = torch.randperm(sketch1.size(0))[:num_samples]
            sel_sketches1 = sketch1[indices].cpu().numpy().transpose(0, 2, 3, 1)
            sel_sketches2 = sketch2[indices].cpu().numpy().transpose(0, 2, 3, 1) if model.num_sketches == 2 else None
            sel_recons1 = recons1[indices].cpu().numpy().transpose(0, 2, 3, 1)
            sel_recons2 = recons2[indices].cpu().numpy().transpose(0, 2, 3, 1) if model.num_sketches == 2 else None
            sel_generated_trajs = generated_trajs[:, indices].cpu().numpy()

            visualize_trajectory(sel_sketches1, sel_sketches2, sel_recons1, sel_recons2, None, None, sel_generated_trajs, f"generated_trajectory_{name}_{i}", root_dir)
            
            # Interpolate between pairs of samples
            sel_mu1 = mu1[indices]
            sel_mu2 = mu2[indices] if model.num_sketches == 2 else None
            num_samples = min(len(sel_mu1) // 2, 2)
            for j in range(num_samples):
                interpolated_mu_1 = []
                interpolated_mu_2 = []
                for alpha in np.linspace(0, 1, 10):
                    interpolated_mu_1.append((1 - alpha) * sel_mu1[j] + alpha * sel_mu1[-j-1])
                    if model.num_sketches == 2:
                        interpolated_mu_2.append((1 - alpha) * sel_mu2[j] + alpha * sel_mu2[-j-1])
                interpolated_mu_1 = torch.stack(interpolated_mu_1, dim=0).cuda()
                interpolated_mu_2 = torch.stack(interpolated_mu_2, dim=0).cuda() if model.num_sketches == 2 else None
                decoded_sketch1 = model.decoder(interpolated_mu_1).permute(0, 2, 3, 1).cpu().numpy()
                decoded_sketch2 = model.decoder(interpolated_mu_2).permute(0, 2, 3, 1).cpu().numpy() if model.num_sketches == 2 else None
                decoded_params = model.mlp(torch.cat((interpolated_mu_1, interpolated_mu_2), dim=1)) if model.num_sketches == 2 else model.mlp(interpolated_mu_1)
                generated_trajs = model.generate_trajectory(decoded_params)
                if model.use_traj_rescale and starts is not None and ends is not None:
                    start_gt_1 = starts[indices[j]]
                    start_gt_2 = starts[indices[-j-1]]
                    end_gt_1 = ends[indices[j]]
                    end_gt_2 = ends[indices[-j-1]]
                    start = []
                    end = []
                    for alpha in np.linspace(0, 1, 10):
                        start.append((1 - alpha) * start_gt_1 + alpha * start_gt_2)
                        end.append((1 - alpha) * end_gt_1 + alpha * end_gt_2)
                    start = torch.stack(start, dim=0).cuda()
                    end = torch.stack(end, dim=0).cuda()
                    generated_trajs = model.rescale_traj(generated_trajs, start, end)
                generated_trajs = generated_trajs.cpu().numpy()

                sketch_gt_1 = [sel_sketches1[j], sel_sketches1[-j-1]]
                sketch_gt_2 = [sel_sketches2[j], sel_sketches2[-j-1]] if model.num_sketches == 2 else None
                
                visualize_and_save_interpolated_trajectories(None, None, generated_trajs, sketch_gt_1, sketch_gt_2, decoded_sketch1, decoded_sketch2, f"interpolated_trajectory_{name}_{i}_{j}", root_dir)
            
            if i == eval_num - 1:
                break


def visualize_trajectory(sketches1, sketches2, recons1, recons2, trajectories, fitted_trajectories, generated_trajectories, img_name, output_dir):
    if trajectories is None and fitted_trajectories is None:
        rows = 5
    else:
        rows = 6
    if sketches2 is None and recons2 is None:
        rows -= 2
    num_samples = len(sketches1)
    fig = plt.figure(figsize=(4 * num_samples, 4 * rows))

    # Create a colormap
    cmap = plt.get_cmap('viridis')

    x_lim, y_lim, z_lim = get_task_axis_limits(img_name)

    for i in range(num_samples):
        row_id = 0
        ax = fig.add_subplot(rows, num_samples, row_id * num_samples + i + 1)
        ax.imshow(rescale(sketches1[i].squeeze()))
        # sss = sketches1[i].squeeze()
        # sss = (sss - sss.min())/(sss.max()-sss.min()) * 255
        # from PIL import Image
        # Image.fromarray(sss.astype(np.uint8)).save("sketch1.png")
        
        # breakpoint()
        ax.set_title(f"Sample {i + 1}: Sketch 1")
        ax.set_aspect('equal', 'box')
        row_id += 1

        ax = fig.add_subplot(rows, num_samples, row_id * num_samples + i + 1)
        ax.imshow(rescale(recons1[i].squeeze()))
        ax.set_title(f"Recon1 MSE: {np.mean((sketches1[i] - recons1[i])**2):.6f}")
        ax.set_aspect('equal', 'box')
        row_id += 1

        if sketches2 is not None and recons2 is not None:
            ax = fig.add_subplot(rows, num_samples, row_id * num_samples + i + 1)
            ax.imshow(rescale(sketches2[i].squeeze()))
            ax.set_title(f"Sample {i + 1}: Sketch 2")
            ax.set_aspect('equal', 'box')
            row_id += 1
            
            ax = fig.add_subplot(rows, num_samples, row_id * num_samples + i + 1)
            ax.imshow(rescale(recons2[i].squeeze()))
            ax.set_title(f"Recon2 MSE: {np.mean((sketches2[i] - recons2[i])**2):.6f}")
            ax.set_aspect('equal', 'box')
            row_id += 1

        if trajectories is None and fitted_trajectories is None:
            ax = fig.add_subplot(rows, num_samples, row_id * num_samples + i + 1, projection='3d')
            for gen_traj in generated_trajectories:
                ax.plot(gen_traj[i][:, 0], gen_traj[i][:, 1], gen_traj[i][:, 2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"Sample {i + 1}: Generated Trajectory")
            ax.set_xlim(x_lim[0], x_lim[1])
            ax.set_ylim(y_lim[0], y_lim[1])
            ax.set_zlim(z_lim[0], z_lim[1])
            ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
            # ax.view_init(elev=20, azim=45)  # Adjust the view angle

        else:
            ax = fig.add_subplot(rows, num_samples, row_id * num_samples + i + 1, projection='3d')
            ax.plot(fitted_trajectories[i][:, 0], fitted_trajectories[i][:, 1], fitted_trajectories[i][:, 2], 'b-', alpha=0.5, linewidth=1)
            ax.scatter(trajectories[i][:, 0], trajectories[i][:, 1], trajectories[i][:, 2], c=np.arange(len(trajectories[i])), cmap=cmap, s=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"Sample {i + 1}: Fitted Trajectory")
            ax.set_xlim(x_lim[0], x_lim[1])
            ax.set_ylim(y_lim[0], y_lim[1])
            ax.set_zlim(z_lim[0], z_lim[1])
            ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
            # ax.view_init(elev=20, azim=45)  # Adjust the view angle
            row_id += 1

            ax = fig.add_subplot(rows, num_samples, row_id * num_samples + i + 1, projection='3d')
            ax.scatter(trajectories[i][:, 0], trajectories[i][:, 1], trajectories[i][:, 2], c=np.arange(len(trajectories[i])), cmap=cmap, s=10)
            for gen_traj in generated_trajectories:
                ax.plot(gen_traj[i][:, 0], gen_traj[i][:, 1], gen_traj[i][:, 2])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            mse_loss = np.mean([np.mean((fitted_trajectories[i] - gen_traj[i])**2) for gen_traj in generated_trajectories])
            ax.set_title(f"Generated Traj MSE: {mse_loss:.6f}")
            ax.set_xlim(x_lim[0], x_lim[1])
            ax.set_ylim(y_lim[0], y_lim[1])
            ax.set_zlim(z_lim[0], z_lim[1])
            ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
            # ax6.view_init(elev=20, azim=45)  # Adjust the view angle

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

    x_lim, y_lim, z_lim = get_task_axis_limits(img_name)

    cmap = plt.get_cmap('viridis')

    # Plot original sketch 1
    row_id = 0
    ax = fig.add_subplot(rows, cols, row_id * cols + 1)
    ax.imshow(rescale(sketch_gt_1[0]))
    ax.set_title("Original Sketch 1")
    ax.axis('off')
    # Plot decoded sketches 1
    for i in range(num_interpolations):
        ax = fig.add_subplot(rows, cols, row_id * cols + i + 2)
        ax.imshow(rescale(decoded_sketch1[i]))
        ax.set_title(f"Decoded Sketch 1 - {i+1}")
        ax.axis('off')
    # Show sketch_gt_1 at the end of the first row
    ax = fig.add_subplot(rows, cols, row_id * cols + cols)
    ax.imshow(rescale(sketch_gt_1[1]))
    ax.set_title("Original Sketch 1")
    ax.axis('off')
    row_id += 1

    # Plot original sketch 2
    if sketch_gt_2 is not None:
        ax = fig.add_subplot(rows, cols, row_id * cols + 1)
        ax.imshow(rescale(sketch_gt_2[0]))
        ax.set_title("Original Sketch 2")
        ax.axis('off')
        # Plot decoded sketches 2
        for i in range(num_interpolations):
            ax = fig.add_subplot(rows, cols, row_id * cols + i + 2)
            ax.imshow(rescale(decoded_sketch2[i]))
            ax.set_title(f"Decoded Sketch 2 - {i+1}")
            ax.axis('off')
        # Show sketch_gt_2 at the end of the second row
        ax = fig.add_subplot(rows, cols, row_id * cols + cols)
        ax.imshow(rescale(sketch_gt_2[1]))
        ax.set_title("Original Sketch 2")
        ax.axis('off')
        row_id += 1

    # Plot original trajectories
    if original_traj1 is not None:
        ax = fig.add_subplot(rows, cols, row_id * cols + 1, projection='3d')
        ax.scatter(original_traj1[:, 0], original_traj1[:, 1], original_traj1[:, 2], c=np.arange(len(original_traj1)), cmap=cmap, s=20)
        ax.plot(original_traj1[:, 0], original_traj1[:, 1], original_traj1[:, 2], label="Original Trajectory 1")
        ax.set_title("Original Trajectories 1")
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_zlim(z_lim[0], z_lim[1])
        ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
        ax.view_init(elev=20, azim=45)  # Adjust the view angle

    # Plot interpolated trajectories
    for i in range(num_interpolations):
        ax = fig.add_subplot(rows, cols, row_id * cols + i + 2, projection='3d')
        traj = generated_trajs[i]
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=np.arange(len(traj)), cmap=cmap, s=20)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"Interp {i+1}")
        ax.set_title(f"Interpolated Trajectory {i+1}")
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_zlim(z_lim[0], z_lim[1])
        ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
        ax.view_init(elev=20, azim=45)  # Adjust the view angle

    # Plot original trajectories
    if original_traj2 is not None:
        ax = fig.add_subplot(rows, cols, row_id * cols + cols, projection='3d')
        ax.scatter(original_traj2[:, 0], original_traj2[:, 1], original_traj2[:, 2], c=np.arange(len(original_traj2)), cmap=cmap, s=20)
        ax.plot(original_traj2[:, 0], original_traj2[:, 1], original_traj2[:, 2], label="Original Trajectory 2")
        ax.set_title("Original Trajectories 2")
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_zlim(z_lim[0], z_lim[1])
        ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()), np.ptp(ax.get_zlim())))
        ax.view_init(elev=20, azim=45)  # Adjust the view angle

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{img_name}.png", bbox_inches='tight')  # Increase DPI and use tight bounding box
    plt.close()


def get_task_axis_limits(task):
    if "ButtonPressTopdownWall" in task:
        return [-0.11, 0.12], [0.39, 0.86], [0.19, 0.43]
    elif "ButtonPressReal" in task:
        return [-0.19, -0.04], [-0.37, -0.19], [0.05, 0.14]
    elif "ButtonPress" in task:
        return [-0.12, 0.13], [0.39, 0.76], [0.04, 0.20]
    elif "DrawerOpen" in task:
        return [-0.10, 0.11], [0.55, 0.73], [0.05, 0.32]
    elif "ReachWall" in task:
        return [-0.05, 0.05], [0.59, 0.88], [0.15, 0.32]
    elif "Reach" in task:
        return [-0.10, 0.09], [0.59, 0.87], [0.08, 0.30]
    else:
        return [-0.51, 0.51], [0.37, 0.93], [0.04, 0.43]

# All train tasks
# Trajectory x limits: -0.51, 0.51
# Trajectory y limits: 0.37, 0.93
# Trajectory z limits: 0.04, 0.43

# All test tasks
# Trajectory x limits: -0.12, 0.13
# Trajectory y limits: 0.39, 0.88
# Trajectory z limits: 0.04, 0.43

# Loading ButtonPress
# Trajectory x limits: -0.12, 0.13
# Trajectory y limits: 0.39, 0.76
# Trajectory z limits: 0.04, 0.20

# Loading ButtonPressTopdownWall
# Trajectory x limits: -0.11, 0.12
# Trajectory y limits: 0.39, 0.86
# Trajectory z limits: 0.19, 0.43

# Loading DrawerOpen
# Trajectory x limits: -0.10, 0.11
# Trajectory y limits: 0.55, 0.73
# Trajectory z limits: 0.05, 0.32

# Loading Reach
# Trajectory x limits: -0.10, 0.09
# Trajectory y limits: 0.59, 0.87
# Trajectory z limits: 0.08, 0.30

# Loading ReachWall
# Trajectory x limits: -0.05, 0.05
# Trajectory y limits: 0.59, 0.88
# Trajectory z limits: 0.15, 0.32
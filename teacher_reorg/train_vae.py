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


def train_vae(model, optimizer, train_loader, val_loader, args, logger: Logger):
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

        # Start training
        for i, batch in enumerate(train_loader):
            sketch = batch.cuda()
            optimizer.zero_grad()
            recons, input, mu, log_var = model(sketch)
            loss = model.loss_function(recons, input, mu, log_var, M_N=kld_weight)
            loss["loss"].backward()
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
            torch.save(model.state_dict(), f'{args.root_dir}/models/vae_model_best.pth')

        # Save the model at every 10 epochs
        if epoch % 10 == 0:
            # visualize some samples every 10 epochs
            visualize_evaluation(model, val_loader, f'val_{epoch}', eval_img_dir, eval_num=1)
            torch.save(model.state_dict(), f'{args.root_dir}/models/vae_model_epoch_{epoch}.pth')

        # Calculate and print out the average training and validation loss
        logger.log_epoch_loss_to_file(epoch)

    # Save the training and validation losses
    logger.log_losses_to_npz()

    # Save the trained model
    torch.save(model.state_dict(), f'{args.root_dir}/models/vae_model_final.pth')


def evaluate_model(model, data_loader, args, eval_mode, logger=None, epoch=None):
    # eval_mode: 'val', 'test', 'hand_draw'
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            sketch = batch.cuda()
            recons, input, mu, log_var = model(sketch)
            loss = model.loss_function(recons, input, mu, log_var, M_N=args.M_N)
            losses.append(loss["loss"].item())

            if logger is not None:
                epoch = i if epoch is None else epoch
                logger.log_loss(epoch, loss, eval_mode)
        
    if logger is not None and (eval_mode == 'test' or eval_mode == 'hand_draw') :
        logger.log_test_loss_to_file(eval_mode)

    return np.mean(losses)


def visualize_evaluation(model, test_loader, name, root_dir, eval_num=5):
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            sketch = batch.cuda()
            recons, input, mu, log_var = model(sketch)

            # Randomly select 16 samples from the batch
            num_samples = min(16, sketch.size(0))
            indices = torch.randperm(sketch.size(0))[:num_samples]
            selected_sketches = sketch[indices].cpu().numpy().transpose(0, 2, 3, 1)
            selected_recons = recons[indices].cpu().numpy().transpose(0, 2, 3, 1)

            visualize_sketches(selected_sketches, selected_recons, f"reconstructed_sketches_{name}_{i}", root_dir)

            # Interpolate between the first and last sketches
            mu = mu[indices]
            num_samples = min(len(mu) // 2, 5)
            original_sketch1 = selected_sketches[:num_samples]
            original_sketch2 = selected_sketches[-num_samples:]
            mu1 = mu[:num_samples]
            mu2 = mu[-num_samples:]
            
            decoded_sketches = []
            for alpha in np.linspace(0, 1, 10):
                interpolated_latents = (1 - alpha) * mu1 + alpha * mu2
                decoded_sketch = model.decoder(interpolated_latents).permute(0, 2, 3, 1).cpu().numpy()
                decoded_sketches.append(decoded_sketch)
            
            decoded_sketches = np.stack(decoded_sketches, axis=0)
            visualize_and_save_interpolated_images(original_sketch1, original_sketch2, decoded_sketches, f"interpolated_sketches_{name}_{i}", root_dir)
            
            if i == eval_num - 1:
                break


def visualize_sketches(sketches, reconstructed_sketches, img_name, output_dir):
    rows = 4
    cols = int(np.ceil(len(sketches)/2))
    fig, axs = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    for idx, (sketch, recons) in enumerate(zip(sketches, reconstructed_sketches)):
        print(f"Original - Min: {sketch.min():.4f}, Max: {sketch.max():.4f}")
        print(f"Reconstructed - Min: {recons.min():.4f}, Max: {recons.max():.4f}")
        print(f"MSE Loss: {np.mean((sketch - recons)**2):.4f}")
        row = idx // cols * 2
        col = idx % cols
        axs[row, col].imshow(rescale(sketch))
        axs[row, col].set_title("Original Sketch")
        axs[row, col].axis('off')
        axs[row+1, col].imshow(rescale(recons))
        axs[row+1, col].set_title(f"ReconMSE: {np.mean((sketch - recons)**2):.6f}")
        axs[row+1, col].axis('off')
    plt.savefig(f"{output_dir}/{img_name}.png")
    plt.close()


def visualize_and_save_interpolated_images(original_sketch1, original_sketch2, decoded_sketches, img_name, output_dir):
    num_interpolations = decoded_sketches.shape[0]
    rows = len(original_sketch1)
    cols = num_interpolations + 2
    
    # Total columns should be num_interpolations + 2 (for original sketches)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    for idx in range(rows):
        # Plot original sketches
        axes[idx, 0].imshow(rescale(original_sketch1[idx]))
        axes[idx, 0].set_title("Original Sketch 1")

        # Plot interpolated sketches
        for i in range(num_interpolations):
            axes[idx, i + 1].imshow(rescale(decoded_sketches[i][idx]))
            axes[idx, i + 1].set_title(f"Interp {i + 1}")
        
        axes[idx, cols - 1].imshow(rescale(original_sketch2[idx]))
        axes[idx, cols - 1].set_title("Original Sketch 2")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{img_name}.png")
    plt.close()


import os
import datetime
import torch
from model import VAE
from data_vae import get_sketch_dataloader, normalize, rescale, Thickening
import shutil
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from PIL import Image



def train_vae(model, optimizer, train_loader, val_loader, num_epochs, root_dir, logging_file, args):
    if args.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, 
                               max_lr=args.lr,
                               steps_per_epoch=len(train_loader),
                               epochs=num_epochs,
                               pct_start=0.3)  # 30% of training for the upward phase
        lr_history = np.zeros((num_epochs, len(train_loader)))
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
        lr_history = np.zeros(num_epochs)
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    model.train()

    train_losses = {}
    val_losses = {}
    best_val_loss = np.inf

    for epoch in range(num_epochs):

        # Training Loop
        model.train()
        for i, batch in enumerate(train_loader):
            sketch = batch.cuda()

            optimizer.zero_grad()
            recons, input, mu, log_var = model(sketch)
            loss = model.loss_function(recons, input, mu, log_var, M_N=args.M_N)

            for key in loss:
                if key not in train_losses:
                    train_losses[key] = np.zeros((num_epochs, len(train_loader)))
                train_losses[key][epoch, i] = loss[key].item()
            
            loss["loss"].backward()
            optimizer.step()
            if args.scheduler == 'onecycle':
                scheduler.step()
                lr_history[epoch, i] = optimizer.param_groups[0]['lr']
        
        if args.scheduler == 'cosine':
            scheduler.step()
            lr_history[epoch] = optimizer.param_groups[0]['lr']

        # Validation Loop
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                sketch = batch.cuda()

                recons, input, mu, log_var = model(sketch)
                val_loss_values = model.loss_function(recons, input, mu, log_var, M_N=args.M_N)

                for key in val_loss_values:
                    if key not in val_losses:
                        val_losses[key] = np.zeros((num_epochs, len(val_loader)))
                    val_losses[key][epoch, i] = val_loss_values[key].item()

        if val_losses["loss"][epoch].mean() < best_val_loss:
            best_val_loss = val_losses["loss"][epoch].mean()
            torch.save(model.state_dict(), f'{root_dir}/models/vae_model_best.pth')

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'{root_dir}/models/vae_model_epoch_{epoch}.pth')

        # Calculate and print out the average training and validation loss
        current_lr = optimizer.param_groups[0]['lr']
        message = [f"{key}: {np.mean(train_losses[key][epoch]):.4f}" for key in train_losses]
        message = ", ".join(message)
        print(f"Epoch {epoch+1}/{num_epochs}, train, {message}, Current LR: {current_lr}")
        with open(logging_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}, train, {message}, Current LR: {current_lr}\n")

        message = [f"{key}: {np.mean(val_losses[key][epoch]):.4f}" for key in val_losses]
        message = ", ".join(message)
        print(f"Epoch {epoch+1}/{num_epochs}, val, {message}\n")
        with open(logging_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}, val, {message}\n\n")

    # Save the training and validation losses
    np.save(os.path.join(root_dir, 'train_losses.npy'), train_losses)
    np.save(os.path.join(root_dir, 'val_losses.npy'), val_losses)
    np.save(os.path.join(root_dir, 'lr_history.npy'), lr_history)


def evaluate_model(model, test_loader, root_dir, logging_file, args):
    model.eval()
    test_loss = {}

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            sketch = batch.cuda()
            recons, input, mu, log_var = model(sketch)
            loss = model.loss_function(recons, input, mu, log_var, M_N=args.M_N)

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


def visualize_evaluation(model, test_loader, name, root_dir):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            sketch = batch.cuda()
            recons, input, mu, log_var = model(sketch)

            # Randomly select 10 samples from the batch
            num_samples = min(10, sketch.size(0))
            indices = torch.randperm(sketch.size(0))[:num_samples]
            selected_sketches = sketch[indices].cpu().numpy().transpose(0, 2, 3, 1)
            selected_recons = recons[indices].cpu().numpy().transpose(0, 2, 3, 1)

            # Visualize the first sample in the batch
            visualize_sketches(selected_sketches, selected_recons, f"reconstructed_sketches_{name}_{i}", root_dir)
            
            if i == 5:
                break


def visualize_sketches(sketches, reconstructed_sketches, img_name, root_dir):
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
    plt.savefig(f"{root_dir}/{img_name}.png")
    plt.close()

def load_hand_draw_data(img_size):
    sketches_names = [
        "rgb_images/demo_0_corner_sketch.tiff", 
        "rgb_images/demo_0_corner2_sketch.tiff",
        "rgb_images/demo_6_corner_sketch.tiff", 
        "rgb_images/demo_6_corner2_sketch.tiff",
    ]
    # load the sketches
    sketches = []
    thickening = Thickening(thickness=4)
    for sketch_path in sketches_names:
        img = Image.open(sketch_path).convert('RGB')
        sketch = np.flipud(np.array(img))
        sketch = normalize(sketch)
        # Apply thickening to the sketch
        sketch = thickening(torch.from_numpy(sketch).permute(2, 0, 1))
        sketch = sketch.permute(1, 2, 0).numpy()

        # Resize the sketch to match the desired image size
        sketch = torch.nn.functional.interpolate(
            torch.from_numpy(sketch).unsqueeze(0).permute(0, 3, 1, 2),
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).permute(1, 2, 0).numpy()
        sketches.append(sketch)
    
    sketches = torch.tensor(sketches).permute(0, 3, 1, 2).float() - 0.5
    return sketches


if __name__ == "__main__":    
    train_vae_flag = False

    parser = argparse.ArgumentParser(description="VAE Training Script")
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['onecycle', 'cosine'], help='Learning rate scheduler')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--M_N', type=float, default=0.00025, help='kld weight for loss function')
    parser.add_argument('--root_dir', type=str, default=None, help='Root directory for logging and saving models')
    args = parser.parse_args()

    if train_vae_flag:
        unique_token = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        unique_name = f"vae_{unique_token}_ep{args.num_epochs}_{args.scheduler}_lr{args.lr}_bs{args.bs}_kld{args.M_N}_aug"
    else:
        unique_name = "vae_2024-09-20_03-48-55_ep200_onecycle_lr0.001_bs256_kld0.0001_aug"

    if args.root_dir is None:
        root_dir = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/{os.environ.get('USER')}/VAE_AugNew/{unique_name}"
    else:
        root_dir = f"{args.root_dir}/{unique_name}"
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(f"{root_dir}/models", exist_ok=True)
    shutil.copy(__file__, root_dir)

    img_size = 64
    model = VAE(img_size=img_size, in_channels=3, latent_dim=256).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # SketchDataLoader for training, validation, and test sets
    sketch_train_loader, sketch_val_loader, sketch_test_loader = get_sketch_dataloader(batch_size=args.bs, num_samples=None, img_size=img_size)
    if train_vae_flag:
        # Train the vae model
        logging_file = f'{root_dir}/vae_logging.txt'
        train_vae(model, optimizer, sketch_train_loader, sketch_val_loader, args.num_epochs, root_dir, logging_file, args)

        # Save the trained model
        torch.save(model.state_dict(), f'{root_dir}/models/vae_model_{img_size}.pth')
    else:
        # Load the best model
        model.load_state_dict(torch.load(f'{root_dir}/models/vae_model_64.pth'))
        logging_file = f'{root_dir}/vae_logging_test.txt'

    # Evaluate on the test set
    evaluate_model(model, sketch_test_loader, root_dir, logging_file, args)

    # Visualize results
    visualize_evaluation(model, sketch_test_loader, "test", root_dir)

    # Evaluate on the hand-drawn sketches
    sketches = load_hand_draw_data(img_size)
    sketches = sketches.unsqueeze(0)
    evaluate_model(model, sketches, root_dir, logging_file, args)
    # Visualize results
    visualize_evaluation(model, sketches, "hand_draw", root_dir)

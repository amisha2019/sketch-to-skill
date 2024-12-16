import argparse
import os
import shutil
import datetime
import torch

from data_func.dataloader import get_dataloader
from data_func.dataloader_sketch import get_sketch_dataloader
from model.vae_mlp import VAE_MLP
from train_both import train_vae_mlp, evaluate_model, visualize_evaluation, visualize_inference
from utils.logging import Logger
from utils.hand_drawn import load_hand_draw_data
from utils.arguments import load_args, save_args


def main(args, root_dir=None):

    if os.path.exists(root_dir):
        # Load the arguments
        args = load_args(args, root_dir)
        train_flag = False
        logger = None
    else:
        # Create the directory
        os.makedirs(root_dir)
        os.makedirs(f"{root_dir}/models")
        args.root_dir = root_dir
        save_args(args, root_dir)
        train_flag = True
        # Set up logger
        logger = Logger(args.root_dir, args)

    model = VAE_MLP(img_size=args.img_size, in_channels=3, latent_dim=args.latent_dim, num_control_points=args.num_control_points, degree=args.degree).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader, val_loader, _ = get_dataloader(batch_size=args.bs, img_size=args.img_size, val_split=0.2, test_split=0, data_path="new", logger=logger)
    _, _, test_loader = get_dataloader(batch_size=32, img_size=args.img_size, val_split=0, test_split=1, data_path="test", logger=logger)
    sketch_train_loader, _, _ = get_sketch_dataloader(batch_size=args.bs, img_size=args.img_size, val_split=0, test_split=0, data_path="new", logger=logger)

    if train_flag:
        # load pretrained vae
        if args.load_pretrained_vae:
            model.load_pretrained_vae(f"{args.pretrained_vae_path}/models/vae_model_final.pth")
            if args.freeze_vae:
                model.freeze_vae()
            else:
                optimizer = torch.optim.Adam([
                    {'params': model.encoder.parameters(), 'lr': args.lr * args.vae_lr_weight},
                    {'params': model.decoder.parameters(), 'lr': args.lr * args.vae_lr_weight},
                    {'params': model.mlp.parameters(), 'lr': args.lr}
                ])
        
        # Train the vae model
        train_vae_mlp(model, optimizer, train_loader, val_loader, sketch_train_loader, args, logger)
    else:
        # Load the best model
        model_path = f"{root_dir}/models/vae_mlp_model_best.pth"
        model.load_state_dict(torch.load(model_path))
        
    # Evaluate on the test set
    evaluate_model(model, test_loader, args, "test", logger)

    # Visualize results
    visualize_evaluation(model, test_loader, "test", root_dir)

    # Evaluate on the hand-drawn sketches
    sketches1, sketches2 = load_hand_draw_data(args.img_size)
    sketches = [[sketches1, sketches2]]
    # Visualize inference results on hand-drawn sketches
    visualize_inference(model, sketches, "hand_draw", root_dir, eval_num=5)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="VAE MLP Training Script")
    parser.add_argument('--img_size', type=int, default=64, help='Size of the input images')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space')
    parser.add_argument('--num_control_points', type=int, default=20, help='Number of control points')
    parser.add_argument('--degree', type=int, default=3, help='Degree of the B-spline')

    parser.add_argument('--project', type=str, default=None, help='Project name for wandb')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['onecycle', 'cosine'], help='Learning rate scheduler')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--bs', type=int, default=256, help='Batch size for training')
    parser.add_argument('--M_N', type=float, default=0.0001, help='kld weight for loss function')
    parser.add_argument('--kld_anneal', action='store_true', help='Use linear annealing for kld weight')
    
    parser.add_argument('--load_pretrained_vae', action='store_true', help='Flag to load pretrained VAE model')
    parser.add_argument('--freeze_vae', action='store_true', help='Freeze the pretrained VAE weights')
    parser.add_argument('--pretrained_vae_path', type=str, default=None, help='Path to pretrained VAE model')
    parser.add_argument('--vae_lr_weight', type=float, default=0.1, help='Learning rate weight for the VAE')
    args = parser.parse_args()

    if args.project is None:
        args.project = f"VAE_MLP_Both_NewData_Unorganized"

    if args.load_pretrained_vae:
        args.pretrained_vae_path = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_Aug_Standardize_NewData/{args.pretrained_vae_path}"
        print(f"pretrained vae path: {args.pretrained_vae_path}")
    
    unique_token = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    unique_name = f"vae_mlp_ep{args.num_epochs}_bs{args.bs}_dim{args.latent_dim}_{args.scheduler}_lr{args.lr}_kld{args.M_N}_aug"
    if args.load_pretrained_vae:
        unique_name += "_pretrained"
    if args.freeze_vae:
        unique_name += "_freeze"
    if args.kld_anneal:
        unique_name += "_Anneal"
    unique_name += f"_{unique_token}"

    args.unique_token = unique_name

    # for testing
    # unique_name = "vae_mlp_2024-09-22_21-37-28_ep2_cosine_lr0.001_bs256_kld0.0001_aug"

    root_dir = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/{os.environ.get('USER')}/{args.project}/{unique_name}"
    
    main(args, root_dir)

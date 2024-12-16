import argparse
import os
import shutil
import datetime
import torch

from data_func.dataloader import get_dataloader
from data_func.dataloader_sketch import get_sketch_dataloader
from model.vae_mlp import VAE_MLP
from train_vae_mlp import train_vae_mlp, evaluate_model, visualize_evaluation, visualize_inference
from utils.logging import Logger
from utils.hand_drawn import load_hand_draw_data, load_hand_draw_data_new
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

    model = VAE_MLP(img_size=args.img_size, in_channels=3, latent_dim=args.latent_dim, num_control_points=args.num_control_points, degree=args.degree, num_sketches=args.num_sketches, use_traj_rescale=args.use_traj_rescale, disable_vae=args.disable_vae).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    data_path = "robomimic" if args.data_name == "can" or args.data_name == "square" else "new"
    train_loader, val_loader, _ = get_dataloader(batch_size=args.bs, img_size=args.img_size, num_control_points=args.num_control_points, num_samples=args.num_samples, val_split=0.15, test_split=0, data_path=data_path, data_name=args.data_name, use_data_aug=not args.disable_data_aug, logger=logger)
    
    # get test dataloader
    if args.data_name is None:
        test_loaders = {'ButtonPress': None,
                        'ButtonPressTopdownWall': None,
                        'DrawerOpen': None,
                        'Reach': None,
                        'ReachWall': None}
        for key in test_loaders.keys():
            _, _, test_loader = get_dataloader(batch_size=args.bs, img_size=args.img_size, num_control_points=args.num_control_points, val_split=0, test_split=1, data_path="test", data_name=key, logger=logger)
            test_loaders[key] = test_loader
    else:
        _, _, test_loader = get_dataloader(batch_size=args.bs, img_size=args.img_size, num_control_points=args.num_control_points, val_split=0, test_split=1, data_path="test", data_name=args.data_name, logger=logger)
        test_loaders = {args.data_name: test_loader} if test_loader is not None else {}

    # get sketch dataloader
    if not args.disable_vae and args.train_both:
        sketch_train_loader, _, _ = get_sketch_dataloader(batch_size=args.bs, img_size=args.img_size, val_split=0, test_split=0, data_path="new", data_name=args.data_name, use_data_aug=not args.disable_data_aug, logger=logger)
    else:
        sketch_train_loader = None

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
        train_vae_mlp(model, optimizer, train_loader, val_loader, test_loaders, sketch_train_loader, args, logger)
    else:
        # Load the best model
        model_path = f"{root_dir}/models/vae_mlp_model_best.pth"
        model.load_state_dict(torch.load(model_path))
        
    # Evaluate on the test set
    for key in test_loaders.keys():
        # Visualize results
        visualize_evaluation(model, test_loaders[key], f"{key}_test", root_dir)

        # Evaluate on the hand-drawn sketches
        if args.use_traj_rescale:
            sketches1, sketches2, starts, ends = load_hand_draw_data_new(key, True)
            data = [[sketches1, sketches2, starts, ends]]
        else:
            sketches1, sketches2 = load_hand_draw_data_new(key)
            data = [[sketches1, sketches2, None, None]]
        # Visualize inference results on hand-drawn sketches
        visualize_inference(model, data, f"{key}_hand_draw", root_dir, eval_num=5)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="VAE MLP Training Script")
    parser.add_argument('--num_sketches', type=int, default=2, help='Number of sketches to use')
    parser.add_argument('--img_size', type=int, default=64, help='Size of the input images')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space')
    parser.add_argument('--num_control_points', type=int, default=20, help='Number of control points')
    parser.add_argument('--degree', type=int, default=3, help='Degree of the B-spline')

    parser.add_argument('--project', type=str, default=None, help='Project name for wandb')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
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

    parser.add_argument('--train_both', action='store_true', help='Train both vae and mlp')
    parser.add_argument('--use_traj_rescale', action='store_true', help='Use trajectory rescaling')
    parser.add_argument('--disable_vae', action='store_true', help='Disable VAE loss')
    parser.add_argument('--disable_data_aug', action='store_true', help='Disable data augmentation')

    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to use (if None, use all)')
    parser.add_argument('--data_name', type=str, default=None, help='Name of the data to use')
    args = parser.parse_args()

    if args.project is None:
        if args.train_both:
            args.project = f"VAE_MLP_Both_NewData_Unorganized"
        else:
            args.project = f"VAE_MLP_NewData_Unorganized"

    if args.load_pretrained_vae:
        args.pretrained_vae_path = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_Aug_Standardize_NewData/{args.pretrained_vae_path}"
        print(f"pretrained vae path: {args.pretrained_vae_path}")
    
    unique_token = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.data_name is not None:
        prefix = f"{args.data_name}_"
    else:
        prefix = ""
    unique_name = f"{prefix}vae_mlp_{args.num_sketches}sketches_{args.num_control_points}cp_ep{args.num_epochs}_bs{args.bs}_dim{args.latent_dim}_{args.scheduler}_lr{args.lr:.5f}_kld{args.M_N:.5f}"
    if not args.disable_data_aug:
        unique_name += "_aug"
    if args.load_pretrained_vae:
        unique_name += "_pretrained"
    if args.freeze_vae:
        unique_name += "_freeze"
    if args.kld_anneal:
        unique_name += "_Anneal"
    if args.num_samples is not None:
        unique_name += f"_num{args.num_samples}"
    if args.use_traj_rescale:
        unique_name += "_rescale"
    if args.disable_vae:
        unique_name += "_novae"
    unique_name += f"_{unique_token}"

    args.unique_token = unique_name

    # for testing
    # unique_name = "vae_mlp_2024-09-22_21-37-28_ep2_cosine_lr0.001_bs256_kld0.0001_aug"

    root_dir = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/{os.environ.get('USER')}/{args.project}/{unique_name}"
    
    main(args, root_dir)

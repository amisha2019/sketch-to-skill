import argparse
import os
import shutil
import datetime
import torch

from data_func.dataloader_sketch import get_sketch_dataloader
from model.vae import VAE
from train_vae import train_vae, evaluate_model, visualize_evaluation
from utils.logging import Logger
from utils.arguments import load_args, save_args
from utils.hand_drawn import load_hand_draw_data


def main(args, root_dir=None):

    if os.path.exists(root_dir):
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
        logger = Logger(args.root_dir, args)
    
    model = VAE(
        img_size=args.img_size, 
        in_channels=3, 
        latent_dim=args.latent_dim, 
        ifPretrained=args.ifPretrained, 
        preTrainedModel_type=args.typeOfPretrainedModel, 
        preTrainedModel_layers=args.layersOfPreTrainedModel, 
        freeze=args.ifFreezePretrainedModel
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sketch_train_loader, sketch_val_loader, _ = get_sketch_dataloader(batch_size=args.bs, img_size=args.img_size, val_split=0.2, test_split=0, data_path="new", logger=logger)
    _, _, sketch_test_loader = get_sketch_dataloader(batch_size=args.bs, img_size=args.img_size, val_split=0, test_split=1, data_path="test", logger=logger)
    
    if train_flag:
        # Train the vae model
        train_vae(model, optimizer, sketch_train_loader, sketch_val_loader, args, logger)
    else:
        # Load the best model
        model_path = f"{root_dir}/models/vae_model_final.pth"
        model.load_state_dict(torch.load(model_path))

    # Evaluate on the test set
    evaluate_model(model, sketch_test_loader, args, "test", logger)

    # Visualize results
    visualize_evaluation(model, sketch_test_loader, "test", root_dir)

    # Evaluate on the hand-drawn sketches
    sketches1, sketches2 = load_hand_draw_data(args.img_size)
    sketches = torch.cat([sketches1, sketches2], dim=0).unsqueeze(0)
    evaluate_model(model, sketches, args, "hand_draw", logger)

    # Visualize results
    visualize_evaluation(model, sketches, "hand_draw", root_dir, eval_num=5)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="VAE Training Script")
    parser.add_argument('--img_size', type=int, default=64, help='Size of the input images')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space')

    parser.add_argument('--project', type=str, default=None, help='Project name for wandb')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['onecycle', 'cosine'], help='Learning rate scheduler')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--bs', type=int, default=256, help='Batch size for training')
    parser.add_argument('--M_N', type=float, default=0.0001, help='kld weight for loss function')
    parser.add_argument('--kld_anneal', action='store_true', help='Use linear annealing for kld weight')

    parser.add_argument('--ifPretrained', action='store_true', help='if pretrained')
    parser.add_argument('--typeOfPretrainedModel', type=str, default='Resnet', help='type of pretrained model')
    parser.add_argument('--layersOfPreTrainedModel', type=int, default=6, help='number of layers of pretrained model')
    parser.add_argument('--ifFreezePretrainedModel', action='store_true', help='freeze pretrained model')

    args = parser.parse_args()

    if args.project is None:
        args.project = f"VAE_Aug_Standardize_NewData_Unorganized"
    
    unique_token = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    unique_name = f"vae_new_ep{args.num_epochs}_bs{args.bs}_dim{args.latent_dim}_{args.scheduler}_lr{args.lr}_kld{args.M_N}"
    if args.kld_anneal:
        unique_name += "_Anneal"
    if args.ifPretrained:
        unique_name += f"_{args.typeOfPretrainedModel}_{args.layersOfPreTrainedModel}Layer_{'Frozen' if args.ifFreezePretrainedModel else 'Trainable'}"
    unique_name += f"_{unique_token}"

    args.unique_token = unique_name

    # for testing
    # unique_name = "vae_new_2024-09-22_15-50-16_ep200_onecycle_lr0.001_bs256_kld0.0001_aug"

    root_dir = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/{os.environ.get('USER')}/{args.project}/{unique_name}"
    
    main(args, root_dir)

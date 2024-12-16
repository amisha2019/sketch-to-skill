import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from data_func.dataloader_helper import normalize, standardize, rescale, Thickening
from model.vae_mlp import VAE_MLP


def load_data(f_name, data_type):
    img_size = 64
    num_control_points = 20
    if data_type == "test":
        root_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/demo_datasets"
    elif data_type == "train":
        root_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_new"
    else:
        raise ValueError(f"Invalid data type: {data_type}")
    sketch1 = torch.load(f'{root_path}/{f_name}/{f_name}_sketches1_{img_size}_cropped.pt')
    sketch2 = torch.load(f'{root_path}/{f_name}/{f_name}_sketches2_{img_size}_cropped.pt')
    traj_gt = torch.load(f'{root_path}/{f_name}/{f_name}_trajectories_raw.pt')
    traj_dense = torch.load(f'{root_path}/{f_name}/{f_name}_fitted_trajectories_50.pt')
    params = torch.load(f'{root_path}/{f_name}/{f_name}_params_{num_control_points}.pt')
    fitted_traj = torch.load(f'{root_path}/{f_name}/{f_name}_fitted_trajectories_{num_control_points}.pt')
    return sketch1, sketch2, traj_gt, traj_dense, params, fitted_traj

def load_hand_drawn_data(f_name):
    root_path = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/hand_drawn_data/sketches/{f_name}"
    sketch1_names = ["demo_0_corner_image_sketch_64",
                     "demo_1_corner_image_sketch_64"
                     "demo_10_corner_image_sketch_64"]
    sketch2_names = ["demo_0_corner2_image_sketch_64",
                     "demo_1_corner2_image_sketch_64",
                     "demo_10_corner2_image_sketch_64"]
    sketches1 = []
    sketches2 = []
    for sketch1_name, sketch2_name in zip(sketch1_names, sketch2_names):
        sketch1 = cv2.imread(f"{root_path}/{sketch1_name}.png")
        sketch1 = cv2.cvtColor(sketch1, cv2.COLOR_BGR2RGB)
        sketch1 = sketch1.permute(2, 0, 1)
        sketches1.append(sketch1)
        sketch2 = cv2.imread(f"{root_path}/{sketch2_name}.png")
        sketch2 = cv2.cvtColor(sketch2, cv2.COLOR_BGR2RGB)
        sketch2 = sketch2.permute(2, 0, 1)
        sketches2.append(sketch2)
    return sketches1, sketches2


def load_model(model_path):
    model_path = f"{model_path}/models/vae_mlp_model_best.pth"
    if "rescale" in model_path:
        model = VAE_MLP(img_size=64, in_channels=3, latent_dim=32, num_control_points=20, degree=3, use_traj_rescale=True).cuda()
    else:
        model = VAE_MLP(img_size=64, in_channels=3, latent_dim=32, num_control_points=20, degree=3).cuda()
    model.load_state_dict(torch.load(model_path))
    return model


def inference(model, sketch1, sketch2, traj):
    sketch1 = apply_dilation(sketch1)
    sketch2 = apply_dilation(sketch2)
    sketch1, sketch2, traj = sketch1.cuda(), sketch2.cuda(), traj.cuda()
    print(f"max of sketch1: {torch.max(sketch1)}, min of sketch1: {torch.min(sketch1)}")
    print(f"max of sketch2: {torch.max(sketch2)}, min of sketch2: {torch.min(sketch2)}")
    with torch.no_grad():
        recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
        generated_traj = model.generate_trajectory(params)
        if model.use_traj_rescale:
            generated_traj = model.rescale_traj(generated_traj, traj[:, 0, :], traj[:, -1, :])
    sketch1 = sketch1.cpu()
    sketch2 = sketch2.cpu()
    recons1 = recons1.cpu()
    recons2 = recons2.cpu()
    generated_traj = generated_traj.cpu()
    return sketch1, sketch2, recons1, recons2, generated_traj


def apply_dilation(sketches):
    sketches = normalize(sketches)
    thickener = Thickening(thickness=2)
    for i in range(len(sketches)):
        sketches[i] = thickener(sketches[i])
    sketches = standardize(sketches)
    return sketches


def visualize_trajectory(sketches1, sketches2, recons1, recons2, traj_gt, traj_dense, fitted_traj, generated_traj, img_name, output_dir):
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

        ax5 = fig.add_subplot(rows, num_samples, 4 * num_samples + i + 1, projection='3d')
        ax5.plot(traj_gt[i][:, 0], traj_gt[i][:, 1], traj_gt[i][:, 2], 'b-', alpha=0.5, linewidth=2)
        ax5.scatter(fitted_traj[i][:, 0], fitted_traj[i][:, 1], fitted_traj[i][:, 2], c=np.arange(len(fitted_traj[i])), cmap=cmap, s=20)
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_zlabel('Z')
        ax5.set_title(f"Sample {i + 1}: Fitted Trajectory")
        ax5.set_box_aspect((1, 1, 1))
        ax5.set_aspect('equal')

        ax6 = fig.add_subplot(rows, num_samples, 5 * num_samples + i + 1, projection='3d')
        ax6.plot(generated_traj[i][:, 0], generated_traj[i][:, 1], generated_traj[i][:, 2], 'r-', alpha=0.5, linewidth=2)
        ax6.scatter(traj_dense[i][:, 0], traj_dense[i][:, 1], traj_dense[i][:, 2], c=np.arange(len(traj_dense[i])), cmap=cmap, s=20)
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')
        ax6.set_title(f"Generated Traj MSE: {np.mean((fitted_traj[i] - generated_traj[i])**2):.6f}")
        ax6.set_box_aspect((1, 1, 1))
        ax6.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{img_name}.png')
    plt.close()


def inference_and_visualize(model, data_file_name, data_type):
    sketch1, sketch2, traj_gt, traj_dense, params, fitted_traj = load_data(data_file_name, data_type)
    sketch1, sketch2, recons1, recons2, generated_traj = inference(model, sketch1, sketch2, traj_dense)
    print("Number of samples: ", sketch1.shape[0])
    num_samples = min(10, sketch1.shape[0])
    idxs = np.random.choice(sketch1.shape[0], num_samples, replace=False)

    sketch1 = sketch1[idxs].numpy().transpose(0, 2, 3, 1)
    sketch2 = sketch2[idxs].numpy().transpose(0, 2, 3, 1)
    recons1 = recons1[idxs].numpy().transpose(0, 2, 3, 1)
    recons2 = recons2[idxs].numpy().transpose(0, 2, 3, 1)
    traj_gt = [traj_gt[i] for i in idxs]
    traj_dense = traj_dense[idxs].numpy()
    fitted_traj = fitted_traj[idxs].numpy()
    generated_traj = generated_traj[idxs].numpy()

    visualize_trajectory(sketch1, sketch2, recons1, recons2, traj_gt, traj_dense, fitted_traj, generated_traj, f"generated_trajectory_{data_file_name}_inference_{data_type}", model_path)
    

if __name__ == "__main__":
    # model_pathes = ['/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.0001_aug_2024-09-26_13-58-16',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.1_aug_2024-09-26_13-58-25',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.001_aug_2024-09-26_13-58-25',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.0001_aug_Anneal_2024-09-26_13-58-22',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.001_aug_Anneal_2024-09-26_13-58-25',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.1_aug_Anneal_2024-09-26_13-58-31',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.0005_aug_2024-09-26_13-58-22',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.0005_aug_Anneal_2024-09-26_13-58-25',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.00025_aug_2024-09-26_13-58-22',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.00025_aug_Anneal_2024-09-26_13-58-22',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.0001_aug_2024-09-26_13-58-09',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.001_aug_2024-09-26_13-58-13',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.1_aug_2024-09-26_13-58-16',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.0001_aug_Anneal_2024-09-26_13-58-09',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.1_aug_Anneal_2024-09-26_13-58-16',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.001_aug_Anneal_2024-09-26_13-58-16',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.0005_aug_2024-09-26_13-58-13',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.0005_aug_Anneal_2024-09-26_13-58-13',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.00025_aug_2024-09-26_13-58-09',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.00025_aug_Anneal_2024-09-26_13-58-13',]
    # model_pathes = ['/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.001_aug_2024-09-26_13-58-25 [bad at ButtonPressTopDownWall, others are reasonable]',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.001_aug_2024-09-26_13-58-13 [bad at ButtonPressTopDownWall, others are reasonable]',
    #                 '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.0005_aug_2024-09-26_13-58-22 [somewhat better]',]
    model_pathes = ['/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_0927_rescale/vae_mlp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00010_aug_2024-09-27_06-44-04',]


    data_file_names = ['ButtonPress',]
                    # 'ButtonPressTopdownWall',
                    # 'DrawerOpen',
                    # 'Reach',
                    # 'ReachWall',]
    
    for model_path in model_pathes:
        print(f"Model path: {model_path}")
        model = load_model(model_path)
        for data_file_name in data_file_names:
            print(f"Data file name: {data_file_name}")
            inference_and_visualize(model, data_file_name, "test")
            # breakpoint()
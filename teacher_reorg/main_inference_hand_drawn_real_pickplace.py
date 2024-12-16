import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import h5py
import yaml
from torchvision import transforms
from data_func.dataloader_helper import normalize, standardize, rescale, RandomThicken, RandomThinning
from model.vae_mlp import VAE_MLP
from utils.hand_drawn import load_hand_draw_data_real_pickplace


rescale_num = 1

def load_data(f_name):
    img_size = 64
    num_control_points = 20
    root_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/demo_datasets"
    # root_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_new"
    sketch1 = torch.load(f'{root_path}/{f_name}/{f_name}_sketches1_{img_size}_cropped.pt')
    sketch2 = torch.load(f'{root_path}/{f_name}/{f_name}_sketches2_{img_size}_cropped.pt')
    traj_gt = torch.load(f'{root_path}/{f_name}/{f_name}_trajectories_raw.pt')
    traj_dense = torch.load(f'{root_path}/{f_name}/{f_name}_fitted_trajectories_50.pt')
    params = torch.load(f'{root_path}/{f_name}/{f_name}_params_{num_control_points}.pt')
    fitted_traj = torch.load(f'{root_path}/{f_name}/{f_name}_fitted_trajectories_{num_control_points}.pt')
    return sketch1, sketch2, traj_gt, traj_dense, params, fitted_traj


def load_model(model_path):
    argfile = f"{model_path}/args.yaml"
    with open(argfile, 'r') as file:
        args = yaml.safe_load(file)
    latent_dim = args["latent_dim"]
    num_control_points = args["num_control_points"]

    model_path = f"{model_path}/models/vae_mlp_model_final.pth"
    # model_path = f"{model_path}/models/vae_mlp_model_epoch_180.pth"
    if "rescale" in model_path:
        model = VAE_MLP(img_size=64, in_channels=3, latent_dim=latent_dim, num_control_points=num_control_points, degree=3, use_traj_rescale=True).cuda()
    else:
        model = VAE_MLP(img_size=64, in_channels=3, latent_dim=latent_dim, num_control_points=num_control_points, degree=3).cuda()
    model.load_state_dict(torch.load(model_path))
    return model


def inference(model, sketch1, sketch2, starts, ends, num_gen=10):
    sketch1, sketch2 = sketch1.cuda(), sketch2.cuda()
    print(f"max of sketch1: {torch.max(sketch1)}, min of sketch1: {torch.min(sketch1)}")
    print(f"max of sketch2: {torch.max(sketch2)}, min of sketch2: {torch.min(sketch2)}")
    with torch.no_grad():
        generated_trajs = []
        for i in range(num_gen):
            recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
            generated_traj = model.generate_trajectory(params, num_points=50)
            if model.use_traj_rescale and starts is not None and ends is not None:
                sample_num = generated_traj.shape[0]
                rescale_z = torch.ones(sample_num)
                rescale_z[sample_num//2:] = rescale_num
                generated_traj = model.rescale_traj(generated_traj, starts.cuda(), ends.cuda(), rescale_z)
            generated_trajs.append(generated_traj.cpu())
    sketch1 = sketch1.cpu()
    sketch2 = sketch2.cpu()
    recons1 = recons1.cpu()
    recons2 = recons2.cpu()
    return sketch1, sketch2, recons1, recons2, generated_trajs



def visualize_trajectory(sketches1, sketches2, recons1, recons2, trajs_gt, generated_trajs, img_name, output_dir):
    rows = 5
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
        for generated_traj in generated_trajs:
            ax5.plot(generated_traj[i][:, 0], generated_traj[i][:, 1], generated_traj[i][:, 2], alpha=0.5, linewidth=2)
        ax5.scatter(trajs_gt[i][:, 0], trajs_gt[i][:, 1], trajs_gt[i][:, 2], c=np.arange(len(trajs_gt[i])), cmap=cmap, s=5)
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_zlabel('Z')
        ax5.set_title(f"Generated Traj")
        ax5.set_box_aspect((1, 1, 1))
        ax5.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{img_name}.png')
    plt.close()


def save_trajectory(generated_trajs, data_file_name, file_name, output_dir):
    # Check if generated_trajs is of length 3
    # if generated_trajs[0].shape[0] != 30:
    #     print(f"Warning: Expected 30 generated trajectories, but got {len(generated_trajs)}")
    #     return

    traj_dict = {}
    traj_dict_ft2 = {}
    if data_file_name == "toast_pick_place":
        sample_ids = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    elif data_file_name == "square":
        sample_ids = np.arange(50)
    elif data_file_name == "can":
        sample_ids = np.arange(1, 21)
    num_stages = len(generated_trajs[0]) // len(sample_ids)
    props = [1, 0]
    for i in range(len(sample_ids)):
        print(f"sample_id: {sample_ids[i]}")
        key = f"demo_{sample_ids[i]}"
        all_trajs = []
        for gen_traj in generated_trajs:
            combined_trajs = []
            for s in range(num_stages):
                traj_idx = s * len(sample_ids) + i
                print(f"stage: {s}, traj_idx: {traj_idx}")
                combined_trajs.append(np.hstack((gen_traj[traj_idx], np.ones((gen_traj[traj_idx].shape[0],1))*props[s])))
            all_trajs.append(np.vstack(combined_trajs))

        all_trajs = np.array(all_trajs)
        traj_dict[key] = {"obs": {}}
        traj_dict[key]["obs"]["robot0_eef_pos"] = all_trajs[:, :, :3]
        gripper_qpos = np.zeros((*all_trajs.shape[:2], 2))
        gripper_qpos[:, :, 0] = all_trajs[:, :, -1]
        traj_dict[key]["obs"]["robot0_gripper_qpos"] = gripper_qpos

        for i in range(len(generated_trajs)):
            traj_dict_ft2[f"{key}_gen{i}"] = {"obs": {}}
            traj_dict_ft2[f"{key}_gen{i}"]["obs"]["robot0_eef_pos"] = all_trajs[i][:, :3]
            traj_dict_ft2[f"{key}_gen{i}"]["obs"]["robot0_gripper_qpos"] = np.zeros((all_trajs[i].shape[0], 2))
            traj_dict_ft2[f"{key}_gen{i}"]["obs"]["robot0_gripper_qpos"][:, 0] = all_trajs[i][:, -1]

    with h5py.File(f'{output_dir}/{file_name}.hdf5', 'w') as f:
        for key, value in traj_dict.items():
            # Create a group for each demo
            grp = f.create_group(key)
            # Create a group for observations
            obs_grp = grp.create_group('obs')
            # Save each observation array separately
            obs_grp.create_dataset('robot0_eef_pos', data=value['obs']['robot0_eef_pos'])
            obs_grp.create_dataset('robot0_gripper_qpos', data=value['obs']['robot0_gripper_qpos'])
    
    with h5py.File(f'{output_dir}/{file_name}_ft2.hdf5', 'w') as f:
        for key, value in traj_dict_ft2.items():
            # Create a group for each demo
            grp = f.create_group(key)
            # Create a group for observations
            obs_grp = grp.create_group('obs')
            # Save each observation array separately
            obs_grp.create_dataset('robot0_eef_pos', data=value['obs']['robot0_eef_pos'])
            obs_grp.create_dataset('robot0_gripper_qpos', data=value['obs']['robot0_gripper_qpos'])

    print(f"Trajectory saved to {output_dir}/{file_name}.hdf5")
    print(f"Trajectory saved to {output_dir}/{file_name}_ft2.hdf5")

def data_augmentation(images):
    for i in range(len(images)):
        im = images[i]
        transform = transforms.ElasticTransform(alpha=50.0, sigma=5.0, fill=0)
        im = transform(im)
        images[i] = im
    return images


def inference_and_visualize(model, data_file_name, save_path, use_data_aug):
    use_traj_rescale = True
    sketches1, sketches2, starts, ends, trajs_gt = load_hand_draw_data_real_pickplace(data_file_name, use_traj_rescale, load_traj=True)
    if use_data_aug:
        sketches1 = data_augmentation(sketches1)
        sketches2 = data_augmentation(sketches2)
    sketch1, sketch2, recons1, recons2, generated_trajs = inference(model, sketches1, sketches2, starts, ends)
    print("Number of samples: ", sketch1.shape[0])

    sketch1 = sketch1.numpy().transpose(0, 2, 3, 1)
    sketch2 = sketch2.numpy().transpose(0, 2, 3, 1)
    recons1 = recons1.numpy().transpose(0, 2, 3, 1)
    recons2 = recons2.numpy().transpose(0, 2, 3, 1)
    generated_trajs = [traj.numpy() for traj in generated_trajs]

    suffix = f"_rescale{rescale_num}" if rescale_num != 1 else ""
    suffix += "_aug" if use_data_aug else ""
    visualize_trajectory(sketch1, sketch2, recons1, recons2, trajs_gt, generated_trajs, f"generated_trajectory_{data_file_name}_inference_hand_drawn{suffix}", save_path)
    save_trajectory(generated_trajs, data_file_name, f"generated_trajectory_{data_file_name}_inference_hand_drawn{suffix}", save_path)
    # save_trajectory(generated_trajs, f"generated_trajectory_{data_file_name}_inference_hand_drawn", save_path)
    

if __name__ == "__main__":
    model_pathes = [
        "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_toast_pick_place_Real/toast_pick_placeReal_vae_mlp_2sketches_10cp_ep200_bs64_dim32_cosine_lr0.00010_kld0.00010_aug_Anneal_num100_rescale_2024-11-17_04-14-02",
        "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_toast_pick_place_Real/toast_pick_placeReal_vae_mlp_2sketches_10cp_ep200_bs64_dim32_cosine_lr0.00010_kld0.00010_aug_num100_rescale_2024-11-17_04-13-34",
        "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_toast_pick_place_Real/toast_pick_placeReal_vae_mlp_2sketches_10cp_ep200_bs64_dim32_cosine_lr0.00100_kld0.00010_aug_Anneal_num100_rescale_2024-11-17_04-13-56",
        "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_toast_pick_place_Real/toast_pick_placeReal_vae_mlp_2sketches_10cp_ep200_bs64_dim32_cosine_lr0.00100_kld0.00010_aug_num100_rescale_2024-11-17_04-13-36",
        "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_toast_pick_place_Real/toast_pick_placeReal_vae_mlp_2sketches_20cp_ep200_bs64_dim32_cosine_lr0.00010_kld0.00010_aug_Anneal_num100_rescale_2024-11-17_04-14-02",
        "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_toast_pick_place_Real/toast_pick_placeReal_vae_mlp_2sketches_20cp_ep200_bs64_dim32_cosine_lr0.00010_kld0.00010_aug_num100_rescale_2024-11-17_04-13-34",
        "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_toast_pick_place_Real/toast_pick_placeReal_vae_mlp_2sketches_20cp_ep200_bs64_dim32_cosine_lr0.00100_kld0.00010_aug_Anneal_num100_rescale_2024-11-17_04-13-56",
        "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_toast_pick_place_Real/toast_pick_placeReal_vae_mlp_2sketches_20cp_ep200_bs64_dim32_cosine_lr0.00100_kld0.00010_aug_num100_rescale_2024-11-17_04-13-36",
    ]
    data_file_names = [
        'toast_pick_place',
    ]
    
    model_pathes = [
        "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_2sketch_sep_tasks_split_traj_robomimic_50val/square_vae_mlp_2sketches_20cp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00010_aug_Anneal_rescale_2024-11-22_02-18-36",
        "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_2sketch_sep_tasks_split_traj_robomimic_50val/square_vae_mlp_2sketches_20cp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00010_aug_rescale_2024-11-22_02-18-41",
        "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_2sketch_sep_tasks_split_traj_robomimic_50val/square_vae_mlp_2sketches_20cp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00050_aug_Anneal_rescale_2024-11-22_02-18-36",
        "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_2sketch_sep_tasks_split_traj_robomimic_50val/square_vae_mlp_2sketches_20cp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00050_aug_rescale_2024-11-22_02-18-36",
    ]
    data_file_names = ["square"]
    num_stages = 2

    # model_pathes = [
    #     "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_2sketch_sep_tasks_split_traj_robomimic_20val/can_vae_mlp_2sketches_20cp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00010_aug_Anneal_rescale_2024-11-20_13-54-40",
    #     "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_2sketch_sep_tasks_split_traj_robomimic_20val/can_vae_mlp_2sketches_20cp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00010_aug_rescale_2024-11-20_13-54-37",
    #     "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_2sketch_sep_tasks_split_traj_robomimic_20val/can_vae_mlp_2sketches_20cp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00050_aug_Anneal_rescale_2024-11-20_13-54-39",
    #     "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_2sketch_sep_tasks_split_traj_robomimic_20val/can_vae_mlp_2sketches_20cp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00050_aug_rescale_2024-11-20_13-54-40",
    # ]
    # data_file_names = ["can"]
    # num_stages = 2

    use_data_aug = False

    for model_path in model_pathes:
        print(f"Model path: {model_path}")
        model = load_model(model_path)
        for data_file_name in data_file_names:
            print(f"Data file name: {data_file_name}")
            inference_and_visualize(model, data_file_name, model_path, use_data_aug)
            # breakpoint()
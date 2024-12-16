python datasetv3.py \
    --dataset "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_0927_rescale/vae_mlp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00010_aug_2024-09-27_06-44-04/generated_trajectory_ButtonPress_inference_hand_drawn.hdf5" \
    --env_name "ButtonPress" 


# python teacher_model/fit_traj/datasetv2.py \
#     --dataset "/fs/nexus-projects/Sketch_VLM_RL/amishab/demo_dataset_bc_96_new/DrawerOpen_frame_stack_1_96x96_end_on_success/dataset.hdf5" \
#     --env_name "DrawerOpen" 

# python teacher_model/fit_traj/datasetv2.py \
#     --dataset "/fs/nexus-projects/Sketch_VLM_RL/amishab/demo_dataset_bc_96_new/Reach_frame_stack_1_96x96_end_on_success/dataset.hdf5" \
#     --env_name "Reach" 

# python teacher_model/fit_traj/datasetv2.py \
#     --dataset "/fs/nexus-projects/Sketch_VLM_RL/amishab/demo_dataset_bc_96_new/CoffeeButton_frame_stack_1_96x96_end_on_success/dataset.hdf5" \
#     --env_name "CoffeeButton" 

# python teacher_model/fit_traj/datasetv2.py \
#     --dataset "/fs/nexus-projects/Sketch_VLM_RL/amishab/demo_dataset_bc_96_new/ButtonPressTopdownWall_frame_stack_1_96x96_end_on_success/dataset.hdf5" \
#     --env_name "ButtonPressTopdownWall" 

# python teacher_model/fit_traj/datasetv2.py \
#     --dataset "/fs/nexus-projects/Sketch_VLM_RL/amishab/demo_dataset_bc_96_new/ReachWall_frame_stack_1_96x96_end_on_success/dataset.hdf5" \
#     --env_name "ReachWall" 
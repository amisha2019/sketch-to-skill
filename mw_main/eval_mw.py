import pickle
import torch
import numpy as np

from env.metaworld_wrapper import PixelMetaWorld
from common_utils import ibrl_utils as utils
from common_utils import Recorder
import cv2


def run_eval(
    env: PixelMetaWorld,
    agent,
    num_game,
    seed,
    record_dir=None,
    verbose=True,
    eval_mode=True,
):
    recorder = None if record_dir is None else Recorder(record_dir)

    scores = []

    # # Render images
    # cv2.namedWindow("render", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("render", 600, 600)

    # Set Randomization
    # env.obj_rand_init(True)

    with torch.no_grad(), utils.eval_mode(agent):
        for episode_idx in range(num_game):
            step = 0
            rewards = []
            np.random.seed(seed + episode_idx)
            obs, image_obs = env.reset()
            
            # Print obj pose
            # print(f"obj pose: {env.get_obj_pose_sktchRL()}")

            terminal = False
            while not terminal:
                if recorder is not None:
                    recorder.add(image_obs)

                # Filter to get 7 dimensions in obs["state"]
                # -----------------------------------------------------
                # ee_pose = obs["state"][:4].cpu()  # Move tensor to CPU
                # goal_pose = obs["state"][-3:].cpu()  # Move tensor to CPU
                # obs["state"] = np.concatenate([ee_pose.numpy(), goal_pose.numpy()], axis=0)
                # obs["state"] = torch.from_numpy(obs["state"]).to("cuda")
                # ------------------------------------------------------

                action = agent.act(obs, eval_mode=eval_mode).numpy()
                obs, reward, terminal, _, image_obs = env.step(action)
                rewards.append(reward)
                step += 1

                # # Render code
                # # img = image_obs["corner2"]
                # img = env.env.env.render(mode="rgb_array")
                # cv2.imshow("render",  cv2.cvtColor(img, cv2.COLOR_BGR2RGB))                    
                # cv2.waitKey(1)

            if verbose:
                print(
                    f"seed: {seed + episode_idx}, "
                    f"reward: {np.sum(rewards)}, len: {env.time_step}"
                )

            scores.append(np.sum(rewards))

            if recorder is not None:
                save_path = recorder.save(f"episode{episode_idx}")
                reward_path = f"{save_path}.reward.pkl"
                print(f"saving reward to {reward_path}")
                pickle.dump(rewards, open(reward_path, "wb"))

    if verbose:
        print(f"num game: {len(scores)}, seed: {seed}, score: {np.mean(scores)}")

    return scores


## Extra added for debugging - RL Sketch
if __name__ == "__main__":
    env_params = dict(
            env_name="ButtonPress",
            robots=["Sawyer"],
            episode_length=100,
            action_repeat=2,
            frame_stack=1,
            obs_stack=1,
            reward_shaping=False,
            rl_image_size=96,
            camera_names=["corner2"],
            rl_camera="corner2",
            device="cuda",
            use_state=True,
        )

    env = PixelMetaWorld(**env_params)
    # env.obj_rand_init(True)

    # Render images
    # cv2.namedWindow("render", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("render", 600, 600)

    action = np.array([0, 0, 0, 0])
    for i in range(10):
        terminal = False
        obs, image_obs = env.reset()

        while not terminal:
            obs, reward, terminal, success, image_obs = env.step(action)


            # Render code
            # img = image_obs["corner2"]
            # img = env.env.env.render(mode="rgb_array")
            # cv2.imshow("render",  cv2.cvtColor(img, cv2.COLOR_BGR2RGB))                    
            # cv2.waitKey(1)
        print(f"Episode {i} done")

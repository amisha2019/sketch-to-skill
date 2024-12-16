import numpy as np
from env.metaworld_wrapper import PixelMetaWorld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_task_oriented_waypoints(start_pos, task_type, num_waypoints):
    # TODO: add other task types to make the generation more diverse
    # TODO: adjust the range of the random values to make the trajectories more realistic
    """Generate waypoints based on the task type with randomness."""
    if task_type == "pick_and_place":
        # Move down, grab, move up, move to target, move down, release
        waypoints = [
            start_pos,
            start_pos + np.array([0, 0, np.random.uniform(-0.15, -0.05)]),  # Move down
            start_pos + np.array([0, 0, np.random.uniform(0.05, 0.15)]),   # Move up
            start_pos + np.array([np.random.uniform(0.15, 0.25), np.random.uniform(0.15, 0.25), 0]), # Move to target
            start_pos + np.array([np.random.uniform(0.15, 0.25), np.random.uniform(0.15, 0.25), np.random.uniform(-0.15, -0.05)]), # Move down
            start_pos + np.array([np.random.uniform(0.15, 0.25), np.random.uniform(0.15, 0.25), np.random.uniform(0.05, 0.15)])  # Move up
        ]
    elif task_type == "button_press":
        # Move to button, press, retract
        waypoints = [
            start_pos,
            start_pos + np.array([np.random.uniform(0.05, 0.15), np.random.uniform(0.05, 0.15), 0]),  # Move to button
            start_pos + np.array([np.random.uniform(0.05, 0.15), np.random.uniform(0.05, 0.15), np.random.uniform(-0.08, -0.02)]),  # Press
            start_pos + np.array([np.random.uniform(0.05, 0.15), np.random.uniform(0.05, 0.15), np.random.uniform(0.05, 0.15)])  # Retract
        ]
    elif task_type == "drawer_open":
        # Open a drawer
        waypoints = [
            start_pos,
            start_pos + np.array([np.random.uniform(0.05, 0.15), 0, 0]),  # Move to drawer handle
            start_pos + np.array([np.random.uniform(0.05, 0.15), 0, np.random.uniform(-0.08, -0.02)]),  # Lower to grasp handle
            start_pos + np.array([np.random.uniform(0.25, 0.35), 0, np.random.uniform(-0.08, -0.02)]),  # Pull drawer open
            start_pos + np.array([np.random.uniform(0.25, 0.35), 0, np.random.uniform(0.05, 0.15)])  # Move up and away
        ]
    elif task_type == "door_open":
        # Open a door
        waypoints = [
            start_pos,
            start_pos + np.array([np.random.uniform(0.15, 0.25), 0, 0]),  # Move to door handle
            start_pos + np.array([np.random.uniform(0.15, 0.25), 0, np.random.uniform(-0.08, -0.02)]),  # Lower to grasp handle
            start_pos + np.array([np.random.uniform(0.15, 0.25), np.random.uniform(-0.35, -0.25), np.random.uniform(-0.08, -0.02)]),  # Swing door open
            start_pos + np.array([np.random.uniform(0.15, 0.25), np.random.uniform(-0.35, -0.25), np.random.uniform(0.05, 0.15)])  # Move up and away
        ]
    elif task_type == "peg_insertion":
        # Insert a peg into a hole
        waypoints = [
            start_pos,
            start_pos + np.array([0, np.random.uniform(0.05, 0.15), 0]),  # Move to peg
            start_pos + np.array([0, np.random.uniform(0.05, 0.15), np.random.uniform(-0.08, -0.02)]),  # Lower to grasp peg
            start_pos + np.array([0, np.random.uniform(0.05, 0.15), np.random.uniform(0.05, 0.15)]),  # Lift peg
            start_pos + np.array([np.random.uniform(0.15, 0.25), 0, np.random.uniform(0.05, 0.15)]),  # Move to hole
            start_pos + np.array([np.random.uniform(0.15, 0.25), 0, np.random.uniform(-0.15, -0.05)]),  # Insert peg
            start_pos + np.array([np.random.uniform(0.15, 0.25), 0, np.random.uniform(0.05, 0.15)])  # Move up
        ]
    else:
        # Default: random waypoints
        waypoints = [start_pos + np.random.uniform(-0.2, 0.2, size=3) for _ in range(num_waypoints)]
    
    return waypoints

def add_obstacle_avoidance(waypoints, obstacles):
    """Modify waypoints to avoid obstacles."""
    new_waypoints = []
    for i in range(len(waypoints) - 1):
        new_waypoints.append(waypoints[i])
        for obstacle in obstacles:
            if np.linalg.norm(waypoints[i] - obstacle) < 0.1 or np.linalg.norm(waypoints[i+1] - obstacle) < 0.1:
                # Add an avoidance point
                avoidance_point = (waypoints[i] + waypoints[i+1]) / 2 + np.array([0, 0, 0.1])
                new_waypoints.append(avoidance_point)
                break
    new_waypoints.append(waypoints[-1])
    return new_waypoints

def generate_trajectories(num_trajectories, points_per_trajectory, env):
    trajectories = []
    types = []
    task_types = ["pick_and_place", "button_press", "drawer_open", "door_open", "peg_insertion", "random"]
    obstacles = [np.array([0.1, 0.1, 0.1]), np.array([-0.1, -0.1, 0.1])]  # Example obstacles
    
    for i in range(num_trajectories):
        # TODO: change this to random start position
        obs, _ = env.reset()
        start_pos = obs["prop"][:3].cpu().numpy()
        
        # Randomly choose a task type
        # task_type = np.random.choice(task_types)
        task_type = task_types[i % len(task_types)]
        
        # Generate task-oriented waypoints
        waypoints = generate_task_oriented_waypoints(start_pos, task_type, 5)
        
        # Add obstacle avoidance
        waypoints = add_obstacle_avoidance(waypoints, obstacles)
        
        # Generate trajectory through waypoints
        trajectory = []
        for i in range(len(waypoints) - 1):
            start = waypoints[0] if i == 0 else trajectory[-1]
            sub_traj = generate_trajectory_servoing(start, waypoints[i+1], points_per_trajectory // len(waypoints), env)
            trajectory.extend(sub_traj)
        
        trajectories.append(np.array(trajectory))
        types.append(task_type)

        # TODO: project the trajectory to the image space
    
    return trajectories, types

def generate_trajectory_servoing(start_pos, end_pos, num_points, env):
    trajectory = []
    # trajectory = [start_pos]
    current_pos = start_pos
    
    for _ in range(num_points - 1):
        action = servoing(current_pos, end_pos)
        obs, _, _, _, _ = env.step(np.append(action, -1))  # -1 for gripper control
        current_pos = obs["prop"][:3].cpu().numpy()
        trajectory.append(current_pos)
        
        if np.linalg.norm(current_pos - end_pos) < 0.01:
            break
    
    return trajectory

def servoing(current_pos, target_pos, Kp=5.0):
    error = target_pos - current_pos
    control_action = Kp * error
    return np.clip(control_action, -1, 1)

def visualize_trajectories(trajectories, types, title="Generated Trajectories using Servoing"):
    num_trajectories = len(trajectories)
    max_cols = 6
    num_rows = (num_trajectories + max_cols - 1) // max_cols
    num_cols = min(num_trajectories, max_cols)
    
    fig = plt.figure(figsize=(4*num_cols, 4*num_rows))
    cmap = plt.get_cmap('viridis')
    
    for i, (trajectory, task_type) in enumerate(zip(trajectories, types)):
        ax = fig.add_subplot(num_rows, num_cols, i+1, projection='3d')
        
        # Plot starting point with a green dot
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', s=50, label='Start')
        # Plot ending point with a red dot
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], color='red', s=50, label='End')
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='blue')
        
        # Scatter plot with color gradient
        scatter = ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                             c=np.arange(len(trajectory)), cmap=cmap, s=10, alpha=0.5)
        
        # # Add a colorbar to show the time sequence
        # cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        # cbar.set_label('Time Sequence')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Trajectory {i+1} ({task_type})')
        
        # Add a legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize='small')
    
    fig.suptitle(title, fontsize=16)
    
    # Adjust the layout to prevent overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig('trajectories.png', bbox_inches='tight', dpi=300)

def main():
    env_params = dict(
        env_name="ButtonPress",
        robots=["Sawyer"],
        episode_length=100,
        action_repeat=2,
        frame_stack=1,
        obs_stack=1,
        reward_shaping=False,
        rl_image_size=96,
        camera_names=["corner2", "corner"],
        rl_camera="corner2",
        device="cuda",
        use_state=True,
    )
    env = PixelMetaWorld(**env_params)
    
    num_trajectories = 18
    points_per_trajectory = 100
    
    trajectories, types = generate_trajectories(num_trajectories, points_per_trajectory, env)
    
    # Visualize all trajectories
    visualize_trajectories(trajectories, types)
    
    # Here you can add code to save the trajectories or use them in your environment

if __name__ == "__main__":
    main()

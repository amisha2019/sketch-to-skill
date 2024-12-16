import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy import interpolate

def scale_trajectory(trajectory, min_size, max_size):
    current_size = np.ptp(trajectory, axis=0).max()
    if current_size < min_size:
        scale_factor = min_size / current_size
    elif current_size > max_size:
        scale_factor = max_size / current_size
    else:
        return trajectory
    
    center = np.mean(trajectory, axis=0)
    return (trajectory - center) * scale_factor

def add_displacement(trajectory, max_displacement=5.0):
    dx = np.random.uniform(-max_displacement, max_displacement)
    dy = np.random.uniform(-max_displacement, max_displacement)
    return trajectory + np.array([dx, dy])

def generate_sine_trajectory(num_points=100, amplitude=1.0, frequency=1.0, noise_level=0.05, rotation=None):
    st = -np.random.uniform(0, 2*np.pi)
    t = np.linspace(st, 4*np.pi, num_points)
    x = t
    y = amplitude * np.sin(frequency * t)
    
    trajectory = np.column_stack([x, y])
    
    if rotation is not None:
        rotation_matrix = np.array([
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)]
        ])
        trajectory = np.dot(trajectory, rotation_matrix)
    
    trajectory += np.random.normal(0, noise_level, trajectory.shape)
    return trajectory

def generate_spiral_trajectory(num_points=100, a=1, b=0.2, noise_level=0.05):
    t = np.linspace(0, a*np.pi, num_points)
    x = (a + b*t) * np.cos(t)
    y = (a + b*t) * np.sin(t)
    
    trajectory = np.column_stack([x, y])
    trajectory += np.random.normal(0, noise_level, trajectory.shape)
    return trajectory

def generate_logarithmic_spiral_trajectory(num_points=100, a=1, b=0.1, t_range=(0, 4*np.pi), noise_level=0):
    t = np.linspace(t_range[0], t_range[1], num_points)
    r = a * np.exp(b * t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    
    trajectory = np.column_stack([x, y])
    trajectory += np.random.normal(0, noise_level, trajectory.shape)
    return trajectory

def generate_cubic_trajectory(num_points=100, a=1, b=1, c=1, d=0, x_range=(-1, 1), noise_level=0):
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = a*x**3 + b*x**2 + c*x + d
    
    trajectory = np.column_stack([x, y])
    trajectory += np.random.normal(0, noise_level, trajectory.shape)
    return trajectory

def trajectory_to_sketch(trajectory, traj_min, traj_max, image_size=224, line_width=3, padding=20):
    # Normalize the trajectory to fit in the image with padding
    normalized_traj = (trajectory - traj_min) / (traj_max - traj_min) * (image_size - 2*padding - 1) + padding
    
    # Create a blank image
    image = Image.new('L', (image_size, image_size), color=255)
    draw = ImageDraw.Draw(image)
    
    # Draw the trajectory
    for i in range(len(normalized_traj) - 1):
        draw.line([tuple(normalized_traj[i]), tuple(normalized_traj[i+1])], 
                  fill=0, width=line_width)
    return np.array(image)

def fit_bspline_trajectory(points, num_control_points=50, degree=3):
    u = np.linspace(0, 1, len(points))
    n_knots = num_control_points + degree + 1
    knots = np.clip(np.linspace(0, 1, n_knots), 0, 1)
    knots[:degree+1] = 0
    knots[-degree-1:] = 1
    # breakpoint()
    tck, u = interpolate.splprep([points[:, 0], points[:, 1]], u=u, s=0, k=degree, t=knots, task=-1)
    control_points = np.column_stack(tck[1])
    
    # Generate trajectory
    trajectories = predict_trajectory_spline(tck, num_points=200)
    
    return control_points.flatten(), trajectories

def predict_trajectory_spline(tck, num_points=200):
    u_new = np.linspace(0, 1, num_points)
    return np.array(interpolate.splev(u_new, tck)).T

def generate_dataset(num_samples=1000, img_size=224, num_control_points=40, trajectory_types=['sine', 'spiral', 'cubic', 'logarithmic_spiral'], max_displacement=2.0, min_size=2.0, max_size=8.0, fit_bspline=False):
    sketches = []
    trajectories = []
    if fit_bspline:
        control_points = []
        fitted_trajectories = []

    for i in range(num_samples):
        # traj_type = np.random.choice(trajectory_types)
        traj_type = trajectory_types[i % len(trajectory_types)]
        # traj_type = trajectory_types[3]
        
        if traj_type == 'sine':
            amplitude = np.random.uniform(0.5, 4.0)
            frequency = np.random.uniform(0.5, 2.0)
            rotation = np.random.uniform(0, 2*np.pi)
            traj = generate_sine_trajectory(amplitude=amplitude, frequency=frequency, rotation=rotation)
        elif traj_type == 'spiral':
            a = np.random.uniform(1.0, 3.0)
            b = np.random.uniform(0.6, 0.8)
            traj = generate_spiral_trajectory(a=a, b=b)
        elif traj_type == 'logarithmic_spiral':
            a = np.random.uniform(0.5, 1)
            b = np.random.uniform(0.1, 0.4)
            t_range = (np.random.uniform(0.5*np.pi, 1.2*np.pi), np.random.uniform(2*np.pi, 3*np.pi))
            traj = generate_logarithmic_spiral_trajectory(a=a, b=b, t_range=t_range)
        elif traj_type == 'cubic':
            a, b, c, d = np.random.uniform(-1, 1, 4)
            x_range = (-np.random.uniform(2, 3), np.random.uniform(2, 3))
            traj = generate_cubic_trajectory(a=a, b=b, c=c, d=d, x_range=x_range)
        
        # Scale the trajectory
        traj = scale_trajectory(traj, min_size, max_size)
        
        # Add random displacement
        traj = add_displacement(traj, max_displacement)
        traj = traj / (max_size + max_displacement)
        trajectories.append(traj)

        if fit_bspline:
            coeffs, fitted_traj = fit_bspline_trajectory(traj, num_control_points=num_control_points, degree=3)
            control_points.append(coeffs)
            fitted_trajectories.append(fitted_traj)
    
    trajectories = np.array(trajectories)
    traj_min = trajectories.min()
    traj_max = trajectories.max()

    for traj in trajectories:
        sketch = trajectory_to_sketch(traj, traj_min, traj_max, image_size=img_size)
        sketches.append(sketch)
    
    if fit_bspline:
        return np.array(sketches), np.array(trajectories), np.array(control_points), np.array(fitted_trajectories)
    return np.array(sketches), np.array(trajectories)

def get_dataloader(batch_size=32, num_samples=1000, img_size=224, num_control_points=40, max_displacement=2.0):
    sketches, trajectories, params, fitted_trajectories = generate_dataset(num_samples=num_samples, img_size=img_size, num_control_points=num_control_points, max_displacement=max_displacement, min_size=2.0, max_size=8.0, fit_bspline=True)
    sketches_tensor = torch.from_numpy(sketches).float().unsqueeze(1)
    sketches_tensor = 1.0 - (sketches_tensor - sketches_tensor.min()) / (sketches_tensor.max() - sketches_tensor.min())
    trajectories_tensor = torch.from_numpy(trajectories).float()
    params_tensor = torch.from_numpy(params).float()
    fitted_trajectories_tensor = torch.from_numpy(fitted_trajectories).float()
    # dataset = TensorDataset(sketches_tensor, trajectories_tensor)
    dataset = TensorDataset(sketches_tensor, trajectories_tensor, params_tensor, fitted_trajectories_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def visualize_samples(sketches, trajectories, fitted_trajectories=None, num_samples=5):
    if fitted_trajectories is not None:
        fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 9))
    else:
        fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    
    for i in range(num_samples):
        axes[0, i].imshow(np.flipud(sketches[i].squeeze()), cmap='gray')
        axes[0, i].set_title(f"Sample {i+1}: Sketch")
        # axes[0, i].axis('off')
        axes[1, i].set_aspect('equal', 'box')
        
        axes[1, i].plot(trajectories[i][:, 0], trajectories[i][:, 1])
        axes[1, i].set_title(f"Sample {i+1}: Trajectory")
        axes[1, i].set_aspect('equal', 'box')
        
        # Set consistent axis limits
        axes[1, i].set_xlim(-1, 1)
        axes[1, i].set_ylim(-1, 1)

        if fitted_trajectories is not None:
            axes[2, i].set_aspect('equal', 'box')
            axes[2, i].plot(fitted_trajectories[i][:, 0], fitted_trajectories[i][:, 1])
            axes[2, i].set_title(f"Sample {i+1}: Fitted Trajectory")
            axes[2, i].set_aspect('equal', 'box')
            axes[2, i].set_xlim(-1, 1)
            axes[2, i].set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig('samples.png')


if __name__ == "__main__":
    # generate some data
    # sketches, trajectories = generate_dataset(num_samples=8, max_displacement=2.0, min_size=2.0, max_size=8.0)
    sketches, trajectories, control_points, fitted_trajectories = generate_dataset(num_samples=8, img_size=64, num_control_points=10, max_displacement=2.0, min_size=2.0, max_size=8.0, fit_bspline=True)

    # visualize some samples
    visualize_samples(sketches, trajectories, fitted_trajectories, 8)

    print(f"Dataset generated with {len(sketches)} samples.")
    print(f"Sketch shape: {sketches.shape}")
    print(f"Trajectory shape: {trajectories.shape}")
    print(f"Trajectory range: {trajectories.min()} - {trajectories.max()}")
    print(f"Control points range: {control_points.min()} - {control_points.max()}")
    print(f"Control points shape: {control_points.shape}")

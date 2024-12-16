import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def generate_bspline(control_points, knots, degree=3, num_points=1000):
    tck = (knots, control_points.T, degree)
    t = np.linspace(knots[degree], knots[-degree-1], num_points)
    return np.array(interpolate.splev(t, tck)).T

def add_scaled_noise(params, noise_level, param_range):
    scaled_noise = noise_level * param_range
    return np.clip(params + np.random.normal(0, scaled_noise, params.shape), 0, param_range)

def add_noise_to_knots(knots, noise_level, degree):
    noisy_knots = add_scaled_noise(knots, noise_level, 1.0)
    noisy_knots[:degree+1] = 0
    noisy_knots[-degree-1:] = 1
    return np.sort(noisy_knots)

def get_noisy_curve(tck, noise_level, num_points=1000):
    knots, control_points, degree = tck
    noisy_knots = add_noise_to_knots(knots, noise_level, degree)
    control_points = np.array(control_points).T
    control_point_range = np.max(control_points) - np.min(control_points)
    noisy_control_points = add_scaled_noise(control_points, noise_level, control_point_range)
    # breakpoint()
    noisy_curve = generate_bspline(noisy_control_points, noisy_knots, degree=degree, num_points=num_points)
    return noisy_curve


if __name__ == "__main__":
    # Generate a sample B-spline
    num_control_points = 7
    degree = 3
    control_points = np.random.rand(num_control_points, 2) * 10  # Scale to [0, 10] range
    knots = np.linspace(0, 1, num_control_points + degree + 1)

    # Ensure boundary knots are repeated
    knots[:degree+1] = 0
    knots[-degree-1:] = 1

    original_curve = generate_bspline(control_points, knots)

    # Calculate range for control points
    control_point_range = np.max(control_points) - np.min(control_points)

    # Test different noise levels
    noise_levels = [0, 0.01, 0.05, 0.1]
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    axs = axs.ravel()

    for i, noise in enumerate(noise_levels):
        noisy_knots = add_noise_to_knots(knots, noise)
        noisy_control_points = add_scaled_noise(control_points, noise, control_point_range)
        noisy_curve = generate_bspline(noisy_control_points, noisy_knots)
        
        axs[i].plot(original_curve[:, 0], original_curve[:, 1], 'b-', label='Original')
        axs[i].plot(noisy_curve[:, 0], noisy_curve[:, 1], 'r-', label='Noisy')
        axs[i].scatter(control_points[:, 0], control_points[:, 1], c='g', label='Original Control Points')
        axs[i].scatter(noisy_control_points[:, 0], noisy_control_points[:, 1], c='m', label='Noisy Control Points')
        
        # Plot vertical lines for knots (scaled to control point range for visibility)
        for k in knots[degree:-degree]:
            axs[i].axvline(x=k * control_point_range, color='b', linestyle='--', alpha=0.3)
        for k in noisy_knots[degree:-degree]:
            axs[i].axvline(x=k * control_point_range, color='r', linestyle='--', alpha=0.3)
        
        axs[i].set_title(f'Noise Level: {noise}')
        axs[i].legend()
        axs[i].set_xlim(0, control_point_range)
        axs[i].set_ylim(0, control_point_range)

    plt.tight_layout()
    plt.show()

    # Print parameter differences
    print("Original knots:", knots)
    print("Noisy knots (highest noise level):", noisy_knots)
    print("\nOriginal control points (first few):", control_points[:3])
    print("Noisy control points (first few):", noisy_control_points[:3])
    print(f"\nControl point range: {control_point_range}")
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def interpolate_volume(input_volume, new_shape):
    """
    Interpolates a 3D volume to a new shape.

    Args:
    input_volume: A 3D numpy array of shape (depth, height, width) representing the volume.
    new_shape: A tuple representing the desired output shape. One dimension should be smaller than the original.

    Returns:
    A 3D numpy array representing the interpolated volume.
    """

    # Get original shape
    original_shape = input_volume.shape

    # Create grid of original coordinates
    x = np.arange(original_shape[0])
    y = np.arange(original_shape[1])
    z = np.arange(original_shape[2])
    grid = (x, y, z)

    # Create interpolator function
    interpolator = RegularGridInterpolator(grid, input_volume, method='linear')

    # Create grid of new coordinates
    new_x = np.linspace(0, original_shape[0] - 1, new_shape[0])
    new_y = np.linspace(0, original_shape[1] - 1, new_shape[1])
    new_z = np.linspace(0, original_shape[2] - 1, new_shape[2])
    new_grid = np.meshgrid(new_x, new_y, new_z, indexing='ij')
    new_points = np.array([new_grid[0].flatten(), new_grid[1].flatten(), new_grid[2].flatten()]).T

    # Interpolate the volume
    interpolated_volume = interpolator(new_points).reshape(new_shape)

    # Convert back to the original data type.
    min_val, max_val = interpolated_volume.min(), interpolated_volume.max()
    if min_val == max_val:
        interpolated_volume = np.full(interpolated_volume.shape, min_val, dtype=input_volume.dtype)
    else:
        scaled_image = (interpolated_volume - min_val) / (max_val - min_val) * (np.iinfo(input_volume.dtype).max - 1)
        interpolated_volume = scaled_image.astype(input_volume.dtype)

    return interpolated_volume

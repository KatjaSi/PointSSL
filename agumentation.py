import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data import uniform_sample_points


def split_pc_in_half(pc, axis, num_points):
    pc_axis_projection = np.array(pc[:,axis])
    median =  np.median(pc_axis_projection)
    pc_upper = pc[pc_axis_projection  > median]
    pc_down = pc[pc_axis_projection < median]
    pc_upper = uniform_sample_points(pc_upper, num_points)
    pc_down = uniform_sample_points(pc_down, num_points)
    # back to take the same space volume as before
    # scale
    pc_max = np.max(pc[:,axis])
    pc_min = np.min(pc[:,axis])
    
    __scale_pc__(pc_upper, axis, pc_min, pc_max)
    __scale_pc__(pc_down, axis, pc_min, pc_max)
    return pc_upper, pc_down

def __scale_pc__(pc, axis, new_min, new_max):
    pc_max = np.max(pc[:,axis])
    pc_min = np.min(pc[:,axis])
    scale_factor = (new_max-new_min)/(pc_max-pc_min)
    pc[:,axis] = pc[:,axis]*scale_factor #+ new_min + new_max-np.min(pc[:,axis])-np.max(pc[:,axis])
    pc[:,axis] = pc[:,axis]+new_min-np.min(pc[:,axis])


def random_volume_crop_pc(pc, crop_percentage):
    """
    Randomly removes a percentage of volume from a point cloud.

    Parameters:
        pc (numpy array): The point cloud.
        crop_percentage (float): The percentage of volume to crop (0.0 to 1.0).

    Returns:
        numpy array: The point cloud with the cropped region removed.
    """
    # Get the bounding box of the point cloud
    min_coords = np.min(pc, axis=0)
    max_coords = np.max(pc, axis=0)
    bounding_box_dimensions = max_coords - min_coords
   # Calculate the volume to crop
    crop_lengths = bounding_box_dimensions*crop_percentage**(1/3)
    
    # Generate random coordinates for the minimum corner of the cuboid
    crop_min = min_coords + np.random.rand(3) * (bounding_box_dimensions - crop_lengths)
    crop_max = crop_min + crop_lengths

    # Keep the points outside the cuboid region
    cropped_pc = pc[(np.any(pc < crop_min, axis=1)) | (np.any(pc > crop_max, axis=1))]
    cropped_pc = uniform_sample_points(cropped_pc, pc.shape[0])
    return cropped_pc

def rotate_pc(pc, angles):
    """
    Rotate a 3D point cloud around specified axes.

    Parameters:
    - point_cloud: NumPy array of shape (N, 3) representing 3D points.
    - angles: Tuple of three values (angle_x, angle_y, angle_z) specifying
              the rotation angles around the x, y, and z axes in radians.

    Returns:
    - rotated_point_cloud: NumPy array of shape (N, 3) representing the rotated 3D points.
    """
    rotation_matrix = np.eye(3)

    if angles[0] != 0:  # Rotate around x-axis
        rotation_matrix = np.dot(rotation_matrix, np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]]))
    if angles[1] != 0:  # Rotate around y-axis
        rotation_matrix = np.dot(rotation_matrix, np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]]))
    if angles[2] != 0:  # Rotate around z-axis
        rotation_matrix = np.dot(rotation_matrix, np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]]))

    rotated_pc = np.dot(pc, rotation_matrix)
    return rotated_pc
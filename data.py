import os
import glob
import copy
import h5py
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import random


def load_data(partition, data_dir):
    all_data = []
    all_label = []
    # Update the glob pattern to search for hdf5 files within the provided path
    for h5_name in glob.glob(os.path.join(data_dir,  f'*{partition}*.h5')):
        with h5py.File(h5_name, 'r') as f:  # Use context manager to ensure files are properly closed
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            all_data.append(data)
            all_label.append(label)
    # Concatenate all data and labels
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_point_cloud_data_from_npy(data_dir, num_files, num_points):
    all_data = []
    # Search for .npy files in the provided path
    i = 0
    for npy_name in glob.glob(os.path.join(data_dir, '*.npy')):
        # Load the point cloud data from each .npy file
        data = np.load(npy_name).astype('float32')[:,:num_points,:]
        all_data.append(data)
        i += 1
        if i == num_files:
            break
    # Concatenate all data
    all_data = np.concatenate(all_data, axis=0)
    return all_data

def random_volume_crop_pc(pc, crop_percentage):
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


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx) > 0:
        # Select random indices from the point cloud to use as replacements
        replace_idx = np.random.choice(len(pc), size=len(drop_idx), replace=False)
        # Replace dropped points with points from random locations within the cloud
        pc[drop_idx, :] = pc[replace_idx, :]
    return pc

def translate_pointcloud(pointcloud):
    #xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    #xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    xyz1 = np.random.uniform(low=.85, high=1.15, size=[3]) # stretching 
    xyz2 = np.random.uniform(low=-0.1, high=0.1, size=[3]) # translation
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02): 
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

def uniform_sample_points(pointcloud, num_points):
    if len(pointcloud) >= num_points:
        sampled_indices = np.random.choice(len(pointcloud), num_points, replace=False)
    else:
        sampled_indices = np.random.choice(len(pointcloud), num_points, replace=True)
    return pointcloud[sampled_indices]

def add_gaussian_noise(pointcloud, sigma=0.05):
    """Add Gaussian noise to the point cloud"""
    noise = np.random.normal(0, sigma, pointcloud.shape)
    pointcloud_noisy = pointcloud + noise
    return pointcloud_noisy


# additional augmentations to make the downstram task harder
def rotate_pointcloud(pointcloud):
    angle_x = np.random.uniform(-np.pi/2, np.pi/2)  
    angle_y = np.random.uniform(-np.pi/2, np.pi/2) 
    angle_z = np.random.uniform(-np.pi/2, np.pi/2) 

    # Rotation matrix around X-axis
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)]])
    
    # Rotation matrix around Y-axis
    R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])
    
    # Rotation matrix around Z-axis
    R_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix: R = Rz * Ry * Rx
    R_combined = np.dot(R_z, np.dot(R_y, R_x))
    
    # Apply the combined rotation matrix to the point cloud
    pointcloud_rotated = np.dot(pointcloud, R_combined.T)
    
    return pointcloud_rotated


class ModelNet(Dataset):
    def __init__(self, data, labels, num_points=1024, set_type="train"):
        self.data = data
        self.labels = labels
        self.set_type = set_type
        self.num_points = num_points

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx): # TODO: check how the data should be agumented
        x = self.data[idx][:self.num_points]
        y = self.labels[idx]
        if self.set_type == 'train':
           # x = jitter_pointcloud(x, sigma=0.01, clip=0.02)
            #x = random_point_dropout(x) 
            x = translate_pointcloud(x)
            np.random.shuffle(x)
        return x, y    

class ModelNetAugmented(Dataset):
    def __init__(self, data, labels, num_points=1024):
        self.data = data
        self.labels = labels
        self.num_points = num_points

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx): # TODO: check how the data should be agumented
        x = self.data[idx][:self.num_points]
        y = self.labels[idx]

        x = jitter_pointcloud(x)
        x = random_point_dropout(x) 
        x = translate_pointcloud(x)
        x = rotate_pointcloud(x) # most difficult augmentation
        x = add_gaussian_noise(x)
        np.random.shuffle(x)
        return x, y  


class ModelNetForSSL(Dataset):
    def __init__(self, data, num_points):
        self.data = data
        self.num_points = num_points

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx): 
        x = self.data[idx][:self.num_points]
        x_prime = copy.deepcopy(x)


        x = jitter_pointcloud(x)
        x = random_point_dropout(x) 
        x = translate_pointcloud(x)
        x = add_gaussian_noise(x)
        np.random.shuffle(x)

        x_prime = random_volume_crop_pc(x_prime, crop_percentage=random.uniform(0.2, 0.8)) #0.2,0.8
        x_prime = jitter_pointcloud(x_prime)
        x_prime = random_point_dropout(x_prime) 
        x_prime = translate_pointcloud(x_prime)
        x_prime = rotate_pointcloud(x_prime)
        x_prime = add_gaussian_noise(x_prime)
        np.random.shuffle(x_prime)

        return x_prime, x

if __name__ == '__main__':
    #train_points, train_labels = load_data2("train", folder='shapenetcorev2_hdf5_2048') #'modelnet10_hdf5_2048'
    x, y, z = train_points[10].T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='b', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    plt.savefig("output.png")


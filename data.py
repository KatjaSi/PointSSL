import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def load_data(partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_data2(partition, folder):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, folder, '%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
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
            x = random_point_dropout(x) 
            x = translate_pointcloud(x)
            np.random.shuffle(x)
            # TODO: anisotropic scaling?
        return x, y    


if __name__ == '__main__':
    train_points, train_labels = load_data2("train", folder='shapenetcorev2_hdf5_2048') #'modelnet10_hdf5_2048'
    x, y, z = train_points[10].T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='b', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    plt.savefig("output.png")
    print(len(train_points))

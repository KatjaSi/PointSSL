import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from data import load_data, ModelNet
import numpy as np

import torch.nn.functional as F
import os

from geomloss import SamplesLoss

if __name__ == '__main__':
    train_points, train_labels = load_data("train", "data/modelnet40_ply_hdf5_2048")
    #test_points, test_labels = load_data("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    train_points_tensor = [torch.tensor(points).to(device).float() for points in train_points]

    sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)

    # Now you can compute the Sinkhorn loss between the first two point clouds in your dataset
    # Make sure that the tensors are 2D (i.e., of shape [N, D] where N is the number of points and D is the dimensionality)
    # You may need to adjust dimensions with .unsqueeze(0) if your tensors are 1D
    loss_value = sinkhorn_loss(train_points_tensor[0], train_points_tensor[0]).item()
    print(loss_value)
   


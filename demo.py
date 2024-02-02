import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from data import load_data, ModelNet
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import torch.nn.functional as F
import os
from sklearn.neighbors import KNeighborsClassifier
from pointSSL import EncoderModule
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class SimpleModel(nn.Module):

    def __init__(self, in_channels=3):
        super(SimpleModel, self).__init__()
        self.embedding_module = EmbeddingModule(in_channels=in_channels, out_channels=256)



    def forward(self,x):
        batch_size, _, _ = x.size()
        x = self.embedding_module(x)


        # collapses the spatial dimension (num_points) to 1, resulting in a tensor of shape (batch_size, num_features, 1)
        x_max_pool = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # this is CLS token representerer hele pc
        x_avg_pool = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        
        x = torch.cat((x_max_pool, x_avg_pool), dim=1)


        return x



class EmbeddingModule(nn.Module):

    def __init__(self, in_channels=3, out_channels=256):
        super(EmbeddingModule, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=out_channels, kernel_size=1, bias=False)

        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x,negative_slope=0.2)

        # Apply second convolution and batch normalization
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x,negative_slope=0.2)

        return x


if __name__ == '__main__':
    train_points, train_labels = load_data("train")
    test_points, test_labels = load_data("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
   # pct = POINT_SSL()
    pct = SimpleModel() 
    pct.eval()


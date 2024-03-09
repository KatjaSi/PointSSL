import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleProjectionHead(nn.Module):
    def __init__(self, input_dim, projected_dim):
        super(SimpleProjectionHead, self).__init__()
        self.linear = nn.Linear(input_dim, projected_dim)
    
    def forward(self, x):
        return self.linear(x)

class MLPProjectionHead2(nn.Module):
    def __init__(self, input_dim, hidden_dim, projected_dim):
        super(MLPProjectionHead2, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(hidden_dim, projected_dim)
        )
    
    def forward(self, x):
        x = self.network(x)
        # Normalize the output to have unit length
        x = F.normalize(x, p=2, dim=1)  # p=2 denotes L2 norm, dim=1 normalizes each vector in the batch
        return x


class ClassifierHead(nn.Module):
    def __init__(self, input_dim=256, output_channels=40):
        super(ClassifierHead, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(input_dim, input_dim//2) # global rep
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(input_dim//2, output_channels)
    
    def forward(self, x):
        x = self.bn1(x) #TODO: do i need this?
        x = F.leaky_relu(x, negative_slope=0.2) 
        x = self.dp1(x) 

        x = self.linear2(x)
        x = F.leaky_relu(x, negative_slope=0.2) 
       # x = self.dp2(x)

        x = self.linear3(x)
        return x

class PointCloudDecoder(nn.Module):
    def __init__(self):
        super(PointCloudDecoder, self).__init__()
        self.conv1 = nn.Conv1d(128+256, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(128, 32, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(32, 3, kernel_size=1, bias=False)

    def forward(self, x):
        # Apply the series of convolutions and activations
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x
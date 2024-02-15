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
        self.linear2 = nn.Linear(input_dim, 128) # global rep
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(128, output_channels)
    
    def forward(self, x):
        x = self.bn1(x) #TODO: do i need this?
        x = F.leaky_relu(x, negative_slope=0.2) 
        x = self.dp1(x) 

        x = self.linear2(x)
        x = F.leaky_relu(x, negative_slope=0.2) 
       # x = self.dp2(x)

        x = self.linear3(x)
        return x


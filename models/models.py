import torch
import torch.nn as nn
import torch.nn.functional as F


'''
PointNet AutoEncoder
Learning Representations and Generative Models For 3D Point Clouds
https://arxiv.org/abs/1707.02392
'''

class PointCloudAEEncoder(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAEEncoder, self).__init__()
        
        self.latent_size = latent_size
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(latent_size)

        # add attn
        self.encoders = nn.ModuleList([EncoderModule(num_heads=4, in_features=128) for _ in range(0)])

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        for encoder in self.encoders:
            x = encoder(x)

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x

class PointCloudAEDecoder(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAEDecoder, self).__init__()
        
        self.point_size = point_size
        self.dec1 = nn.Linear(latent_size, 256)
        self.dec2 = nn.Linear(256, 256)
        self.dec3 = nn.Linear(256, point_size * 3)

    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)


class PointCloudAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointCloudAE, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size,256)
        self.dec2 = nn.Linear(256,256)
        self.dec3 = nn.Linear(256,self.point_size*3)

    def encoder(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Linear_Decoder(nn.Module):
    def __init__(self, point_size, latent_size):
        super(Linear_Decoder, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        
        self.dec1 = nn.Linear(self.latent_size,256)
        self.dec2 = nn.Linear(256,256)
        self.dec3 = nn.Linear(256,self.point_size*3)
    
    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)



class EncoderModule(nn.Module):

    def __init__(self, in_features=256, num_heads=8):
        super(EncoderModule, self).__init__()
        self.q_conv = nn.Conv1d(in_features, in_features, 1, bias=False) # Q
        self.k_conv = nn.Conv1d(in_features, in_features, 1, bias=False) # K, queries and keys of same dimentionality

        self.v_conv = nn.Conv1d(in_features, in_features, 1) 
        self.mh_sa = nn.MultiheadAttention(num_heads=num_heads, embed_dim=in_features, batch_first=True)
        self.bn_after_sa = nn.BatchNorm1d(in_features)

        self.linear_after_sa =  nn.Linear(in_features,in_features) 
        self.bn_after_linear_sa = nn.BatchNorm1d(in_features) 
        
    def forward(self, x):
        x_attention, _ = self.mh_sa(x.permute(0,2,1), x.permute(0,2,1), x.permute(0,2,1))
        x_attention = x_attention.permute(0, 2, 1)
        x = x + x_attention
        x = self.bn_after_sa(x)
        x = x.permute(0, 2, 1)
        x_linear = self.linear_after_sa(x)
        x = x + x_linear
        x = x.permute(0, 2, 1)
        x = self.bn_after_linear_sa(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class POINT_SSL(nn.Module):

    def __init__(self, in_channels=3, output_channels=40):
        super(POINT_SSL, self).__init__()

        self.embedding_module = EmbeddingModule(in_channels=in_channels, out_channels=256)

        self.encoder1 = EncoderModule(256, num_heads=4)
        self.encoder2 = EncoderModule(256, num_heads=4)
        self.encoder3 = EncoderModule(256, num_heads=4)
        self.encoder4 = EncoderModule(256, num_heads=4)

        self.conv_fuse = nn.Sequential(nn.Conv1d(256*4, 256*4, kernel_size=1, bias=False), #*4 to conctat all ecnoders
                                   nn.BatchNorm1d(256*4), 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(256*4+256*4, 256, bias=False)    
        self.bn6 = nn.BatchNorm1d(256) 
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(256, 128) # global rep

        # projection layers      
       # self.linear3 = nn.Linear(128, 64) # This representation will go into loss function
        self.projection = ProjectionHead(128) # output is 32

        # added for fine-tuning, do not use linear3 when fine-tuning
        self.linear4 = nn.Linear(128, output_channels)

    def forward(self, x_prime, x=None, downstream=False):

        if x is not None:
            # Pretraining mode: compute features for both x_prime and x
            x_prime_rep, x_prime_projection = self.forward_single(x_prime,downstream)
            x_rep, x_projection = self.forward_single(x, downstream)
            return x_prime_rep, x_rep, x_prime_projection, x_projection
        else:
            # Fine-tuning mode: compute features for x_prime only
            x = self.forward_single(x_prime, downstream=downstream) 
            return x #, x_prime_projection
        

    def forward_single(self,x, downstream):
        batch_size, _, _ = x.size()

        x = self.embedding_module(x)

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)

        # collapses the spatial dimension (num_points) to 1, resulting in a tensor of shape (batch_size, num_features, 1)
        x_max_pool = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # this is CLS token representerer hele pc
        x_avg_pool = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        #x = x.view(batch_size, -1)
        x = torch.cat((x_max_pool, x_avg_pool), dim=1)

        x = self.linear1(x)
        x = self.bn6(x) 
        x = F.leaky_relu(x, negative_slope=0.2) 
        x = self.dp1(x) 
        
        x_rep = self.linear2(x)
        x = F.leaky_relu(x_rep, negative_slope=0.2) # F.relu(x)
        #x = self.dp2(x) 
        if not downstream:
           # x = self.linear3(x)
            x = self.projection(x)
            return x_rep, x # global rep, projection
        else: # if downstream
            x = self.linear4(x)
            return x
    


class EmbeddingModule(nn.Module):
    """
    A module for projecting a 3D point cloud into an embedding space.

    This module transforms the input point cloud into a higher-dimensional embedding space.

     Args:
        in_channels (int, optional): The number of input channels representing the dimensions of the input point cloud (default: 3).

    Input:
        x (torch.Tensor): The input point cloud tensor of shape (batch_size, num_points, 3).

    Output:
        torch.Tensor: The output tensor after projection, of shape (batch_size, num_points, embedding_dim).
    """

    def __init__(self, in_channels=3, out_channels=256):
        """
        Initializes an EmbeddingModule instance.

        This module consists of convolutional layers and batch normalization operations
        to project a 3D point cloud into an embedding space.
        """
        super(EmbeddingModule, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, bias=False)

        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        """
        Forward pass through the EmbeddingModule.

        Applies a series of convolutional layers and batch normalization operations to
        project the input point cloud into an embedding space.

        Args:
            x (torch.Tensor): The input point cloud tensor of shape (batch_size, num_points, 3).

        Returns:
            torch.Tensor: The output tensor after projection, of shape (batch_size, num_points, embedding_dim).
        """
        # Apply first convolution and batch normalization
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x,negative_slope=0.2)

        # Apply second convolution and batch normalization
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x,negative_slope=0.2)

        return x

class EncoderModule(nn.Module):
    """
    A module implementing an encoder block.

    This module incorporates a Multi-Head Self Attention mechanism followed by linear transformations
    and batch normalization to process input data.

    Args:
        in_features (int, optional): The dimensionality of each input feature (default: 256).
        num_heads (int, optional): The number of attention heads to use in the Multi-Head Self Attention mechanism (default: 8).
    """
    def __init__(self, in_features=256, num_heads=8):
        super(EncoderModule, self).__init__()
        
        self.mh_sa = MultiHeadSelfAttention(in_features=in_features, head_dim=int(in_features/num_heads), num_heads=num_heads) # in_features is dim of each point
        self.bn_after_sa = nn.BatchNorm1d(in_features)

        # Linear layer is to learn  interactions within each embedding independently of other embeddings
        self.linear_after_sa =  nn.Linear(in_features,in_features) 
        self.bn_after_linear_sa = nn.BatchNorm1d(in_features) 
        
    def forward(self, x):
        x_attention = self.mh_sa(x)
        x = x + x_attention
        x = self.bn_after_sa(x)
        x = x.permute(0, 2, 1)
        x_linear = self.linear_after_sa(x)
        x = x + x_linear
        x = x.permute(0, 2, 1)
        x = self.bn_after_linear_sa(x)
        
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_features, head_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.q_conv = nn.Conv1d(in_features, in_features, 1, bias=False) # Q
        self.k_conv = nn.Conv1d(in_features, in_features, 1, bias=False) # K, queries and keys of same dimentionality
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(in_features, in_features, 1)
        assert  num_heads*head_dim == in_features
        
        self.out_linear = nn.Linear(head_dim * num_heads, in_features)
        
    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim)
    
    def forward(self, x):
        # Split heads
        q = self.q_conv(x).permute(0, 2, 1)
        k = self.k_conv(x).permute(0, 2, 1)
        v = self.v_conv(x).permute(0, 2, 1)
        
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        x = flash_attn_func(q, k, v) 
        x = torch.cat(tuple(x.unbind(3)), dim=2)
        x = x.permute(0,2,1)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_features):
        super(ProjectionHead, self).__init__()
        self.linear1 = nn.Linear(in_features, 64) 
        self.linear2 = nn.Linear(64, 32) # This representation will go into loss function# This representation will go into loss function

    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, negative_slope=0.2) 
        x = self.linear2(x)
        return x

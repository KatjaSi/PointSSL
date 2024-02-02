import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class PCT(nn.Module):

    def __init__(self, in_channels=3, out_channels=256):
        super(PCT, self).__init__()

        self.embedding_module = EmbeddingModule(in_channels=in_channels, out_channels=256)

        self.encoder1 = EncoderModule(256, num_heads=4)
        self.encoder2 = EncoderModule(256, num_heads=4)
        self.encoder3 = EncoderModule(256, num_heads=4)
        self.encoder4 = EncoderModule(256, num_heads=4)

        self.conv_fuse = nn.Sequential(nn.Conv1d(256*4, 256*4, kernel_size=1, bias=False), #*4 to conctat all ecnoders
                                   nn.BatchNorm1d(256*4), 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(256*4+256*4, out_channels, bias=False)     # # global rep 256
        self.bn6 = nn.BatchNorm1d(out_channels) 
        self.dp1 = nn.Dropout(0.5)
       # self.linear2 = nn.Linear(1024, 512) # global rep

        

    def forward(self,x):
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

        x = self.linear1(x) # best learned representations are after this layer
        x = self.bn6(x) 
      #  x = F.leaky_relu(x, negative_slope=0.2) # TODO this a part of projection layer
        x = self.dp1(x)       
       # x = self.linear2(x) # representation
        return x



class EmbeddingModule(nn.Module):
   
    def __init__(self, in_channels=3, out_channels=256):

        super(EmbeddingModule, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, bias=False)

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

class EncoderModule(nn.Module):

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

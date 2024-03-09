import torch
import torch.nn as nn
import torch.nn.functional as F

#from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class PCT_ml(nn.Module):
    def __init__(self, out_dim=128, mask=None):
        super(PCT_ml, self).__init__()
        self.out_dim = out_dim
        # Assuming EmbeddingModule and EncoderModule are defined elsewhere
        self.embedding_module = EmbeddingModule(in_channels=3, out_channels=out_dim, masked=mask)
        self.encoder1 = EncoderModule(out_dim, num_heads=4)
        self.encoder2 = EncoderModule(out_dim, num_heads=4)
        self.encoder3 = EncoderModule(out_dim, num_heads=4)
        self.encoder4 = EncoderModule(out_dim, num_heads=4)

        #self.encoders = nn.ModuleList([SA_Layer(channels=out_dim) for _ in range(4)])
        self.conv_fuse_conv = nn.Conv1d(out_dim*4, out_dim, kernel_size=1, bias=False)
        if not mask:
            self.conv_fuse_bn = nn.BatchNorm1d(out_dim)
        else:
            self.conv_fuse_bn = MaskedBatchNorm1d(out_dim)

        self.conv_fuse_activation = nn.LeakyReLU(negative_slope=0.2)

        self.transform1 = nn.Linear(out_dim, out_dim)
        self.transform2 = nn.Linear(out_dim, out_dim)

    def forward(self, x, mask=None): # the last dim is binary mask
        batch_size, _, num_points = x.size()

        x = self.embedding_module(x, mask)
        #for encoder in self.encoders:
         #   x = encoder(x, mask=mask)
        x1 = self.encoder1(x, mask=mask)
        x2 = self.encoder2(x1, mask=mask)
        x3 = self.encoder3(x2, mask=mask)
        x4 = self.encoder4(x3, mask=mask)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse_conv(x)
        x = self.conv_fuse_bn(x)
      #  if mask is None:
       #     x = self.conv_fuse_bn(x)
       # else:
       #     x = self.conv_fuse_bn(x, mask)
        lf = self.conv_fuse_activation(x)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).to(dtype=lf.dtype)
            lf = torch.where(mask_expanded> 0, lf, torch.tensor(float('-inf')))
            max_pool = torch.max(lf,2, keepdim=True)[0].view(batch_size, -1)
            lf = torch.where(mask_expanded > 0, lf, torch.tensor(0.0, device=lf.device))
            sum_unmasked = lf.sum(dim=2)
            num_unmasked = mask_expanded.sum(dim=2).clamp(min=1)
            avg_pool = (sum_unmasked / num_unmasked).view(batch_size, -1)
        else:
            max_pool = torch.max(x, 2, keepdim=True)[0].view(batch_size, -1)
            avg_pool = torch.mean(x, 2).view(batch_size, -1)

        transform1 = self.transform1(max_pool)
        transform2 = self.transform2(avg_pool)

        gf = torch.cat([transform1, transform2], dim=1)
        #lf = torch.cat([lf, gf.unsqueeze(-1).repeat(1,1,lf.shape[-1])], dim=1)
        return gf

class SPCT(nn.Module):


    def __init__(self, output_channels=40):
        super(SPCT, self).__init__()

        self.conv1 = nn.Conv1d(3, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=1, bias=False) #128 <-> 256

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)

        self.encoder1 = EncoderModule(256, num_heads=4)
        self.encoder2 = EncoderModule(256, num_heads=4)
        self.encoder3 = EncoderModule(256, num_heads=4)
        self.encoder4 = EncoderModule(256, num_heads=4)

        self.conv_fuse = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024), # 1024
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.linear1 = nn.Linear(1024, 512, bias=False) #1024, 512
        self.bn6 = nn.BatchNorm1d(512) # 512
        self.dp1 = nn.Dropout(0.5)
 
        self.linear2 = nn.Linear(512, 256) #512, 256
        self.bn7 = nn.BatchNorm1d(256) #256
        self.dp2 = nn.Dropout(0.5)

        self.linear3 = nn.Linear(256, output_channels) #256

    def forward(self, x):
        batch_size, _, _ = x.size()
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)  

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv_fuse(x)
        x = F.adaptive_max_pool1d(x, 1)
        x = x.view(batch_size, -1)

        x = self.linear1(x)
        x = self.bn6(x) 
        x = F.leaky_relu(x, negative_slope=0.2) 
        x = self.dp1(x) 

        x = self.linear2(x)
        #x = self.bn7(x)
        x = F.leaky_relu(x, negative_slope=0.2) 
        x = self.dp2(x) 

        x = self.linear3(x)
        return x


class PCT_Decoder(nn.Module):
    def __init__(self, input_dim=128):
        super(PCT_Decoder, self).__init__()
        self.encoder = EncoderModule(num_heads=4, in_features=input_dim)
        self.linear_1 = nn.Sequential(nn.Conv1d(input_dim, 64, kernel_size=1, bias=False), 
                                   nn.BatchNorm1d(64), 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear_2 = nn.Sequential(nn.Conv1d(64, 32, kernel_size=1, bias=False), 
                                   nn.BatchNorm1d(32), 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear_3 = nn.Conv1d(32, 3, kernel_size=1, bias=False)

    def forward(self, x):
       # x = self.encoder(x)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x

class PCT_BASE2(nn.Module): 

    def __init__(self, in_channels=3, out_channels=128):
        super(PCT_BASE2, self).__init__()

        self.embedding_module = EmbeddingModule(in_channels=in_channels, out_channels=out_channels)

        self.encoder1 = EncoderModule(out_channels, num_heads=4)
        self.encoder2 = EncoderModule(out_channels, num_heads=4)
        self.encoder3 = EncoderModule(out_channels, num_heads=4)
        self.encoder4 = EncoderModule(out_channels, num_heads=4)

        self.conv_fuse = nn.Sequential(nn.Conv1d(out_channels*4, out_channels*4, kernel_size=1, bias=False), #*4 to conctat all ecnoders
                                   nn.BatchNorm1d(out_channels*4), 
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(out_channels*8, out_channels*2, bias=False)    # alternative global rep
      #  self.bn6 = nn.BatchNorm1d(256) 
      #  self.dp1 = nn.Dropout(0.5)
      #  self.linear2 = nn.Linear(256, out_channels) # global rep
    

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
        x = self.linear1(x)
       # x = self.bn6(x) 
       # x = F.leaky_relu(x, negative_slope=0.2) 
       # x = self.dp1(x) 

       # x = self.linear2(x)

        return x


class EmbeddingModule(nn.Module):
   
    def __init__(self, in_channels=3, out_channels=256, masked=False):

        super(EmbeddingModule, self).__init__()
        self.masked = masked
        # Convolutional Layers
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=out_channels//2, out_channels=out_channels, kernel_size=1, bias=False)

        if not masked: 
            self.bn1 = nn.BatchNorm1d(out_channels//2)
            self.bn2 = nn.BatchNorm1d(out_channels)
        else:
            self.bn1 = MaskedBatchNorm1d(out_channels//2) 
            self.bn2 = MaskedBatchNorm1d(out_channels)

    def forward(self, x, mask=None):
        x = self.conv1(x)
        if not self.masked:
            x = self.bn1(x)
        else:
            x = self.bn1(x, mask)
        x = F.leaky_relu(x,negative_slope=0.2)

        # Apply second convolution and batch normalization
        x = self.conv2(x)
        if not self.masked:
            x = self.bn2(x)
        else:
            x = self.bn2(x,mask)
        x = F.leaky_relu(x,negative_slope=0.2)

        if mask is not None:
            x = x * mask.unsqueeze(1) # broadcasting mask across feature dim to zero out mask point features

        return x

class EncoderModule(nn.Module):

    def __init__(self, in_features=256, num_heads=8, masked=False):
        super(EncoderModule, self).__init__()
        self.masked=masked
        #self.q_conv = nn.Conv1d(in_features, in_features, 1, bias=False) # Q
        #self.k_conv = nn.Conv1d(in_features, in_features, 1, bias=False) # K, queries and keys of same dimentionality
        #self.v_conv = nn.Conv1d(in_features, in_features, 1) 

        self.mh_sa = nn.MultiheadAttention(num_heads=num_heads, embed_dim=in_features, batch_first=True)
       # self.mh_sa = MultiHeadSelfAttention(in_features=in_features, head_dim=int(in_features/num_heads), num_heads=num_heads) 
        if not masked:
            self.bn_after_sa = nn.BatchNorm1d(in_features)
        else:
            self.bn_after_sa = MaskedBatchNorm1d(in_features)

        # Linear layer is to learn  interactions within each embedding independently of other embeddings
        self.linear_after_sa =  nn.Linear(in_features,in_features) 
        if not masked:
            self.bn_after_linear_sa = nn.BatchNorm1d(in_features) 
        else:
            self.bn_after_linear_sa = MaskedBatchNorm1d(in_features) 

        
    def forward(self, x, mask=None):
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask.squeeze(1) == False 
        x_attention, _ = self.mh_sa(x.permute(0,2,1), x.permute(0,2,1), x.permute(0,2,1), key_padding_mask=key_padding_mask)
        x_attention = x_attention.permute(0, 2, 1)
        x = x + x_attention
        if not self.masked:
            x = self.bn_after_sa(x)
        else:
            x = self.bn_after_sa(x,mask)
        x = x.permute(0, 2, 1)
        x_linear = self.linear_after_sa(x)
        x = x + x_linear
        x = x.permute(0, 2, 1)
        if not self.masked:
            x = self.bn_after_linear_sa(x)
        else:
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


class MultiHeadSelfAttentionTorch(nn.Module):
    def __init__(self, in_features, head_dim, num_heads):
        super(MultiHeadSelfAttentionTorch, self).__init__()
        
        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attention = nn.MultiheadAttention(embed_dim=in_features, num_heads=num_heads, batch_first=True)
        self.out_linear = nn.Linear(in_features, in_features)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        attn_output, _ = self.attention(x, x, x)  # Self-attention
        #output = self.out_linear(attn_output)
       # output = output.permute(0, 2, 1)
        
        return  attn_output.permute(0,2,1) #output


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class MaskedBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # Parameters for scale and shift
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x, mask=None):
        if self.training:
            if mask is not None:
                mask = mask.unsqueeze(1).to(x.dtype)  # [batch_size, 1, num_points]
                masked_x = x * mask
                # Calculate mean and variance only on unmasked points
                sum_x = masked_x.sum([0, 2])
                sum_x_sq = (masked_x ** 2).sum([0, 2])
                n = mask.sum([0, 2])  # Number of unmasked points per feature
                mean = sum_x / n
                var = (sum_x_sq / n) - (mean ** 2)
            else:
                # Normal mean and variance calculation
                mean = x.mean([0, 2])
                var = x.var([0, 2], unbiased=False)
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x = (x - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)
        # Apply scale and shift
        out = self.gamma[None, :, None] * x + self.beta[None, :, None]
        return out
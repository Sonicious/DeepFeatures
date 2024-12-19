import numpy as np
import torch
from torch import nn, nn as nn
from torch.nn import init


class ChannelAttention3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention3D, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool3d((None, 1, 1))  # Preserve temporal dimension
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Preserve temporal dimension
        self.se = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // reduction, channel, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SqueezeAndExcitation3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeAndExcitation3D, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Preserve temporal dimension
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c, t)  # Preserve temporal dimension
        y = self.fc(y.view(-1, c)).view(b, c, t, 1, 1)
        return y

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, kernel_size, kernel_size), padding=(0, kernel_size // 2, kernel_size // 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Max and average pooling along the channel dimension
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # Shape: (batch, 1, frames, h, w)
        avg_result = torch.mean(x, dim=1, keepdim=True)    # Shape: (batch, 1, frames, h, w)
        result = torch.cat([max_result, avg_result], dim=1)  # Shape: (batch, 2, frames, h, w)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention3D(channel=channel, reduction=reduction)
        self.sa = SpatialAttention3D(kernel_size=kernel_size)

    def forward(self, x):
        residual = x
        out = x * self.ca(x)  # Apply Channel Attention
        out = out * self.sa(out)  # Apply Spatial Attention
        return out + residual  # Residual connection


class CBAMWithSE(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAMWithSE, self).__init__()
        # Replace CBAM Channel Attention with SE Block
        self.channel_attention = SqueezeAndExcitation3D(channel, reduction)
        # Keep CBAM Spatial Attention
        self.spatial_attention = SpatialAttention3D(kernel_size)

    def forward(self, x):
        residual = x
        # Apply SE block for channel attention
        out = x * self.channel_attention(x)
        # Apply CBAM spatial attention
        out = out * self.spatial_attention(out)
        return out + residual


class MultiScaleAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], reduction_ratio=8):
        super(MultiScaleAttentionBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, k, k), padding=(0, k // 2, k // 2))
            for k in kernel_sizes
        ])
        self.attention = CBAMBlock(out_channels, reduction=reduction_ratio)
        self.fusion = nn.Conv3d(len(kernel_sizes) * out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Apply convolutions of different kernel sizes
        multi_scale_features = [conv(x) for conv in self.convs]
        combined = torch.cat(multi_scale_features, dim=1)  # Concatenate along channel dimension
        fused = self.fusion(combined)  # Fuse multi-scale features
        attended = self.attention(fused)  # Apply CBAM attention
        return attended


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=16, num_heads=4, dropout_prob=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Input shape: (batch_size, channels=16, frames=11, height=15, width=15)
        batch_size, channels, frames, height, width = x.shape

        # Flatten spatial dimensions for attention
        x = x.view(batch_size, channels, frames * height * width).permute(0, 2, 1)  # (batch_size, seq_len, embed_dim)

        # Apply attention
        attended, _ = self.attention(x, x, x)  # Self-attention
        attended = self.norm(attended)
        attended = self.dropout(attended)

        # Reshape back to original dimensions
        attended = attended.permute(0, 2, 1).view(batch_size, channels, frames, height, width)

        return attended

class TemporalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_position=50):
        super(TemporalPositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position + 1, d_model)

    def forward(self, cumulative_positions):
        embeddings = self.position_embeddings(cumulative_positions)  # Shape: (frames, d_model)
        return embeddings
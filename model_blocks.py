import torch
import torch.nn as nn
from attention import CBAMBlock, MultiScaleAttentionBlock, ConvAttentionBlock



class MultiScaleDimensionalityReducer(nn.Module):
    def __init__(self, in_channels=221, reduction_ratio=8, dropout_prob=0.07):
        super(MultiScaleDimensionalityReducer, self).__init__()

        # Block 1: Multi-scale attention with CBAM and spatial downsampling
        self.block1 = MultiScaleAttentionBlock(
            in_channels, in_channels, kernel_sizes=[3, 7], reduction_ratio=reduction_ratio
        )
        #self.conv_res = nn.Conv3d(in_channels=in_channels, out_channels=4, kernel_size=(1, 4, 4), stride=(1, 3, 3))
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )

        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Further reduction
        self.block2 = MultiScaleAttentionBlock(
            64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=2, stride=2, padding=1),  # Downsample to ~5x5
            nn.SiLU()  # Add SiLU here
        )


        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final block for dimensionality reduction
        self.block3 = MultiScaleAttentionBlock(
            16, 4, kernel_sizes=[3], reduction_ratio=2
        )



    def forward(self, x):
        # Input: (batch_size, frames, height=15, width=15, spectral_indices=209)
        x = x.view(-1, 221, 15, 15)

        y = x
        # Block 1
        x = self.block1(x)
        #print(x.shape)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)

        x = self.conv1(x)
        #print(x.shape)

        x = self.dropout1(x)

        # Block 2
        y = x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        #print(x.shape)

        x = self.conv2(x)
        #print(x.shape)

        x = self.dropout2(x)

        # Block 3
        x = self.block3(x)
        #print(x.shape)

        #x = x + z
        #x = self.conv3(x)
        x = x.view(-1, 11, 4, 4, 4)
        return x


class MultiScaleAttentionUpscaler(nn.Module):
    def __init__(self, in_channels=16, out_channels=221, reduction_ratio=8, dropout_prob=0.1):
        super(MultiScaleAttentionUpscaler, self).__init__()

        #self.conv0 = nn.Conv3d(4, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1))  # Downsample to ~5x5
        self.conv0 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=1, stride=1),
            nn.SiLU()  # Add SiLU here
        )
        # Block 1: Multi-scale attention with CBAM and upsampling
        self.block1 = MultiScaleAttentionBlock(
            in_channels, in_channels, kernel_sizes=[3], reduction_ratio=reduction_ratio
        )
        self.pool =nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        #self.conv_res = nn.ConvTranspose3d(in_channels=in_channels, out_channels=128, kernel_size=(1, 6, 6), stride=(1, 3, 3))

        #self.conv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=64, kernel_size=2, stride=2),
            nn.SiLU()  # Add SiLU here
        )
        #self.upsample1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # Block 2: Multi-scale attention with CBAM and further upsampling
        self.block2 = MultiScaleAttentionBlock(
            64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        )
        #self.upsample2 = nn.Upsample(scale_factor=(1, 1.9, 1.9), mode="trilinear", align_corners=True)
        self.upsample2 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)

        # Final Block: Full reconstruction
        self.block3 = MultiScaleAttentionBlock(
            128, out_channels, kernel_sizes=[3, 7], reduction_ratio=reduction_ratio
        )

    def forward(self, x):
        # Input: (batch_size, reduced_channels=12, frames, height=5, width=5)

        # Block 1
        #print('_____________________')
        #print(x.shape)

        x = self.conv0(x)
        #print(x.shape)
        y=x
        x = self.block1(x)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)
        #print(x.shape)
        #print(z.shape)
        x = self.conv1(x)  # Upsample to ~10x10
        #print(x.shape)
        x = self.dropout1(x)

        # Block 2
        y=x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        x = self.upsample2(x)  # Upsample to ~15x15
        x = self.dropout2(x)
        #x = x+z

        # Block 3
        x = self.block3(x)
        #print(x.shape)

        # Change back to original format
        x = x.view(-1, 11, 221, 15, 15)

        # Output shape: (batch_size, frames, height=15, width=15, out_channels=209)
        return x

class AttentionUpscaler(nn.Module):
    def __init__(self, in_channels=16, out_channels=221, reduction_ratio=8, dropout_prob=0.07):
        super(AttentionUpscaler, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.SiLU()  # Add SiLU here
        )
        # Block 1: Multi-scale attention with CBAM and upsampling
        self.block1 = ConvAttentionBlock(
            in_channels, in_channels, kernel_size=3, reduction_ratio=reduction_ratio
        )
        self.pool =nn.MaxPool3d(kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))
        #self.conv_res = nn.ConvTranspose3d(in_channels=in_channels, out_channels=128, kernel_size=(1, 6, 6), stride=(1, 3, 3))

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.SiLU()  # Add SiLU here
        )
        self.dropout1 = nn.Dropout3d(p=dropout_prob)

        # Block 2: Multi-scale attention with CBAM and further upsampling
        self.block2 = ConvAttentionBlock(
            64, 64, kernel_size=3, reduction_ratio=reduction_ratio
        )
        self.upsample2 = nn.ConvTranspose3d(64, out_channels, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.dropout2 = nn.Dropout3d(p=dropout_prob)


    def forward(self, x):
        # Input: (batch_size, reduced_channels=12, frames, height=5, width=5)

        # Block 1
        #print('_____________________')
        #print(x.shape)

        x = self.conv0(x)
        #print(x.shape)
        y=x
        x = self.block1(x)
        x = self.pool(x)
        x = x + y
        #z = self.conv_res(x)
        #print(x.shape)
        #print(z.shape)
        x = self.conv1(x)  # Upsample to ~10x10
        #print(x.shape)
        x = self.dropout1(x)

        # Block 2
        y=x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        #print(x.shape)
        x = self.upsample2(x)  # Upsample to ~15x15
        #print(x.shape)
        x = self.dropout2(x)
        #print(x.shape)
        #x = x+z

        # Change back to original format
        x = x.permute(0, 2, 3, 4, 1)  # (batch_size, frames, height, width, spectral_indices=209)
        return x






"""class DimensionalityReducer(nn.Module):
    def __init__(self, in_channels=209, reduction_ratio=8, dropout_prob=0.07):
        super(DimensionalityReducer, self).__init__()

        # Block 1: Multi-scale attention with CBAM and spatial downsampling

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool =nn.MaxPool3d(kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))

        #self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))  # Downsample to ~7x7
        self.dropout1 = nn.Dropout3d(p=dropout_prob)

        # Block 2: Further reduction
        #self.block2 = MultiScaleAttentionBlock(
        #    110, 110, kernel_sizes=[3, 7], reduction_ratio=reduction_ratio
        #)
        #self.pool2 = nn.Conv3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))  # Downsample to ~5x5
        self.dropout2 = nn.Dropout3d(p=dropout_prob)

        # Final block for dimensionality reduction
        self.block3 = MultiScaleAttentionBlock(
            64, 16, kernel_sizes=[3], reduction_ratio=reduction_ratio
        )

    def forward(self, x):
        # Input: (batch_size, frames, height=15, width=15, spectral_indices=209)
        x = x.permute(0, 4, 1, 2, 3)  # Change to (batch_size, channels=209, frames, height, width)
        y = x
        # Block 1
        x = self.block1(x)
        x = self.pool(x)
        x = x + y
        #print(x.shape)

        x = self.conv1(x)
        x = self.dropout1(x)
        #print(x.shape)


        # Block 2
        y = x
        #x = self.block2(x)
        x = self.pool(x)
        x = x + y
        #print(x.shape)
        x = self.conv2(x)
        x = self.dropout2(x)
        #print(x.shape)


        # Block 3
        x = self.block3(x)
        #print(x.shape)

        # Output shape: (batch_size, reduced_channels=12, frames, height=5, width=5)
        return x



class MultiScaleAttentionUpscaler(nn.Module):
    def __init__(self, in_channels=16, out_channels=209, reduction_ratio=8, dropout_prob=0.07):
        super(MultiScaleAttentionUpscaler, self).__init__()

        # Block 1: Multi-scale attention with CBAM and upsampling
        self.block1 = MultiScaleAttentionBlock(
            in_channels, in_channels, kernel_sizes=[3], reduction_ratio=reduction_ratio
        )
        self.pool =nn.MaxPool3d(kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))

        self.conv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        #self.upsample1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.dropout1 = nn.Dropout3d(p=dropout_prob)

        # Block 2: Multi-scale attention with CBAM and further upsampling
        #self.block2 = MultiScaleAttentionBlock(
        #    50, 50, kernel_sizes=[3, 7], reduction_ratio=reduction_ratio
        #)
        #self.upsample2 = nn.Upsample(scale_factor=(1, 1.9, 1.9), mode="trilinear", align_corners=True)
        self.upsample2 = nn.ConvTranspose3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.dropout2 = nn.Dropout3d(p=dropout_prob)

        # Final Block: Full reconstruction
        self.block3 = MultiScaleAttentionBlock(
            128, out_channels, kernel_sizes=[3, 11], reduction_ratio=reduction_ratio
        )

    def forward(self, x):
        # Input: (batch_size, reduced_channels=12, frames, height=5, width=5)

        # Block 1
        #print('_____________________')
        y=x
        x = self.block1(x)
        x = self.pool(x)
        x = x + y
        #print(x.shape)
        x = self.conv1(x)  # Upsample to ~10x10
        #print(x.shape)
        x = self.dropout1(x)

        # Block 2
        y=x
        #x = self.block2(x)
        x = self.pool(x)
        x = x + y
        #print(x.shape)
        x = self.upsample2(x)  # Upsample to ~15x15
        #print(x.shape)
        x = self.dropout2(x)

        # Block 3
        x = self.block3(x)
        #print(x.shape)

        # Change back to original format
        x = x.permute(0, 2, 3, 4, 1)  # (batch_size, frames, height, width, spectral_indices=209)

        # Output shape: (batch_size, frames, height=15, width=15, out_channels=209)
        return x"""
import torch
import torch.nn as nn
from attention import CBAMBlock, MultiScaleAttentionBlock


class DimensionalityReducer(nn.Module):
    def __init__(self, in_channels=209, reduction_ratio=8, dropout_prob = 0.1):
        super(DimensionalityReducer, self).__init__()
        kernel = (1, 2, 2)
        # First convolutional block for dimensionality reduction
        self.cbam1 = CBAMBlock(209, reduction=reduction_ratio)
        self.conv1 = nn.Conv3d(in_channels=209, out_channels=168, kernel_size=kernel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout1 = nn.Dropout3d(p=dropout_prob)
        self.conv12 = nn.Conv3d(in_channels=168, out_channels=128, kernel_size=kernel)
        self.relu12 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout12 = nn.Dropout3d(p=dropout_prob)


        # Second convolutional block for further reduction
        self.cbam2 = CBAMBlock(128, reduction=reduction_ratio)
        self.conv2 = nn.Conv3d(in_channels=128, out_channels=96, kernel_size=kernel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout2 = nn.Dropout3d(p=dropout_prob)
        self.conv22 = nn.Conv3d(in_channels=96, out_channels=64, kernel_size=kernel)
        self.relu22 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout22 = nn.Dropout3d(p=dropout_prob)

        # Third convolutional block for further reduction
        self.cbam3 = CBAMBlock(64, reduction=reduction_ratio)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=48, kernel_size=kernel)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout3 = nn.Dropout3d(p=dropout_prob)
        self.conv32 = nn.Conv3d(in_channels=48, out_channels=32, kernel_size=kernel)
        self.relu32 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout32 = nn.Dropout3d(p=dropout_prob)

        # Fourth convolutional block for further reduction
        self.cbam4 = CBAMBlock(32, reduction=reduction_ratio)
        self.conv4 = nn.Conv3d(in_channels=32, out_channels=24, kernel_size=kernel)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout4 = nn.Dropout3d(p=dropout_prob)
        self.conv42 = nn.Conv3d(in_channels=24, out_channels=16, kernel_size=kernel)
        self.relu42 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout42 = nn.Dropout3d(p=dropout_prob)
        self.cbam5 = CBAMBlock(16, reduction=reduction_ratio)



        # Fully connected layer to project reduced dimensions into latent space
        #self.flatten = nn.Flatten()
        #self.fc = nn.Linear(in_channels * 11 * 3 * 3, 16)  # Adjust based on reduced dimensions

    def forward(self, x):
        # Input: (batch_size, frames, h, w, channels)
        x = x.permute(0, 4, 1, 2, 3)  # (batch_size, channels, frames, h, w)

        # First convolutional block
        x = self.cbam1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv12(x)
        x = self.relu12(x)
        x = self.dropout12(x)
        # Second convolutional block
        x = self.cbam2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv22(x)
        x = self.relu22(x)
        x = self.dropout22(x)
        # Third convolution block
        x = self.cbam3(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.conv32(x)
        x = self.relu32(x)
        x = self.dropout32(x)
        # Fourth convolution block
        x = self.cbam4(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.conv42(x)
        x = self.relu42(x)
        x = self.dropout42(x)
        x = self.cbam5(x)

        # Flatten and reduce to latent space
        #x = self.flatten(x)
        #x = self.fc(x)  # Reduce to (batch_size, latent_dim)
        return x


class Upscaler(nn.Module):
    def __init__(self, out_channels=209, reduction_ratio=8, dropout_prob=0.1):
        super(Upscaler, self).__init__()
        # Fully connected layer to expand latent space to initial spatial-temporal dimensions
        #self.fc = nn.Linear(16, out_channels * 11 * 3 * 3)

        kernel = (1, 2, 2)

        # First transposed convolutional block for upscaling
        self.cbam1 = CBAMBlock(16, reduction=reduction_ratio)
        self.conv1 = nn.ConvTranspose3d(in_channels=16, out_channels=24, kernel_size=kernel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout1 = nn.Dropout3d(p=dropout_prob)
        self.conv12 = nn.ConvTranspose3d(in_channels=24, out_channels=32, kernel_size=kernel)
        self.relu12 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout12 = nn.Dropout3d(p=dropout_prob)

        # Second transposed convolutional block for further upscaling
        self.cbam2 = CBAMBlock(32, reduction=reduction_ratio)
        self.conv2 = nn.ConvTranspose3d(in_channels=32, out_channels=48, kernel_size=kernel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout2 = nn.Dropout3d(p=dropout_prob)
        self.conv22 = nn.ConvTranspose3d(in_channels=48, out_channels=64, kernel_size=kernel)
        self.relu22 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout22 = nn.Dropout3d(p=dropout_prob)

        # Second transposed convolutional block for further upscaling
        self.cbam3 = CBAMBlock(64, reduction=reduction_ratio)
        self.conv3 = nn.ConvTranspose3d(in_channels=64, out_channels=96, kernel_size=kernel)
        self.relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout3 = nn.Dropout3d(p=dropout_prob)
        self.conv32 = nn.ConvTranspose3d(in_channels=96, out_channels=128, kernel_size=kernel)
        self.relu32 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout32 = nn.Dropout3d(p=dropout_prob)

        # Second transposed convolutional block for further upscaling
        self.cbam4 = CBAMBlock(128, reduction=reduction_ratio)
        self.conv4 = nn.ConvTranspose3d(in_channels=128, out_channels=168, kernel_size=kernel)
        self.relu4 = nn.LeakyReLU(negative_slope=0.1)
        self.dropout4 = nn.Dropout3d(p=dropout_prob)
        self.conv42 = nn.ConvTranspose3d(in_channels=168, out_channels=209, kernel_size=kernel)
        self.relu42 = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        # Input: (batch_size, latent_dim)
        #x = self.fc(x)  # Expand to (batch_size, out_channels * frames * h * w)
        #x = x.view(-1, 209, 11, 3, 3)  # Reshape to (batch_size, channels, frames, h, w)

        # First transposed convolutional block
        x = self.cbam1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv12(x)
        x = self.relu12(x)
        # Second transposed convolutional block
        x = self.cbam2(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv22(x)
        x = self.relu22(x)
        # Third transposed convolutional block
        x = self.cbam3(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv32(x)
        x = self.relu32(x)
        # Fourth transposed convolutional block
        x = self.cbam4(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.dropout4(x)
        x = self.conv42(x)
        x = self.relu42(x)

        # Restore original dimensions order
        x = x.permute(0, 2, 3, 4, 1)  # (batch_size, frames, h, w, channels)
        return x


class MultiScaleAttentionUpscaler(nn.Module):
    def __init__(self, in_channels=16, out_channels=209, reduction_ratio=8, dropout_prob=0.1):
        super(MultiScaleAttentionUpscaler, self).__init__()

        # First upscaling block
        self.block1 = MultiScaleAttentionBlock(
            12, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        )
        self.upsample1 = nn.Upsample(scale_factor=(1, 1.2, 1.2), mode="trilinear", align_corners=True)
        self.dropout1 = nn.Dropout3d(p=dropout_prob)

        # Second upscaling block
        self.block2 = MultiScaleAttentionBlock(
            64, 128, kernel_sizes=[3, 5], reduction_ratio=reduction_ratio
        )
        self.upsample2 = nn.Upsample(scale_factor=(1, 1.9, 1.9), mode="trilinear", align_corners=True)
        self.dropout2 = nn.Dropout3d(p=dropout_prob)

        # Third upscaling block
        self.block3 = MultiScaleAttentionBlock(
            128, 168, kernel_sizes=[3, 5, 7], reduction_ratio=reduction_ratio
        )
        self.upsample3 = nn.Upsample(scale_factor=(1, 1.2, 1.2), mode="trilinear", align_corners=True)
        self.dropout3 = nn.Dropout3d(p=dropout_prob)

        # Final block for full reconstruction
        self.block4 = MultiScaleAttentionBlock(
            168, out_channels, kernel_sizes=[3, 5, 7], reduction_ratio=reduction_ratio
        )

    def forward(self, x):
        # Input: (batch_size, reduced_channels=16, frames=11, reduced_height=2, reduced_width=2)

        # Block 1
        #print('block1')
        x = self.block1(x)
        #print(x.shape)
        x = self.upsample1(x)  # Upscale spatial dimensions
        #print(x.shape)
        x = self.dropout1(x)

        # Block 2
        #print('block2')
        x = self.block2(x)
        #print(x.shape)
        x = self.upsample2(x)  # Further upscale spatial dimensions
        #print(x.shape)
        x = self.dropout2(x)

        # Block 3
        #print('block3')
        x = self.block3(x)
        #print(x.shape)
        x = self.upsample3(x)  # Final spatial upsampling
        #print(x.shape)
        x = self.dropout3(x)

        # Block 4
        #print('block4')
        x = self.block4(x)
        #print(x.shape)
        x = x.permute(0, 2, 3, 4, 1)
        # Output shape: (batch_size, out_channels=209, frames=11, height=15, width=15)
        return x


import torch
import torch.nn as nn

class DimensionalityReducer(nn.Module):
    def __init__(self, in_channels=209, reduction_ratio=8, dropout_prob=0.07):
        super(DimensionalityReducer, self).__init__()

        # Block 1: Multi-scale attention with CBAM and spatial downsampling
        self.block1 = MultiScaleAttentionBlock(
            in_channels, in_channels, kernel_sizes=[3, 9], reduction_ratio=reduction_ratio
        )
        self.conv_res = nn.Conv3d(in_channels=in_channels, out_channels=16, kernel_size=(1, 4, 4), stride=(1, 3, 3))
        self.pool =nn.MaxPool3d(kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=128, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.SiLU()  # Add SiLU here
        )

        #self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))  # Downsample to ~7x7
        self.dropout1 = nn.Dropout3d(p=dropout_prob)

        # Block 2: Further reduction
        self.block2 = MultiScaleAttentionBlock(
            128, 128, kernel_sizes=[3, 7], reduction_ratio=reduction_ratio
        )
        #self.pool2 = nn.Conv3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        self.conv2 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1)),  # Downsample to ~5x5
            nn.SiLU()  # Add SiLU here
        )


        self.dropout2 = nn.Dropout3d(p=dropout_prob)

        # Final block for dimensionality reduction
        self.block3 = MultiScaleAttentionBlock(
            64, 16, kernel_sizes=[3], reduction_ratio=reduction_ratio
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, 4, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.SiLU()  # Add SiLU here
        )


    def forward(self, x):
        # Input: (batch_size, frames, height=15, width=15, spectral_indices=209)
        x = x.permute(0, 4, 1, 2, 3)  # Change to (batch_size, channels=209, frames, height, width)
        y = x
        # Block 1
        x = self.block1(x)
        x = self.pool(x)
        x = x + y

        z = self.conv_res(x)
        #print(z.shape)

        x = self.conv1(x)
        x = self.dropout1(x)
        #print(x.shape)


        # Block 2
        y = x
        x = self.block2(x)
        x = self.pool(x)
        x = x + y
        #print(x.shape)
        x = self.conv2(x)
        x = self.dropout2(x)
        #print(x.shape)


        # Block 3
        x = self.block3(x)
        x = x + z
        x = self.conv3(x)
        #print(x.shape)

        # Output shape: (batch_size, reduced_channels=12, frames, height=5, width=5)
        return x



class MultiScaleAttentionUpscaler(nn.Module):
    def __init__(self, in_channels=16, out_channels=209, reduction_ratio=8, dropout_prob=0.07):
        super(MultiScaleAttentionUpscaler, self).__init__()

        #self.conv0 = nn.Conv3d(4, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1))  # Downsample to ~5x5
        self.conv0 = nn.Sequential(
            nn.Conv3d(4, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.SiLU()  # Add SiLU here
        )
        # Block 1: Multi-scale attention with CBAM and upsampling
        self.block1 = MultiScaleAttentionBlock(
            in_channels, in_channels, kernel_sizes=[3], reduction_ratio=reduction_ratio
        )
        self.pool =nn.MaxPool3d(kernel_size=(1,3,3), stride=1, padding=(0, 1, 1))
        #self.conv_res = nn.ConvTranspose3d(in_channels=in_channels, out_channels=128, kernel_size=(1, 6, 6), stride=(1, 3, 3))

        #self.conv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=64, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.SiLU()  # Add SiLU here
        )
        #self.upsample1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.dropout1 = nn.Dropout3d(p=dropout_prob)

        # Block 2: Multi-scale attention with CBAM and further upsampling
        self.block2 = MultiScaleAttentionBlock(
            64, 64, kernel_sizes=[3], reduction_ratio=reduction_ratio
        )
        #self.upsample2 = nn.Upsample(scale_factor=(1, 1.9, 1.9), mode="trilinear", align_corners=True)
        self.upsample2 = nn.ConvTranspose3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.dropout2 = nn.Dropout3d(p=dropout_prob)

        # Final Block: Full reconstruction
        self.block3 = MultiScaleAttentionBlock(
            128, out_channels, kernel_sizes=[3, 9], reduction_ratio=reduction_ratio
        )

    def forward(self, x):
        # Input: (batch_size, reduced_channels=12, frames, height=5, width=5)

        # Block 1
        #print('_____________________')
        #print(x.shape)

        x = self.conv0(x)
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

        # Block 3
        x = self.block3(x)
        #print(x.shape)

        # Change back to original format
        x = x.permute(0, 2, 3, 4, 1)  # (batch_size, frames, height, width, spectral_indices=209)

        # Output shape: (batch_size, frames, height=15, width=15, out_channels=209)
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
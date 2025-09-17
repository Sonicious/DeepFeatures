from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
import torch.optim as optim
import lightning.pytorch as pl
try:
    from model.attention import TemporalPositionalEmbedding
    from model.loss import WeightedMaskedLoss
    from model.model_blocks import *
except:
    from attention import TemporalPositionalEmbedding
    from loss import WeightedMaskedLoss
    from model_blocks import *

import torch
import torch.nn as nn


class TransformerAE(pl.LightningModule):
    def __init__(self, dbottleneck, frames=11, max_position=350, loss_fn=WeightedMaskedLoss(), learning_rate=1e-4):
        super(TransformerAE, self).__init__()

        self.frames = frames
        self.num_reduced_tokens = 4

        num_dims = 10

        if num_dims == 221:
            self.dim_reducer = MultiScaleDimensionalityReducer_221()
            self.upscaler = MultiScaleAttentionUpscaler_221()
            self.d_model = 64
            dim_feedforward = 2
        elif num_dims == 149:
            self.d_model = 64 * 4
            dim_feedforward = 2
            self.dim_reducer = MultiScaleDimensionalityReducer_149(out_channels=int(self.d_model / 16))#in_channels=149, reduction_ratio=8)
            self.upscaler = MultiScaleAttentionUpscaler_149(in_channels=int(self.d_model / 16))#in_channels=8, out_channels=149)
        elif num_dims == 10:
            self.d_model = 64 * 4
            dim_feedforward = 8
            #self.dim_reducer = MultiScaleDimensionalityReducer_12(out_channels=int(self.d_model / 16))#in_channels=149, reduction_ratio=8)
            self.dim_reducer = MultiScaleDimensionalityReducer_12(in_channels=12, out_channels=int(self.d_model / 16))#in_channels=149, reduction_ratio=8)
            #self.upscaler = MultiScaleAttentionUpscaler_12(in_channels=int(self.d_model / 16))#in_channels=8, out_channels=149)
            self.upscaler = MultiScaleAttentionUpscaler_12(out_channels=12, in_channels=int(self.d_model / 16))#in_channels=8, out_channels=149)
        else:
            self.dim_reducer = MultiScaleDimensionalityReducer_68(in_channels=67, reduction_ratio=8)
            self.upscaler = MultiScaleAttentionUpscaler_68(in_channels=24, out_channels=67)
            self.d_model = 64 * 4
            dim_feedforward = 10

        self.positional_embedding_shared = TemporalPositionalEmbedding(d_model=self.d_model, max_position=max_position)
        self.transformer_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=8,
                dim_feedforward=dim_feedforward,
                dropout=0.25,
                batch_first=True  # Ensure batch-first data layout
            ),
            num_layers=3 #3 # Number of encoder layers
        )

        self.transformer_dec = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=8,
                dim_feedforward=2,
                dropout=0.25,
               batch_first=True  # Ensure batch-first data layout
            ),
            num_layers=3 #3 # Number of encoder layers
        )


        self.encoder_linear = nn.Sequential(
            nn.Linear(self.d_model, 16),
            #nn.LeakyReLU(negative_slope=0.1, inplace=True) ,  # Activation after linear layer
            nn.SELU()
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(16, self.d_model),
            #nn.LeakyReLU(negative_slope=0.1, inplace=True) ,  # Activation after linear layer
            nn.SELU()
        )

        # Token reduction and upsampling layers
        self.token_reducer = nn.Sequential(
            nn.Linear(self.frames, self.num_reduced_tokens),
            #nn.LeakyReLU(negative_slope=0.1, inplace=True),  # Activation for token reduction
            nn.SELU()
        )
        self.token_upsampler = nn.Sequential(
            nn.Linear(self.num_reduced_tokens, self.frames),
            #nn.LeakyReLU(negative_slope=0.01, inplace=True),   # Activation for token upsampling
            nn.SELU()
        )


        self.bottleneck_reducer = nn.Sequential(
            nn.Linear(16 * self.num_reduced_tokens, dbottleneck),
            nn.Softplus(beta=1, threshold=20)
        ) # Reduce to bottleneck
        self.bottleneck_expander = nn.Sequential(
            nn.Linear(dbottleneck, 16 * self.num_reduced_tokens),
            nn.Softplus(beta=1, threshold=20)
        ) # Expand from bottleneck


        # Loss and learning rate
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate


    def compute_cumulative_positions(self, time_gaps):
        # Initialize cumulative positions with zeros for each sample in the batch
        batch_size, frames_minus_one = time_gaps.size()
        cumulative_positions = torch.zeros((batch_size, self.frames), dtype=torch.long, device=time_gaps.device)

        # Fill cumulative positions by computing the cumulative sum for each batch sample
        cumulative_positions[:, 1:] = torch.cumsum(time_gaps, dim=1)

        return cumulative_positions

    def encode(self, x, time_gaps):
        # Apply dimensionality reduction
        x = self.dim_reducer(x)  # Expected shape: (batch_size, frames, 15, 15, 64)
        x = x.reshape(x.size(0), self.frames, -1)  # Expected shape: (batch_size, frames, din)
        # Add Positional embeddings & Pass through Transformer encoder
        cumulative_positions = self.compute_cumulative_positions(time_gaps)  # Shape: (batch_size, frames)
        pos_emb = self.positional_embedding_shared(cumulative_positions)  # Shape: (batch_size, frames, d2)
        pos_emb = pos_emb / torch.sqrt(torch.tensor(pos_emb.size(-1), dtype=torch.float))
        x = x + pos_emb  # Shape should match (frames, batch_size, d2)
        x = self.transformer_enc(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.encoder_linear(x)

        x = self.token_reducer(x.permute(0, 2, 1))

        x = x.reshape(x.size(0), -1)
        x = self.bottleneck_reducer(x)
        return x

    def decode(self, x, time_gaps):
        x = self.bottleneck_expander(x)
        #print(x.shape)
        #

        x = x.reshape(x.size(0), self.num_reduced_tokens, 16)

        x = self.token_upsampler(x.permute(0, 2, 1)).permute(0, 2, 1)
        #print(x.shape)
        x = self.decoder_linear(x)
        #print(x.shape)
        cumulative_positions = self.compute_cumulative_positions(time_gaps)  # Shape: (batch_size, frames)
        pos_emb = self.positional_embedding_shared(cumulative_positions)  # Shape: (frames, batch_size, d2)
        pos_emb = pos_emb/ torch.sqrt(torch.tensor(pos_emb.size(-1), dtype=torch.float))
        x = x + pos_emb
        # Pass through the Transformer decoder
        x = self.transformer_enc(x)
        x = x.reshape(x.size(0) * self.frames, self.d_model // 16, 4, 4)
        #print(x.shape)
        x = self.upscaler(x)  # Shape: (batch_size, frames, 15, 15, 209)
        #print(x.shape)
        return x

    def forward(self, x, time_gaps):
        encoded = self.encode(x, time_gaps)
        decoded = self.decode(encoded, time_gaps)
        return decoded, encoded


    def training_step(self, batch, batch_idx):
        x, time_gaps, mask = batch
        output, encoded = self(x, time_gaps)
        # Compute total days (sum of time gaps) per sample
        total_days_per_sample = time_gaps.sum(dim=1)

        # Compute number of False values in the mask per sample
        # mask shape: (batch, features, time, y, x)
        # Collapse all dimensions except batch
        false_values_per_sample = (~mask).reshape(mask.shape[0], -1).sum(axis=1)

        # Print both values together
        #for i, (days, false_count) in enumerate(zip(total_days_per_sample, false_values_per_sample)):
            #print(f"Sample {i}: total_days = {days}, false_values_in_mask = {false_count}")

        # Optional: also print shapes in case of mismatch
        #print("Output shape:", output.shape)
        #print("Target (x) shape:", x.shape)
        #print("Mask shape:", mask.shape)

        #try:
        total_loss, mse_loss, ssim_loss, sam_loss, center_mae = self.loss_fn(output, x, mask)
        #except:
        #    print(output)
        #    print(x)
        #    print(mask)
        self.log("train_total", total_loss, prog_bar=True, on_epoch=True)
        self.log("train_mae", mse_loss, prog_bar=True, on_epoch=True)
        self.log("train_ssim", ssim_loss, prog_bar=True, on_epoch=True)
        self.log("train_sam", sam_loss, prog_bar=True, on_epoch=True)
        self.log("train_center", center_mae, prog_bar=True, on_epoch=True)
        # Log current learning rate
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, time_gaps, mask = batch
        output, encoded = self(x, time_gaps)
        total_loss, mse_loss, ssim_loss, sam_loss, center_mae = self.loss_fn(output, x, mask, val=True)
        self.log("val_total", total_loss, prog_bar=True, on_epoch=True)
        self.log("val_mae", mse_loss, prog_bar=True, on_epoch=True)
        self.log("val_ssim", ssim_loss, prog_bar=True, on_epoch=True)
        self.log("val_sam", sam_loss, prog_bar=True, on_epoch=True)
        self.log("val_loss", center_mae, prog_bar=True, on_epoch=True)
        return center_mae

    def test_step(self, x, time_gaps, mask, batch_idx):
        """
        Defines the operations for a single test step.
        """
        # Unpack the batch
        #x, time_gaps, mask = batch

        # Forward pass
        output, encoded = self(x, time_gaps)

        # Compute the loss using the provided loss function
        loss = self.loss_fn(output, x, mask)

        # Log the test loss
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)

        return loss, output, encoded

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1.5e-4)

        # Define the warm-up scheduler
        warmup_scheduler = {
            "scheduler": LinearLR(optimizer, start_factor=0.001, total_iters=30000),
            "interval": "step",  # Apply at every training step
            "frequency": 1,
        }

        # Define the ReduceLROnPlateau scheduler
        plateau_scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.35,
                patience=5,
                verbose=False,
                min_lr=1.5e-6,
                threshold=1.5e-6,
                threshold_mode="abs",
            ),
            "interval": "epoch",  # Apply at the end of each epoch
            "monitor": "val_loss",  # Validation loss to monitor
        }

        # Return the optimizer and schedulers
        return [optimizer], [warmup_scheduler, plateau_scheduler]


def main():
    # Define model parameters
    frames = 11  # Number of frames we want to consider
    max_position = 50  # Maximum position index for embeddings
    batch_size = 2  # Batch size for dummy data

    # Instantiate the model
    model = TransformerAE(dbottleneck=7)

    # Generate dummy input data
    x = torch.randn(batch_size, frames, 12, 15, 15)


    # Generate dummy time gaps for each sample in the batch (shape: (batch_size, frames - 1))
    time_gaps = torch.tensor([
        [2, 5, 3, 2, 4, 2, 2, 4, 5, 3],  # Time gaps for the first sample in the batch
        [1, 3, 2, 4, 3, 5, 1, 2, 4, 3]  # Time gaps for the second sample in the batch
    ], dtype=torch.long)

    mask = torch.ones_like(x, dtype=torch.bool)
    # Ensure the model is in evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        decoded, _ = model(x, time_gaps)
        print('==============')
        #print(output.shape)

    # Output result
    print("Input shape:", x.shape)
    print("Time gaps shape:", time_gaps.shape)
    print("Output shape:", decoded.shape)

    print(model.loss_fn(decoded, x, mask))



if __name__ == "__main__":
    main()
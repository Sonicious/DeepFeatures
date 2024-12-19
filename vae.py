import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from attention import TemporalPositionalEmbedding
from model_blocks import *

#from model_blocks import DimensionalityReducer, Upscaler


class WeightedMaskedMSELoss(nn.Module):
    def __init__(self, spatial_size=(15, 15), frames=11, sparsity_weight=0.01, sparsity_target=0.05):
        super(WeightedMaskedMSELoss, self).__init__()
        self.spatial_size = spatial_size
        self.sparsity_weight = sparsity_weight  # Weight for sparsity penalty
        self.sparsity_target = sparsity_target  # Target sparsity level
        self.frames = frames
        self.weight_map = self.create_weight_map()

    def create_weight_map(self):
        """
        Create a spatial weight map for the spatial dimensions.
        The most central pixel gets a weight of 1, the surrounding pixels get
        progressively smaller weights based on their distance from the center.
        """
        h, w = self.spatial_size
        center = (h // 2, w // 2)

        weight_map = torch.zeros(h, w)
        for i in range(h):
            for j in range(w):
                dist = max(abs(i - center[0]), abs(j - center[1]))
                if dist == 0:
                    weight = 1.0
                elif dist > 4:
                    weight = 0.
                else:
                    weight = 0.01 * (0.1 ** (dist - 1))

                weight_map[i, j] = weight

        return weight_map

    def create_temporal_weights(self, frames):
        """
        Create temporal weights for each frame based on their distance from the central frame.
        """
        center_frame = frames // 2
        temporal_weights = torch.zeros(frames)

        for i in range(frames):
            dist = abs(i - center_frame)
            if dist == 0:
                weight = 1.0
            elif dist > 5:
                weight = 0.
            else:
                #weight = 0.1 * (0.1 ** (dist - 1))
                weight = 0.1 ** dist

            temporal_weights[i] = weight

        return temporal_weights

    def sparsity_penalty(self, activations):
        """
        Compute the sparsity penalty using KL divergence.
        activations: Tensor of shape (batch_size, latent_dim, ...) representing intermediate activations.
        """
        activation_mean = torch.mean(activations, dim=0)  # Average activation across the batch
        kl_divergence = self.sparsity_target * torch.log(self.sparsity_target / activation_mean + 1e-10) + \
                        (1 - self.sparsity_target) * torch.log(
            (1 - self.sparsity_target) / (1 - activation_mean + 1e-10))
        return torch.sum(kl_divergence)  # Sum over all latent dimensions


    def forward(self, output, target, mask, latent_activations = None):
        if torch.isnan(output).any():
            raise RuntimeError("Output contains NaN values")
        if torch.isnan(target).any():
            print("Target contains NaN values")
        if not mask.any():
            print("No valid values in the batch")
            return torch.tensor(0.0, requires_grad=True, device=output.device)

        batch_size, frames, h, w, indices = output.size()

        # Spatial weight map
        spatial_weight_map = self.weight_map.to(output.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        spatial_weight_map = spatial_weight_map.expand(batch_size, frames, h, w, indices)

        # Temporal weight map
        temporal_weights = self.create_temporal_weights(frames).to(output.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        temporal_weight_map = temporal_weights.expand(batch_size, frames, h, w, indices)

        # Combine spatial and temporal weights
        combined_weight_map = spatial_weight_map * temporal_weight_map

        # Apply the mask to select valid elements
        masked_output = output[mask]
        masked_target = target[mask]
        masked_weights = combined_weight_map[mask]

        # Compute weighted MSE loss
        weighted_loss = torch.mean(masked_weights * (masked_output - masked_target) ** 2)

        # Add sparsity penalty if latent_activations are provided
        sparsity_loss = 0.0
        if latent_activations is not None:
            sparsity_loss = self.sparsity_penalty(latent_activations)

            total_loss = weighted_loss + self.sparsity_weight * sparsity_loss
            return total_loss
        return weighted_loss


class TransformerAE(pl.LightningModule):
    def __init__(self, d2=637, dbottleneck=8, frames=11, max_position=50, in_channels=209, reduction_ratio=16,
                 loss_fn=WeightedMaskedMSELoss(), learning_rate=1e-4, num_reduced_tokens=3):
        super(TransformerAE, self).__init__()

        self.frames = frames
        self.num_reduced_tokens = num_reduced_tokens

        # Dimensionality reduction and upscaling modules
        self.kernel = (1, 2, 2)
        self.din = (15-(self.kernel[1]-1)*8) * (15-(self.kernel[2]-1)*8) * 16  # Flattened input dimension after reduction (15 * 15 spatial, 64 channels)

        self.dim_reducer = DimensionalityReducer(in_channels=209, reduction_ratio=8)

        # Attention block for latent features
        #self.attention = MultiHeadSelfAttention(embed_dim=16, num_heads=4)

        # Upscaler to reconstruct original dimensions
        self.upscaler = MultiScaleAttentionUpscaler(in_channels=16, out_channels=209)


        self.positional_embedding_enc = TemporalPositionalEmbedding(d_model=64, max_position=max_position)
        self.positional_embedding_dec = TemporalPositionalEmbedding(d_model=64, max_position=max_position)
        self.transformer_enc = nn.Transformer(d_model=64, nhead=4, num_encoder_layers=2,
                                              num_decoder_layers=2, dim_feedforward=64,
                                              dropout=0.1).encoder

        self.transformer_dec = nn.Transformer(d_model=64, nhead=4, num_encoder_layers=2,
                                              num_decoder_layers=2, dim_feedforward=64,
                                              dropout=0.1).encoder
        self.encoder_linear = nn.Sequential(
            nn.Linear(64, 16),
            nn.LeakyReLU(negative_slope=0.01),  # Activation after linear layer
        )
        self.decoder_linear = nn.Sequential(
            nn.Linear(16, 64),
            nn.LeakyReLU(negative_slope=0.01),  # Activation after linear layer
        )

        # Token reduction and upsampling layers
        self.token_reducer = nn.Sequential(
            nn.Linear(self.frames, num_reduced_tokens),
            nn.LeakyReLU(negative_slope=0.01)  # Activation for token reduction
        )
        self.token_upsampler = nn.Sequential(
            nn.Linear(num_reduced_tokens, self.frames),
            nn.LeakyReLU(negative_slope=0.01)  # Activation for token upsampling
        )

        #self.positional_embedding_enc = TemporalPositionalEmbedding(d_model=16, max_position=num_reduced_tokens)
        #self.positional_embedding_dec = TemporalPositionalEmbedding(d_model=16, max_position=num_reduced_tokens)
        self.transformer_enc2 = nn.Transformer(d_model=16, nhead=4, num_encoder_layers=2,
                                              num_decoder_layers=2, dim_feedforward=16,
                                              dropout=0.1).encoder

        self.transformer_dec2 = nn.Transformer(d_model=16, nhead=4, num_encoder_layers=2,
                                              num_decoder_layers=2, dim_feedforward=16,
                                              dropout=0.1).encoder
        self.bottleneck_reducer = nn.Sequential(
            nn.Linear(16 * num_reduced_tokens, dbottleneck),
            nn.LeakyReLU(negative_slope=0.01)
        ) # Reduce to bottleneck
        self.bottleneck_expander = nn.Sequential(
            nn.Linear(dbottleneck, 16 * num_reduced_tokens),
            nn.LeakyReLU(negative_slope=0.01)
        ) # Expand from bottleneck
        self.fc_mu = nn.Linear(dbottleneck, dbottleneck)
        self.fc_logvar = nn.Linear(dbottleneck, dbottleneck)

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
        cumulative_positions = self.compute_cumulative_positions(time_gaps)  # Shape: (batch_size, frames)

        # Add Positional embeddings & Pass through Transformer encoder
        pos_emb = self.positional_embedding_enc(cumulative_positions)  # Shape: (batch_size, frames, d2)
        pos_emb = pos_emb / torch.sqrt(torch.tensor(pos_emb.size(-1), dtype=torch.float))
        x = x + pos_emb  # Shape should match (frames, batch_size, d2)
        #print(x.shape)
        x = self.transformer_enc(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.encoder_linear(x)
        #print(x.shape)#"""
        x = self.token_reducer(x.permute(0, 2, 1))
        #latent_pos_emb = self.positional_embedding_enc(torch.arange(self.num_reduced_tokens, device=x.device)  # Shape: (num_reduced_tokens, d_model)).unsqueeze(0).expand(x.size(0), -1, -1).permute(1, 0, 2)
        #x = x.permute(2, 0, 1) + latent_pos_emb
        x = self.transformer_enc2(x.permute(2, 0, 1)).permute(1, 2, 0)
        x = x.reshape(x.size(0), -1)
        x = self.bottleneck_reducer(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        #print(x.shape)
        #return x

    def decode(self, x, time_gaps):
        x = self.bottleneck_expander(x)
        x = x.reshape(x.size(0), 16, 3)
        cumulative_positions = self.compute_cumulative_positions(time_gaps)  # Shape: (batch_size, frames)
        # latent_pos_emb = self.positional_embedding_dec2(torch.arange(self.num_reduced_tokens, device=x.device)).unsqueeze(0).expand(x.size(0), -1, -1).permute(1, 0, 2)
        # x = x.permute(2, 0, 1) + latent_pos_emb
        x = self.transformer_dec2(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.token_upsampler(x).permute(0, 2, 1)
        x = self.decoder_linear(x)
        #print(cumulative_positions)
        pos_emb = self.positional_embedding_dec(cumulative_positions).permute(1, 0, 2)  # Shape: (frames, batch_size, d2)
        pos_emb = pos_emb/ torch.sqrt(torch.tensor(pos_emb.size(-1), dtype=torch.float))
        #print(x.shape)
        #print(pos_emb.shape)
        x = x + pos_emb.permute(1, 0, 2)
        #print(x.shape)
        # Pass through the Transformer decoder
        x = self.transformer_dec(x)
        #print(x)

        #x = x.permute(1, 0, 2)  # Shape: (frames, frames, din)
        #print(x.shape)
        x = x.reshape(x.size(0), 4, self.frames, 4, 4)
        #print(x.shape)"""
        x = self.upscaler(x)  # Shape: (batch_size, frames, 15, 15, 209)
        #print(x.shape)
        return x

    def forward(self, x, time_gaps):
        encoded, mu, logvar = self.encode(x, time_gaps)
        decoded = self.decode(encoded, time_gaps)
        return decoded, encoded, mu, logvar


    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample z from N(mu, var) using N(0, 1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, output, target, mask, mean, logvar):
        """
        Compute the combined loss: reconstruction + KL divergence.
        `beta` is a weighting factor for the KL divergence term.
        """
        reconstruction_loss = self.loss_fn(output, target, mask)
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_div = kl_div / output.size(0)  # Normalize by batch size
        #beta = reconstruction_loss / (kl_div + 1e-7)
        beta = 0.1
        return reconstruction_loss + beta * kl_div, reconstruction_loss, kl_div

    def training_step(self, batch, batch_idx):
        x, time_gaps, mask = batch
        x = torch.clamp(x, min=-1e6, max=1e6)
        output, encoded, mu, logvar = self(x, time_gaps)
        #loss = self.loss_fn(output, x, mask)
        loss, recon_loss, kl_div = self.loss_function(output, x, mask, mu, logvar)
        self.log("train_loss", loss, prog_bar=True)
        self.log("kl_div", kl_div, prog_bar=True)
        self.log("recon_loss", recon_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, time_gaps, mask = batch
        x = torch.clamp(x, min=-1e6, max=1e6)
        output,encoded, mu, logvar = self(x, time_gaps)
        loss, recon_loss, kl_div = self.loss_function(output, x, mask, mu, logvar)

        # Log the metrics separately for better monitoring
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_recon_loss", recon_loss, prog_bar=True)
        self.log("val_kl_div", kl_div, prog_bar=True)

        return loss


    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-7)

        # Define a scheduler (e.g., ReduceLROnPlateau or LambdaLR)
        """scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",  # Minimize the monitored metric
            factor=0.1,  # Reduce LR by a factor of 0.1
            patience=3,  # Number of epochs with no improvement before reducing LR
            verbose=True,  # Print a message when LR is reduced
        )

        # Return optimizer and scheduler in the Lightning format
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Metric to monitor for ReduceLROnPlateau
                "interval": "epoch",  # Scheduler step frequency
                "frequency": 1,  # How often to call the scheduler
            },
        }"""
        return optimizer


def main():
    # Define model parameters
    dbottleneck = 16  # Bottleneck dimension
    frames = 11  # Number of frames we want to consider
    max_position = 50  # Maximum position index for embeddings
    batch_size = 2  # Batch size for dummy data

    # Instantiate the model
    model = TransformerAE()


    # Generate dummy input data
    # Input data shape: (batch_size, frames, 15, 15, 209)
    x = torch.randn(batch_size, frames, 15, 15, 209)


    # Generate dummy time gaps for each sample in the batch (shape: (batch_size, frames - 1))
    time_gaps = torch.tensor([
        [2, 5, 3, 2, 4, 2, 2, 4, 5, 3],  # Time gaps for the first sample in the batch
        [1, 3, 2, 4, 3, 5, 1, 2, 4, 3]  # Time gaps for the second sample in the batch
    ], dtype=torch.long)

    # Ensure the model is in evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        output = model(x, time_gaps)
        print('==============')
        #print(output.shape)

    # Output result
    #print("Input shape:", x.shape)
    #print("Time gaps shape:", time_gaps.shape)
    #print("Output shape:", output.shape)


if __name__ == "__main__":
    main()
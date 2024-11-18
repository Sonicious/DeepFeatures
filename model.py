import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from torchsummary import summary

class DimensionalityReducer(nn.Module):
    def __init__(self):
        super(DimensionalityReducer, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=209, out_channels=128, kernel_size=(1, 3, 3))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=(1, 3, 3))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=(1, 3, 3))
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # (batch_size, 209, 11, 15, 15)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x.permute(0, 2, 3, 4, 1)  # (batch_size, 11, 15, 15, 64)


class Upscaler(nn.Module):
    def __init__(self):
        super(Upscaler, self).__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels=32, out_channels=64, kernel_size=(1, 3, 3))
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose3d(in_channels=64, out_channels=128, kernel_size=(1, 3, 3))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose3d(in_channels=128, out_channels=209, kernel_size=(1, 3, 3))
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)  # (batch_size, 64, 11, 15, 15)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        return x.permute(0, 2, 3, 4, 1)  # (batch_size, 11, 15, 15, 209)


class TemporalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_position=50):
        super(TemporalPositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position + 1, d_model)

    def forward(self, cumulative_positions):
        embeddings = self.position_embeddings(cumulative_positions)  # Shape: (frames, d_model)
        return embeddings


class TransformerAE(pl.LightningModule):
    def __init__(self, d2=512, dbottleneck=16, frames=11, max_position=50,
                 loss_fn=nn.MSELoss(), learning_rate=1e-3):
        super(TransformerAE, self).__init__()

        self.frames = frames
        self.din = 9 * 9 * 32  # Flattened input dimension after reduction (15 * 15 spatial, 64 channels)

        # Dimensionality reduction and upscaling modules
        self.dim_reducer = DimensionalityReducer()
        self.upscaler = Upscaler()

        # Encoder linear layer
        self.encoder_linear = nn.Linear(self.din, d2)


        # Transformer encoder and decoder
        self.transformer = nn.Transformer(d_model=d2, nhead=16, num_encoder_layers=8,
                                          num_decoder_layers=8, dim_feedforward=d2)

        # Bottleneck linear layers
        self.encoder_intermidiate1 = nn.Linear(d2, d2 // 2)
        self.relu1 = nn.ReLU()
        self.encoder_intermidiate2 = nn.Linear(d2 // 2, d2 // 4)
        self.relu2 = nn.ReLU()
        self.encoder_intermidiate3 = nn.Linear(d2 // 4, d2 // 8)
        self.relu3 = nn.ReLU()
        self.encoder_intermidiate4 = nn.Linear(d2 // 8, d2 // 16)
        self.relu4 = nn.ReLU()
        self.encoder_bottleneck = nn.Linear(d2 // 16, dbottleneck)

        self.decoder_bottleneck = nn.Linear(dbottleneck, d2 // 16)
        self.relu5 = nn.ReLU()
        self.decoder_intermidiate1 = nn.Linear(d2 // 16, d2 // 8)
        self.relu6 = nn.ReLU()
        self.decoder_intermidiate2 = nn.Linear(d2 // 8, d2 // 4)
        self.relu7 = nn.ReLU()
        self.decoder_intermidiate3 = nn.Linear(d2 // 4, d2 // 2)
        self.relu8 = nn.ReLU()
        self.decoder_intermidiate4 = nn.Linear(d2 // 2, d2)

        # Output linear layer to map back to flattened input size
        self.output_linear = nn.Linear(d2, self.din)

        # Temporal positional embedding
        self.positional_embedding = TemporalPositionalEmbedding(d_model=d2, max_position=max_position)

        # Loss and learning rate
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

    def compute_cumulative_positions(self, time_gaps):
        # Initialize cumulative positions with zeros for each sample in the batch
        batch_size, frames_minus_one = time_gaps.size()
        cumulative_positions = torch.zeros((batch_size, self.frames), dtype=torch.long, device=time_gaps.device)
        #print(cumulative_positions.shape)

        # Fill cumulative positions by computing the cumulative sum for each batch sample
        cumulative_positions[:, 1:] = torch.cumsum(time_gaps, dim=1)
        # print(cumulative_positions)

        return cumulative_positions

    def encode(self, x, time_gaps):
        # Apply dimensionality reduction
        x = self.dim_reducer(x)  # Expected shape: (batch_size, frames, 15, 15, 64)

        # Flatten spatial dimensions for the Transformer input
        x = x.reshape(x.size(0), self.frames, -1)  # Expected shape: (batch_size, frames, din)
        print(x.shape)
        x = self.encoder_linear(x)  # Map flattened input to d2 dimension
        # print("lin encoder shape:", x.shape)

        # Calculate cumulative positions from time gaps
        cumulative_positions = self.compute_cumulative_positions(time_gaps)  # Shape: (batch_size, frames)
        # print("cumulative_positions shape:", cumulative_positions.shape)

        # Obtain positional embeddings without flattening
        pos_emb = self.positional_embedding(cumulative_positions)  # Shape: (batch_size, frames, d2)
        # print("pos_emb shape:", pos_emb.shape)

        # Add positional embeddings to the input
        x = x + pos_emb  # Shape should match (batch_size, frames, d2)
        # print("Shape after adding pos_emb:", x.shape)

        # Permute to (frames, batch_size, d2) for Transformer input
        x = x.permute(1, 0, 2)
        # print("Shape after permute:", x.shape)

        # Pass through Transformer encoder
        encoded = self.transformer.encoder(x)
        # print("Shape after transformer encoder:", encoded.shape)

        # Bottleneck layers
        encoded  = self.encoder_intermidiate1(encoded)
        # print("Shape after intermidiate encoder:", encoded.shape)
        encoded = self.relu1(encoded)
        encoded = self.encoder_intermidiate2(encoded)
        encoded = self.relu2(encoded)
        encoded = self.encoder_intermidiate3(encoded)
        encoded = self.relu3(encoded)
        encoded = self.encoder_intermidiate4(encoded)
        encoded = self.relu4(encoded)
        # print("Shape after intermidiate encoder:", encoded.shape)

        encoded = self.encoder_bottleneck(encoded)
        # print("Shape after bottleneck:", encoded.shape)
        return encoded

    def decode(self, encoded, time_gaps):
        # Calculate cumulative positions from time gaps
        encoded = self.decoder_bottleneck(encoded)
        encoded = self.relu5(encoded)
        encoded = self.decoder_intermidiate1(encoded)
        encoded = self.relu6(encoded)
        encoded = self.decoder_intermidiate2(encoded)
        encoded = self.relu7(encoded)
        encoded = self.decoder_intermidiate3(encoded)
        encoded = self.relu8(encoded)
        encoded = self.decoder_intermidiate4(encoded)

        #encoded = self.decoder_linear(encoded)

        # print("Shape after lin decoder:", encoded.shape)

        cumulative_positions = self.compute_cumulative_positions(time_gaps)  # Shape: (batch_size, frames)
        # print(cumulative_positions.shape)

        # Obtain positional embeddings in d2 dimension and map to dbottleneck
        pos_emb = self.positional_embedding(cumulative_positions)  # Shape: (batch_size, frames, d2)
        # print("pos_emb shape:", pos_emb.shape)
        #pos_emb = self.decoder_bottleneck(pos_emb)  # Map to shape (batch_size, frames, dbottleneck)
        pos_emb = pos_emb.permute(1, 0, 2)
        # print("pos_emb shape in decode after bottleneck:", pos_emb.shape)

        # Add positional embeddings to the encoded input
        encoded = encoded + pos_emb  # Shapes now match: (batch_size, frames, dbottleneck)

        # Pass through Transformer decoder (requires same dimensions)
        decoded = self.transformer.decoder(encoded, encoded)

        # Apply output layer to map back to the flattened input size
        decoded = self.output_linear(decoded)  # Map to (batch_size, frames, din)

        # Reshape back to original dimensions for upscaling
        decoded = decoded.permute(1, 0, 2)  # Shape: (batch_size, frames, din)
        decoded = decoded.view(decoded.size(0), self.frames, 9, 9, 32)  # (batch_size, frames, 15, 15, 64)

        # Upscale channels back to the original input dimension
        decoded = self.upscaler(decoded)  # Shape: (batch_size, frames, 15, 15, 209)
        # print("Shape after upscaler:", decoded.shape)
        return decoded

    def forward(self, x, time_gaps):
        encoded = self.encode(x, time_gaps)
        decoded = self.decode(encoded, time_gaps)
        return decoded

    # def training_step(self, batch, batch_idx):
    #     x, time_gaps = batch
    #     output = self(x, time_gaps)
    #     loss = self.loss_fn(output, x)
    #     self.log("train_loss", loss, prog_bar=True)
    #     return loss

    def training_step(self, batch, batch_idx):
        x, time_gaps = batch
        if torch.isnan(x).any() or torch.isnan(time_gaps).any():
            print("NaN detected in input data.")

        output = self(x, time_gaps)
        loss = self.loss_fn(output, x)

        if torch.isnan(loss):
            print("NaN detected in loss.")
            return loss  # Return early if loss is NaN to prevent further computation

    def validation_step(self, batch, batch_idx):
        x, time_gaps = batch
        output = self(x, time_gaps)
        loss = self.loss_fn(output, x)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


import torch


def main():
    # Define model parameters
    d2 = 512  # Hidden dimension
    dbottleneck = 16  # Bottleneck dimension
    frames = 11  # Number of frames we want to consider
    max_position = 50  # Maximum position index for embeddings
    batch_size = 2  # Batch size for dummy data

    # Instantiate the model
    model = TransformerAE(d2=d2, dbottleneck=dbottleneck, frames=frames, max_position=max_position)


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

    # Output result
    print("Input shape:", x.shape)
    print("Time gaps shape:", time_gaps.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()

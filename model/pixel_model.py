import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau


class Autoencoder(pl.LightningModule):
    def __init__(self, dbottleneck=7, learning_rate=1e-4, loss_fn=nn.MSELoss(), time_steps=11, feature_dim=12):
        super().__init__()
        self.save_hyperparameters()

        self.time_steps = time_steps
        self.feature_dim = feature_dim
        self.d_model = feature_dim  # using same dim for simplicity

        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

        self.encoder = nn.Sequential(
            nn.Linear(12, 16),
            nn.SELU(),
            nn.Linear(16, 32),
            nn.SELU(),
            nn.Linear(32, 64),
            nn.SELU(),
            nn.Linear(64, 128),
            nn.SELU(),
            nn.Linear(128, 64),
            nn.SELU(),
            nn.Linear(64, 16),
            nn.SELU(),
            nn.Linear(16, dbottleneck),
            nn.Softplus(beta=1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(dbottleneck, 16),
            nn.SELU(),
            nn.Linear(16, 64),
            nn.SELU(),
            nn.Linear(64, 128),
            nn.SELU(),
            nn.Linear(128, 64),
            nn.SELU(),
            nn.Linear(64, 32),
            nn.SELU(),
            nn.Linear(32, 16),
            nn.SELU(),
            nn.Linear(16, 12)
        )


    def forward(self, x):  # x: (batch, time, feature)
        # Add position encoding

        x = self.encoder_linear(x)   # (B, T, 16)

        # Token reduction (T=11 → T=4)
        x = x.permute(0, 2, 1)              # (B, 16, T)
        x = self.token_reducer(x)           # (B, 16, 4)
        x = x.permute(0, 2, 1)              # (B, 4, 16)

        # Flatten and bottleneck
        z = self.bottleneck_reducer(x)      # (B, bottleneck)

        # Decode
        z = self.bottleneck_expander(z)     # (B, 4*16)
        x = z.view(-1, 4, 16)               # (B, 4, 16)

        x = x.permute(0, 2, 1)              # (B, 16, 4)
        x = self.token_upsampler(x)         # (B, 16, 11)
        x = x.permute(0, 2, 1)              # (B, 11, 16)

        x = self.decoder_linear(x)          # (B, 11, D)

        return x                            # shape matches input

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_fn(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.loss_fn(x_hat, x)
        self.log("val_loss", loss)
        return loss


    def configure_optimizers(self):
        # Define the optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1.5e-4)

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
    model = Autoencoder(dbottleneck=7)

    # Generate dummy input data
    x = torch.randn(batch_size, 12)



    # Ensure the model is in evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        decoded, _ = model(x)
        print('==============')
        #print(output.shape)

    # Output result
    print("Input shape:", x.shape)
    print("Output shape:", decoded.shape)

    print(model.loss_fn(decoded, x))



if __name__ == "__main__":
    main()
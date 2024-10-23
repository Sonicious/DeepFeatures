import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as L  # Use the new import for Lightning


# Define the LightningModule for the Autoencoder
class LitAutoencoder(L.LightningModule):
    def __init__(self, input_dim=209, latent_dim=64, learning_rate=0.001):
        super(LitAutoencoder, self).__init__()
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )


    def forward(self, x):
        # Pass input through encoder
        latent = self.encoder(x)
        # Pass latent vector through decoder to reconstruct
        reconstructed = self.decoder(latent)
        return reconstructed

    def training_step(self, batch, batch_idx):
        # Training step defines how the model is trained
        x = batch  # Custom DataLoader should return batch in shape (batch_size, 209)
        x_reconstructed = self(x)
        loss = self.criterion(x_reconstructed, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step defines how the model is evaluated during validation
        x = batch
        x_reconstructed = self(x)
        loss = self.criterion(x_reconstructed, x)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        # Define the optimizer
        return optim.Adam(self.parameters(), lr=self.learning_rate)

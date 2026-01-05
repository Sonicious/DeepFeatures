import torch
import pickle
import random
import lightning.pytorch as L
from model.model import TransformerAE
from dataset.dataset import HDF5Dataset
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping

torch.set_float32_matmul_precision('medium')  # For better performance


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # Define the parameters
    batch_size = 16

    train_dataset = HDF5Dataset("train.h5")
    val_dataset = HDF5Dataset("val.h5")


    # Create DataLoaders
    val_iterator   = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=32)
    train_iterator = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    # Initialize the autoencoder model
    autoencoder = TransformerAE()
    autoencoder.to(device)  # Move the model to GPU 3

    # Define checkpointing (optional)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='ae-{epoch:02d}-{val_loss:.3e}',
        save_top_k=3,
        mode='min'
    )

    # Define early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=15,  # Number of epochs to wait for improvement
        mode='min',  # Stop if the metric stops decreasing
        verbose=True,  # Print early stopping message
    )

    # Initialize the PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=500,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,  # Ensure validation is run every epoch
        devices=[3],
        accelerator="gpu",
        callbacks=[checkpoint_callback, early_stopping_callback],
    )


    # Train the model using your custom train and validation iterators
    with torch.autograd.set_detect_anomaly(True):
        trainer.fit(autoencoder, train_dataloaders=train_iterator, val_dataloaders=val_iterator)

if __name__ == "__main__":
    main()

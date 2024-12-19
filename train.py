import torch
import pickle
import lightning.pytorch as L
from model import TransformerAE
from dataset import HDF5Dataset
from si_dataset import ds
from data_iterator import XrDataset
from ml4xcube.splits import assign_block_split
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from ml4xcube.preprocessing import get_statistics, get_range
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping


torch.set_float32_matmul_precision('medium')  # For better performance


def main():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    # Define the parameters
    batch_size = 24

    train_dataset = HDF5Dataset("train_data_11_15_15.h5")
    val_dataset = HDF5Dataset("val_data_11_15_15.h5")


    train_size = int(0.75 * len(train_dataset))
    remain_size = len(val_dataset) - train_size

    val_size = int(0.2 * len(val_dataset))
    test_size = len(val_dataset) - val_size

    # Randomly split into validation and test
    val_split, test_split = random_split(val_dataset, [val_size, test_size])
    train_split, _ = random_split(val_dataset, [train_size, remain_size])

    # Create DataLoaders
    val_iterator   = DataLoader(val_split, batch_size=batch_size, shuffle=True, num_workers=8)
    test_iterator  = DataLoader(test_split, batch_size=batch_size, shuffle=True, num_workers=8)
    train_iterator = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=8)

    # Initialize the autoencoder model
    autoencoder = TransformerAE()
    autoencoder.to(device)  # Move the model to GPU 3
    #autoencoder.reset_parameters()

    # Define checkpointing (optional)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='autoencoder-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    # Define early stopping
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=5,  # Number of epochs to wait for improvement
        mode='min',  # Stop if the metric stops decreasing
        verbose=True  # Print early stopping message
    )

    # Initialize the PyTorch Lightning trainer
    trainer = L.Trainer(
        max_epochs=100,
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

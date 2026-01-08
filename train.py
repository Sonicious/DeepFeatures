import torch
import pickle
import random
import numpy as np
import lightning.pytorch as L
from model.model import TransformerAE
from dataset.dataset import HDF5Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping

torch.set_float32_matmul_precision('medium')  # For better performance
gpu = 3


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def make_loader(dataset, batch_size, drop_frac=0.5, num_workers=32, pin_memory=True, shuffle=None):
    N = len(dataset)
    keep = int((1 - drop_frac) * N)
    indices = np.random.choice(N, keep, replace=False)

    sampler = SubsetRandomSampler(indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )

def main():
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    # Define the parameters

    train_dataset = HDF5Dataset("train_si_final.h5")
    val_dataset = HDF5Dataset("val_si_final.h5")


    # Create DataLoaders
    val_iterator   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=32)
    train_iterator = make_loader(train_dataset, batch_size=16, num_workers=32)

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
        devices=[gpu],
        accelerator="gpu",
        callbacks=[checkpoint_callback, early_stopping_callback],
    )


    # Train the model using your custom train and validation iterators
    with torch.autograd.set_detect_anomaly(True):
        trainer.fit(autoencoder, train_dataloaders=train_iterator, val_dataloaders=val_iterator)

if __name__ == "__main__":
    main()

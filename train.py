import torch
import pickle
import random
import lightning.pytorch as L
from model import TransformerAE
from dataset import HDF5Dataset
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.data import random_split
from ml4xcube.preprocessing import get_statistics, get_range
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
from initialize import hierarchical_initialize_weights

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
    train_dataset_heinich = HDF5Dataset("train_data_11_15_15.h5")
    val_dataset = HDF5Dataset("test.h5")
    val_dataset_heinich = HDF5Dataset("val_data_11_15_15.h5")

    # Randomly select samples from train_dataset_heinich
    train_indices = random.sample(range(len(train_dataset_heinich)), 4092)
    selected_train_subset = Subset(train_dataset_heinich, train_indices)

    # Randomly select samples from val_dataset_heinich
    val_indices = random.sample(range(len(val_dataset_heinich)), 1364)
    selected_val_subset = Subset(val_dataset_heinich, val_indices)

    # Concatenate the selected subsets with the original datasets
    combined_train_dataset = ConcatDataset([train_dataset, selected_train_subset])
    combined_val_dataset = ConcatDataset([val_dataset, selected_val_subset])


    #train_size = int(0.75 * len(train_dataset))
    #remain_size = len(val_dataset) - train_size

    #val_size = int(0.2 * len(val_dataset))
    #test_size = len(val_dataset) - val_size

    # Randomly split into validation and test
    #val_split, test_split = random_split(val_dataset, [val_size, test_size])
    #train_split, _ = random_split(val_dataset, [train_size, remain_size])

    # Create DataLoaders
    val_iterator   = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False, num_workers=32)
    #test_iterator  = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=32)
    train_iterator = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    # Initialize the autoencoder model
    autoencoder = TransformerAE()
    #autoencoder.apply(hierarchical_initialize_weights)
    autoencoder.to(device)  # Move the model to GPU 3
    #autoencoder.reset_parameters()

    # Define checkpointing (optional)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        #filename='foundation-ae-{epoch:02d}-{val_loss:.3e}',
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

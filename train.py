import xarray as xr
import lightning.pytorch as L
from si_dataset import ds
from model import LitAutoencoder
from data_iterator import XrDataset
from ml4xcube.splits import assign_block_split
from lightning.pytorch.callbacks import ModelCheckpoint

cube_path = r'/net/scratch/mreinhardt/testcube.zarr'
cube = xr.open_zarr(cube_path)
print(cube)


def main():
    # Open the Zarr data cube using xarray
    # Assume the Zarr store is structured with 'samples' as a dimension

    xds = ds#[['ARI', 'ARI2']]
    data_cube = assign_block_split(
        ds=xds,
        block_size=[("time", 11), ("y", 150), ("x", 150)],
        split=0.8
    )

    # Define the parameters
    batch_size = 512

    # Create the iterator
    train_iterator = XrDataset(data_cube, batch_size, batch_size)
    val_iterator   = XrDataset(data_cube, batch_size, batch_size,split_val=0.)

    # Iterate through the batches
    #for batch in val_iterator:
    #    print(f"Received batch with shape: {batch.shape}")

    # Initialize the autoencoder model
    autoencoder = LitAutoencoder(input_dim=209, latent_dim=32)

    # Define checkpointing (optional)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='autoencoder-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    # Initialize the PyTorch Lightning trainer
    trainer = L.Trainer(max_epochs=50, callbacks=[checkpoint_callback])

    # Train the model using your custom train and validation iterators
    trainer.fit(autoencoder, train_dataloaders=train_iterator, val_dataloaders=val_iterator)

if __name__ == "__main__":
    main()

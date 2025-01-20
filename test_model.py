import torch
from lightning.pytorch import Trainer
from model import TransformerAE
from dataset import HDF5Dataset
from torch.utils.data import DataLoader

import torch




def test_model():
    def filter_collate_fn(batch):
        """
        Custom collate function to filter out samples with time_gaps >= 6.
        Args:
            batch: A list of tuples (x, time_gaps, mask) from the dataset.
        Returns:
            Filtered batch with samples meeting the condition.
        """
        filtered_batch = []
        for x, time_gaps, mask in batch:
            if not torch.any(time_gaps >= 6):  # Exclude samples with time_gaps >= 6
                filtered_batch.append((x, time_gaps, mask))

        if not filtered_batch:
            raise ValueError("All samples in the batch were filtered out.")

        # Unpack the filtered batch into separate tensors
        x_filtered, time_gaps_filtered, mask_filtered = zip(*filtered_batch)

        return (
            torch.stack(x_filtered),
            torch.stack(time_gaps_filtered),
            torch.stack(mask_filtered),
        )

    # Path to the checkpoint file
    checkpoint_path = "./checkpoints/autoencoder-epoch=34-val_loss=0.00.ckpt"

    # Load the model from the checkpoint
    model = TransformerAE()

    # Ensure the model is in evaluation mode
    model.eval()

    # Define your data module or test dataset
    dataset = HDF5Dataset("val_data_11_15_15.h5")
    test_loader = DataLoader(dataset, batch_size=24, shuffle=False, num_workers=32, collate_fn=filter_collate_fn)  # Use the custom collate function)


    # Initialize the Trainer
    trainer = Trainer(
        devices=[3],  # Specify the GPU device ID
        accelerator="gpu",  # Use GPU for acceleration
        enable_progress_bar=True,  # Enable the progress bar
    )

    # Run the test loop
    test_results = trainer.test(
        model=model,  # No need to provide the model here
        dataloaders=test_loader,
        ckpt_path=checkpoint_path,
        verbose=True
    )


    # Print the test results
    print(test_results)

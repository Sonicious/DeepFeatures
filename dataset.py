import h5py
import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path):
        """
        PyTorch Dataset for HDF5 data.

        Args:
            h5_file_path (str): Path to the HDF5 file.
        """
        self.h5_file_path = h5_file_path
        # Open the HDF5 file in read mode
        self.h5_file = h5py.File(h5_file_path, "r")

        # Load dataset references
        self.data = self.h5_file["data"]
        self.time_gaps = self.h5_file["time_gaps"]
        self.mask = self.h5_file["mask"]

        # Dataset length is determined by the number of samples in the "data" dataset
        self.length = self.data.shape[0]

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.length

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple: (data, time_gaps, mask) where
                - data is a torch.Tensor of shape (209, 11, 15, 15),
                - time_gaps is a torch.Tensor of shape (10,),
                - mask is a torch.Tensor of the same shape as data (209, 11, 15, 15).
        """
        data = torch.tensor(self.data[index], dtype=torch.float32)
        time_gaps = torch.tensor(self.time_gaps[index], dtype=torch.int32)
        mask = torch.tensor(self.mask[index], dtype=torch.bool)

        return data, time_gaps, mask

    def __del__(self):
        """Ensure the HDF5 file is properly closed when the object is deleted."""
        self.h5_file.close()


# Usage Example:
if __name__ == "__main__":
    h5_dataset = HDF5Dataset("train_dataset.h5")

    # Example: Iterate over the dataset using DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(h5_dataset, batch_size=1, shuffle=False, num_workers=4)

    for i, (data, time_gaps, mask) in enumerate(dataloader):
        #print(f"Batch {i}:")
        #print(f"Data shape: {data.shape}")
        #print(f"Time gaps shape: {time_gaps.shape}")
        #print(f"Mask shape: {mask.shape}")
        # Check for NaNs in data
        if torch.isnan(data).any():
            print(f"NaNs detected in data for Batch {i}.")
            nan_indices = torch.isnan(data).nonzero(as_tuple=False)
            print(f"Indices of NaNs in Batch {i}: {nan_indices}")

            # Access the data at those indices
            nan_values = data[nan_indices.split(1, dim=1)]  # Unpacks indices to access the tensor
            print(f"NaN values at those indices: {nan_values}")
        else:
            print(f"No NaNs in data for Batch {i}.")
        print(data)
        break

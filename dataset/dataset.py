import h5py
import torch
from torch.utils.data import Dataset
from model.model import TransformerAE

class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path, return_coords=False, da=True):
        """
        PyTorch Dataset for HDF5 data.

        Args:
            h5_file_path (str): Path to the HDF5 file.
            return_coords (bool): If True, return coordinate arrays instead of time_gaps.
        """
        self.h5_file_path = h5_file_path
        self.return_coords = return_coords
        self.h5_file = h5py.File(h5_file_path, "r", swmr=True)

        # Always present
        self.data = self.h5_file["data"]
        self.time_gaps = self.h5_file["time_gaps"]
        self.mask = self.h5_file["mask"]
        self.da = da

        # Optional returns
        if return_coords:
            self.coord_time = self.h5_file["coord_time"]
            self.coord_x = self.h5_file["coord_x"]
            self.coord_y = self.h5_file["coord_y"]

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
            - If return_coords is False:
                tuple: (data, time_gaps, mask)
            - If return_coords is True:
                tuple: (data, mask, coord_time, coord_x, coord_y)
        """
        #print('=================================')
        #print(self.data[index][:, 5, 7, 7])
        if self.da:
            data = torch.tensor(self.data[index], dtype=torch.float32).permute(1, 0, 2, 3)
            mask = torch.tensor(self.mask[index], dtype=torch.bool).permute(1, 0, 2, 3)

        else:
            data = torch.tensor(self.data[index], dtype=torch.float32).permute(0, 3, 1, 2)
            mask = torch.tensor(self.mask[index], dtype=torch.bool).permute(0, 3, 1, 2)

        #print(data[5, :, 7, 7])
        #print('=================================')

        time_gaps = torch.tensor(self.time_gaps[index], dtype=torch.int32)



        # Get size of second dimension
        #dim_size = data.size(1)

        # Create indices excluding index 9
        #indices = torch.cat((torch.arange(9), torch.arange(10, dim_size)))

        # Remove index 9 from second dimension
        #data = torch.index_select(data, dim=1, index=indices)
        #mask = torch.index_select(mask, dim=1, index=indices)
        #data = data[:, :12, :, :]  # (batch, frames, 12, x, y)
        #mask = mask[:, :12, :, :]  # (batch, frames, 12, x, y)

        #keep_indices = [i for i in range(12) if i != 9]
        #keep_indices = [i for i in range(12) if i not in [0,9]]
#
        #data = data[:, keep_indices, :, :]
        #mask = mask[:, keep_indices, :, :]
        if self.return_coords:
            coord_time = torch.tensor(self.coord_time[index], dtype=torch.int64)
            coord_x = torch.tensor(self.coord_x[index], dtype=torch.float32)
            coord_y = torch.tensor(self.coord_y[index], dtype=torch.float32)

            return data, time_gaps, mask, coord_time, coord_x, coord_y
        else:
            return data, time_gaps, mask

    def __del__(self):
        """Ensure the HDF5 file is properly closed when the object is deleted."""
        self.h5_file.close()


# Usage Example:
if __name__ == "__main__":
    #h5_dataset_da = HDF5Dataset("./train_149.h5")
    #h5_dataset = HDF5Dataset("./ds_test_149.h5", da=False)
    h5_dataset = HDF5Dataset("../train_val_s1_s2.h5")
    print(len(h5_dataset))#42843 #46297
    model = TransformerAE(dbottleneck=7)

    # Example: Iterate over the dataset using DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(h5_dataset, batch_size=1, shuffle=False)#, num_workers=4)

    for i, (data, time_gaps, mask) in enumerate(dataloader):

        # Remove batch dimension since batch_size=1
        data = data.squeeze(0)  # shape: (frames, channels, y, x)
        mask = mask.squeeze(0)  # shape: (frames, channels, y, x)
        time_gaps = time_gaps.squeeze(0)

        #print(f"  data   -> min: {data.min().item():.4f}, max: {data.max().item():.4f}")
        print(time_gaps)
        #print(mask.shape)
        #print(data.shape)
        print('======================')
        #break
        # Compute the sum
        total = time_gaps.sum()

        # Check if larger than 200
        if total > 350:
            print(f"Sum = {total.item()} (larger than 200)")
            break
        else:
            print(f"Sum = {total.item()} (not larger than 200)")


    #dataloader = DataLoader(h5_dataset_da, batch_size=1, shuffle=False)#, num_workers=4)
#
    #for i, (data, time_gaps, mask) in enumerate(dataloader):
    #    print(data.shape)
#
    #    # Remove batch dimension since batch_size=1
    #    data = data.squeeze(0)  # shape: (frames, channels, y, x)
    #    mask = mask.squeeze(0)  # shape: (frames, channels, y, x)
    #    time_gaps = time_gaps.squeeze(0)
#
    #    #print(f"  data   -> min: {data.min().item():.4f}, max: {data.max().item():.4f}")
    #    print(time_gaps.shape)
    #    print(mask.shape)
    #    print(data.shape)
    #    break


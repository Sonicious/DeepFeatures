import torch
import random
import numpy as np
import xarray as xr
from time import time
from typing import Dict
from typing import Tuple
from si_dataset import ds
from ml4xcube.utils import split_chunk
from ml4xcube.utils import get_chunk_sizes
from ml4xcube.utils import get_chunk_by_index
from ml4xcube.utils import calculate_total_chunks
from multiprocessing import Pool
from ml4xcube.splits import assign_block_split
#from concurrent.futures import ProcessPoolExecutor
from ml4xcube.preprocessing import standardize, normalize
from ml4xcube.preprocessing import drop_nan_values
from ml4xcube.preprocessing import fill_nan_values
from pathos.multiprocessing import ProcessingPool as Pool



def compute_time_gaps(time_coords: np.ndarray) -> torch.Tensor:
    """Calculate time gaps between consecutive timestamps."""
    time_deltas = np.diff(time_coords).astype('timedelta64[D]').astype(int)
    return torch.tensor(time_deltas)


def worker_preprocess_chunk(args): # process_samples):
    ds_obj, idx, stats = args
    chunk, coords = get_chunk_by_index(ds_obj.data_cube, idx)
    mask = chunk['split'] == ds_obj.split_val

    #print(type(chunk['split']))
    #print(chunk['split'])

    cf = {var: np.ma.masked_where(~mask, chunk[var]).filled(np.nan) for var in chunk if var != 'split'}
    # print('split chunk')
    cf, coords = split_chunk(cf, coords, sample_size=ds_obj.sample_size, overlap=ds_obj.overlap)
    vars = list(cf.keys())
    # print('drop nan values')
    cf, coords = drop_nan_values(cf, coords, mode='if_all_nan', vars=vars)
    # print('fill nan values')
    cf = fill_nan_values(cf, vars=vars, method='sample_mean')
    cf = standardize(cf, stats)
    return cf, coords


class XrDataset:
    def __init__(self, data_cube, batch_size, statistics = None, process_batch = None, split_val = 1., overlap = None, sample_size = None):
        """
        Args:
            data_cube: The giant data cube (numpy array or another large data structure).
            batch_size: Number of samples per batch.
            process_samples: Function to process samples.
        """
        self.data_cube = data_cube
        self.block_size = get_chunk_sizes(data_cube)
        self.batch_size = batch_size
        self.current_chunk = None
        self.nproc = 6
        self.num_chunks = 6

        self.statistics = statistics

        # Calculate number of chunks
        self.total_chunks = calculate_total_chunks(self.data_cube)
       # self.chunk_idx_list = list(range(self.total_chunks))

       # random.shuffle(self.chunk_idx_list)
       # print(self.chunk_idx_list)
        self.chunk_idx_list = [20, 43, 12, 5, 13, 30, 32, 33, 21, 46, 34, 36, 26, 22, 17, 25, 27, 40, 19, 14, 38, 55, 16, 37, 4, 8, 9, 18, 2, 54, 53, 51, 28, 47, 6, 42, 45, 29, 52, 35, 7, 0, 48, 39, 11, 50, 1, 15, 49, 44, 41, 10, 31, 24, 3, 23]
        #self.chunk_idx_list = [20, 43, 12, 5, 13, 30, ]#32, 33, 21, 46, 34, 36]

        self.chunk_idx = 0
        self.process_batch = process_batch
        self.sample_size = sample_size
        self.overlap = overlap
        self.split_val = split_val
        self.current_chunks = None
        self.coords = None

        self.chunk_position = 0
        self.remaining_data = None
        self.remaining_coords = None


    def concatenate(self, chunks, coords) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Concatenate the chunks along the time dimension.

        Returns:
            Dict[str, np.ndarray]: A dictionary of concatenated data chunks.
        """
        concatenated_chunks = {}
        concatenated_coords = {}
        # Get the keys of the first dictionary in self.chunks
        keys = list(chunks[0].keys())
        coord_keys = list(coords[0].keys())

        #for chunk in chunks:
        #    print(chunk['ARI'].shape)

        # Loop over the keys and concatenate the arrays along the time dimension
        for key in keys:
            if key == 'split': continue
            concatenated_chunks[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)

        for key in coord_keys:
            concatenated_coords[key] = np.concatenate([coord[key] for coord in coords], axis=0)

        stacked_data = np.stack([concatenated_chunks[var_name] for var_name in concatenated_chunks], axis=-1)
        stacked_coords = {coord_key: concatenated_coords[coord_key] for coord_key in concatenated_coords}

        #print(self.remain)
        if self.remaining_data is not None:
            stacked_data = np.concatenate([self.remaining_data, stacked_data], axis=0)

            # Clear remaining data after concatenation
            self.remaining_data = None

        if self.remaining_coords is not None:
            for coord_key in stacked_coords.keys():
                stacked_coords[coord_key] = np.concatenate(
                    [self.remaining_coords[coord_key], stacked_coords[coord_key]], axis=0)
            self.remaining_coords = None

            # Shuffle the rows of the stacked data and coordinates
        idx = np.random.permutation(stacked_data.shape[0])  # Generate a random permutation of indices
        stacked_data = stacked_data[idx, :]  # Shuffle along the sample axis (axis=0)
        for coord_key in stacked_coords:
            stacked_coords[coord_key] = stacked_coords[coord_key][idx, :]

        return stacked_data, stacked_coords


    def load_chunk(self):
        """Load a random chunk of data."""
        start = time()

        batch_indices = self.chunk_idx_list[self.chunk_idx:self.chunk_idx + self.num_chunks]
        bi_time = time()
        # print(f'chunk indexes received after {bi_time - start} seconds')
        if not batch_indices:
            raise StopIteration("No more chunks to load. All samples have been processed.")
        with Pool(processes=self.nproc) as pool:
            mapped_results = pool.map(worker_preprocess_chunk, [
                (self, idx, self.statistics)
                for idx in batch_indices
            ])

        # Separate processed_chunks and coords from mapped_results
        processed_chunks, coords = zip(*mapped_results)  # Unzip the list of tuples

        pc_time = time()
        # print(f'chunks processed in {pc_time - bi_time} seconds')
        self.chunk_idx += self.num_chunks
        self.current_chunks, self.coords = self.concatenate(processed_chunks, coords)
        cc_time = time()
        # print(f'chunks concatenated in {cc_time - pc_time} seconds')
        self.chunk_position = 0  # Reset position in the concatenated chunks

    def __iter__(self):
        return self

    def __next__(self):
        """Return the next batch."""

        # Check if current chunk needs to be loaded or concatenated with remaining data
        if self.current_chunks is None or self.current_chunks.shape[0] - self.chunk_position < self.batch_size:
            # Save the remaining data (if any) before loading the next chunk
            if self.current_chunks is not None and self.current_chunks.shape[0] - self.chunk_position > 0:
                self.remaining_data = self.current_chunks[self.chunk_position:]  # Save remaining samples
                self.remaining_coords = {key: self.coords[key][self.chunk_position:] for key in self.coords.keys()}

            self.load_chunk()

        # Select a batch of batch_size
        end_position = min(self.chunk_position + self.batch_size, self.current_chunks.shape[0])

        if end_position <= self.chunk_position:
            raise StopIteration  # No more data left

        # Extract the batch from the current chunks
        batch = self.current_chunks[self.chunk_position:end_position, :]  # Shape: (batch_size, n_features)
        batch_coords = {key: self.coords[key][self.chunk_position:end_position] for key in self.coords.keys()}

        # Compute time gaps for the current batch
        time_coords = batch_coords['time']
        if len(time_coords) > 1:
            time_deltas = np.diff(time_coords.astype('datetime64[D]')).astype(int)
            time_gaps = torch.tensor(time_deltas, dtype=torch.int32)  # Ensure tensor format
        else:
            time_gaps = torch.empty((0,), dtype=torch.int32)  # Empty tensor if not enough time points

        # Move the position forward
        self.chunk_position = end_position

        # Return the selected batch
        return torch.from_numpy(batch), time_gaps


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
    batch_size = 8

    # Create the iterator
    data_iterator = XrDataset(data_cube, batch_size, sample_size=[("time", 11), ("y", 15), ("x", 15)], split_val=0.0)

    # Iterate through the batches
    for batch in data_iterator:
        #pass
        #data, coords, time_gaps = batch
        #print(f"Received batch with shape: {data.shape}")
        #print(f'Time coordinates received: {coords['time'][0]}')
        #print(f'Lat coordinates received: {coords['x'][0]}')
        #print(f'Lon coordinates received: {coords['y'][0]}')
        #print(f"Computed time gaps (torch tensor): {time_gaps}")
        data, time_gaps = batch
        #print(f"Received batch with shape: {data.shape}")

        # Check for NaNs in the batch
        if torch.isnan(data).any():
            print("Warning: NaN values detected in the batch.")
            break
        else:
            print("No NaN values in the batch.")

        # Print other details if needed
        #print(f"Computed time gaps (torch tensor): {time_gaps}")



if __name__ == "__main__":
    main()

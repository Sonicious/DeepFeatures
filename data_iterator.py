import random
import numpy as np
from time import time
from typing import Dict
from si_dataset import ds
from multiprocessing import Pool
from multiprocessing import Process
from ml4xcube.splits import assign_block_split
#from concurrent.futures import ProcessPoolExecutor
from ml4xcube.preprocessing import drop_nan_values, fill_nan_values
from ml4xcube.utils import get_chunk_by_index, get_chunk_sizes, split_chunk, calculate_total_chunks
from pathos.multiprocessing import ProcessingPool as Pool




def worker_preprocess_chunk(args): # process_samples):
    ds_obj, idx = args
    chunk = get_chunk_by_index(ds_obj.data_cube, idx)
    mask = chunk['split'] == ds_obj.split_val

    cf = {var: np.ma.masked_where(~mask, chunk[var]).filled(np.nan) for var in chunk if var != 'split'}
    cf = split_chunk(cf, sample_size=ds_obj.sample_size, overlap=ds_obj.overlap)
    vars = list(cf.keys())
    cf = drop_nan_values(cf, mode='if_all_nan', vars=vars)
    cf = fill_nan_values(cf, vars=vars, method='sample_mean')
    return cf


class XrDataset:
    def __init__(self, data_cube, batch_size, process_batch, split_val = 1., overlap = None, sample_size = None):
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

        # Calculate number of chunks
        self.total_chunks = calculate_total_chunks(self.data_cube)
        self.chunk_idx_list = list(range(self.total_chunks))
        random.shuffle(self.chunk_idx_list)

        self.chunk_idx = 0
        self.process_batch = process_batch
        self.sample_size = sample_size
        self.overlap = overlap
        self.split_val = split_val
        self.current_chunks = None

        self.chunk_position = 0
        self.remaining_data = None


    def concatenate_chunks(self, chunks) -> np.ndarray:
        """
        Concatenate the chunks along the time dimension.

        Returns:
            Dict[str, np.ndarray]: A dictionary of concatenated data chunks.
        """
        concatenated_chunks = {}
        # Get the keys of the first dictionary in self.chunks
        keys = list(chunks[0].keys())

        # Loop over the keys and concatenate the arrays along the time dimension
        for key in keys:
            if key == 'split': continue
            concatenated_chunks[key] = np.concatenate([chunk[key] for chunk in chunks], axis=0)

        stacked_data = np.stack([concatenated_chunks[var_name] for var_name in concatenated_chunks], axis=-1)

        print(stacked_data.shape)
        #print(self.remain)
        if self.remaining_data is not None:
            stacked_data = np.concatenate([self.remaining_data, stacked_data], axis=0)

            # Clear remaining data after concatenation
            self.remaining_data = None

        # Shuffle the rows of the stacked data (shuffle along axis=1 for samples)
        # Shuffle only along the sample dimension
        idx = np.random.permutation(stacked_data.shape[0])  # Generate a random permutation of indices
        stacked_data = stacked_data[idx, :]  # Shuffle along the sample axis (axis=1)

        return stacked_data

    def load_chunk(self):
        """Load a random chunk of data."""
        start = time()
        processed_chunks = list()

        #
        batch_indices = self.chunk_idx_list[self.chunk_idx:self.chunk_idx + self.num_chunks]
        bi_time = time()
        print(f'chunk indexes received after {bi_time - start} seconds')
        if not batch_indices:
            raise StopIteration("No more chunks to load. All samples have been processed.")
        with Pool(processes=self.nproc) as pool:
            processed_chunks = pool.map(worker_preprocess_chunk, [
                (self, idx)
                for idx in batch_indices
            ])
        pc_time = time()
        print(f'chunks processed in {pc_time - bi_time} seconds')
        self.chunk_idx += self.num_chunks
        self.current_chunks = self.concatenate_chunks(processed_chunks)
        cc_time = time()
        print(f'chunks concatenated in {cc_time - pc_time} seconds')
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

            self.load_chunk()

        # Select a batch of batch_size
        end_position = min(self.chunk_position + self.batch_size, self.current_chunks.shape[0])

        if end_position <= self.chunk_position:
            raise StopIteration  # No more data left

        # Extract the batch from the current chunks
        batch = self.current_chunks[self.chunk_position:end_position, :]  # Shape: (batch_size, n_features)

        # Move the position forward
        self.chunk_position = end_position

        # Return the selected batch
        return batch


def main():
    # Open the Zarr data cube using xarray
    # Assume the Zarr store is structured with 'samples' as a dimension

    xds = ds#[['ARI', 'ARI2']]

    #for var_name in xds.data_vars:
    #    print(var_name)

    data_cube = assign_block_split(
        ds=xds,
        block_size=[("time", 11), ("y", 150), ("x", 150)],
        split=0.8
    )

    # Define the parameters
    chunk_size = 1000
    batch_size = 64

    # Create the iterator
    data_iterator = XrDataset(data_cube, chunk_size, batch_size)

    # Iterate through the batches
    for batch in data_iterator:
        pass
        #print(f"Received batch with shape: {batch.shape}")


if __name__ == "__main__":
    main()

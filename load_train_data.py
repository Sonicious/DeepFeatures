import h5py
import torch
import pickle
import numpy as np
import dask.array as da
from time import time
from typing import Dict, Tuple, List
from si_dataset import ds
from ml4xcube.utils import split_chunk, get_chunk_sizes, get_chunk_by_index, calculate_total_chunks
from ml4xcube.splits import assign_block_split
from global_land_mask import globe
from ml4xcube.preprocessing import standardize, get_median
from ml4xcube.preprocessing import assign_mask, apply_filter
from ml4xcube.preprocessing import  drop_nan_values, fill_nan_values, drop_inf_values
from pathos.multiprocessing import ProcessingPool as Pool


def drop_if_central_point_nan_or_inf(
        ds: Dict[str, np.ndarray],
        coords: Dict[str, np.ndarray],
        vars: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Drop samples entirely if the central point of any variable contains NaN or inf values.

    Args:
        ds (Dict[str, np.ndarray]): The dataset to filter. It should be a dictionary where keys are variable names and values are numpy arrays.
        coords (Dict[str, np.ndarray]): The coordinates associated with the dataset.
        vars (List[str]): The variables to check for NaN or inf values at their central point.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: The filtered dataset and coordinates.
    """
    valid_mask_lists = []

    for var in vars:
        if var not in ds:
            continue
        value = ds[var]

        # Determine the central point index
        central_index = tuple(dim_size // 2 for dim_size in value.shape[1:])

        if value.ndim == 1:  # For 1D arrays, the central point is the single value
            valid_mask = ~np.isnan(value) & ~np.isinf(value)
        elif value.ndim >= 2:  # For multi-dimensional arrays
            central_values = value[(slice(None),) + central_index]  # Extract central point along non-batch axes
            valid_mask = ~np.isnan(central_values) & ~np.isinf(central_values)
        else:
            raise ValueError("Unsupported number of dimensions for the variable.")

        valid_mask_lists.append(valid_mask)

    # Combine masks across all variables using logical AND
    valid_mask = np.all(valid_mask_lists, axis=0)

    # Filter the dataset and coordinates
    ds = {key: ds[key][valid_mask] for key in ds.keys()}
    coords = {key: coords[key][valid_mask] for key in coords.keys()}

    return ds, coords




def compute_time_gaps(time_coords: np.ndarray) -> torch.Tensor:
    """Calculate time gaps between consecutive timestamps."""
    time_deltas = np.diff(time_coords).astype('timedelta64[D]').astype(int)
    return torch.tensor(time_deltas)


def worker_preprocess_chunk(args):
    ds_obj, idx, stats = args
    chunk, coords = get_chunk_by_index(ds_obj.data_cube, idx)

    # Separate masks for train and validation splits
    mask_train = chunk['split'] == 1.0
    mask_val = chunk['split'] == 0.0

    # Split chunks into train and validation
    cf_train = {var: np.ma.masked_where(~mask_train, chunk[var]).filled(np.nan) for var in chunk if var != 'split'}
    cf_val = {var: np.ma.masked_where(~mask_val, chunk[var]).filled(np.nan) for var in chunk if var != 'split'}

    # Process train data
    print('split train chunk')
    cf_train, coords_train = split_chunk(cf_train, coords, sample_size=ds_obj.sample_size, overlap=ds_obj.overlap)
    vars_train = list(cf_train.keys())
    print('drop nan values from train chunk')
    cf_train, coords_train = drop_nan_values(cf_train, coords_train, mode='if_all_nan', vars=vars_train)
    print('drop inf values from train chunk')
    cf_train, coords_train = drop_inf_values(cf_train, coords_train, vars=vars_train)
    print('drop if central point is not numeric in train chunk')
    cf_train, coords_train = drop_if_central_point_nan_or_inf(cf_train, coords_train, vars=vars_train)
    print('create train validity mask')
    valid_train = {var: ~np.isnan(cf_train[var]) for var in vars_train}
    print('fill nan values in train chunk')
    cf_train = fill_nan_values(cf_train, vars=vars_train, method='sample_mean')
    for var in cf_train:
        cnt = np.isnan(cf_train[var]).sum()
        if cnt > 0: print(f"NaNs in train variable '{var}' after filling: {cnt}")
    print('standardizing train chunk')
    cf_train = standardize(cf_train, stats)
    for var in cf_train:
        cnt = np.isnan(cf_train[var]).sum()
        if cnt > 0: print(f"NaNs in train variable '{var}' after standardization: {cnt}")

    # Process validation data
    print('split validation chunk')
    cf_val, coords_val = split_chunk(cf_val, coords, sample_size=ds_obj.sample_size, overlap=ds_obj.overlap)
    vars_val = list(cf_val.keys())
    print('drop nan values from validation chunk')
    cf_val, coords_val = drop_nan_values(cf_val, coords_val, mode='if_all_nan', vars=vars_val)
    print('drop inf values from validation chunk')
    cf_val, coords_val = drop_inf_values(cf_val, coords_val, vars=vars_val)
    print('drop if central point is not numeric in train chunk')
    cf_val, coords_val = drop_if_central_point_nan_or_inf(cf_val, coords_val, vars=vars_val)
    print('create validation validity mask')
    valid_val = {var: ~np.isnan(cf_val[var]) for var in vars_val}
    print('fill nan values in validation chunk')
    cf_val = fill_nan_values(cf_val, vars=vars_val, method='sample_mean')
    for var in cf_val:
        cnt = np.isnan(cf_val[var]).sum()
        if cnt > 0: print(f"NaNs in validation variable '{var}' after filling: {cnt}")
    print('standardizing validation chunk')
    cf_val = standardize(cf_val, stats)
    for var in cf_val:
        cnt = np.isnan(cf_val[var]).sum()
        if cnt > 0: print(f"NaNs in validation variable '{var}' after standardization: {cnt}")

    return cf_train, coords_train, valid_train, cf_val, coords_val, valid_val




class XrDataset:
    def __init__(self, data_cube, statistics = None, process_batch = None, overlap = None, sample_size = None):
        """
        Args:
            data_cube: The giant data cube (numpy array or another large data structure).
            batch_size: Number of samples per batch.
            process_samples: Function to process samples.
        """
        self.data_cube = data_cube
        self.block_size = get_chunk_sizes(data_cube)
        self.current_chunk = None
        self.nproc = 7

        self.statistics = statistics

        # Calculate number of chunks
        self.total_chunks = calculate_total_chunks(self.data_cube)
       # self.chunk_idx_list = list(range(self.total_chunks))

       # random.shuffle(self.chunk_idx_list)
       # print(self.chunk_idx_list)
        self.chunk_idx_list = [20, 43, 12, 5, 13, 30, 32, 33, 21, 46, 34, 36, 26, 22, 17, 25, 27, 40, 19, 14, 38, 55, 16, 37, 4, 8, 9, 18, 2, 54, 53, 51, 28, 47, 6, 42, 45, 29, 52, 35, 7, 0, 48, 39, 11, 50, 1, 15, 49, 44, 41, 10, 31, 24, 3, 23]
        self.num_chunks = len(self.chunk_idx_list) // 4

        #self.chunk_idx_list = [20, 43, 12, 5, 13, 30, ]#32, 33, 21, 46, 34, 36]

        self.chunk_idx = 0
        self.process_batch = process_batch
        self.sample_size = sample_size
        self.overlap = overlap
        self.current_train_chunks = None
        self.current_val_chunks = None
        self.train_coords = None
        self.val_coords = None
        self.valid_train = None
        self.valid_val = None

        self.chunk_position = 0
        self.remaining_data = None
        self.remaining_coords = None

    def concatenate(self, chunks, coords, masks) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Concatenate the chunks along the time dimension.

        Returns:
            Tuple:
                - Stacked data array
                - Concatenated coordinates
                - Stacked masks array
        """
        # Preallocate memory-efficient structures
        chunk_keys = [key for key in chunks[0] if key != 'split']
        coord_keys = list(coords[0].keys())

        # Concatenate data chunks directly into arrays
        stacked_data = np.concatenate(
            [np.stack([chunk[key] for key in chunk_keys], axis=-1) for chunk in chunks],
            axis=0
        )

        # Concatenate masks directly into arrays
        stacked_masks = np.concatenate(
            [np.stack([mask[key] for key in chunk_keys], axis=-1) for mask in masks],
            axis=0
        )

        # Concatenate coordinates
        concatenated_coords = {key: np.concatenate([coord[key] for coord in coords], axis=0) for key in coord_keys}

        # Shuffle in-place to save memory
        idx = np.random.permutation(stacked_data.shape[0])
        stacked_data = stacked_data[idx]  # Shuffle data
        stacked_masks = stacked_masks[idx]  # Shuffle masks
        for key in concatenated_coords:
            concatenated_coords[key] = concatenated_coords[key][idx]  # Shuffle coordinates

        return stacked_data, concatenated_coords, stacked_masks


    def load_chunk(self):
        """Load a random chunk of data."""
        start = time()

        #batch_indices = self.chunk_idx_list
        batch_indices = self.chunk_idx_list[self.chunk_idx:self.chunk_idx + self.num_chunks]
        bi_time = time()
        print(f'chunk indexes received after {bi_time - start} seconds')
        print(batch_indices)
        if not batch_indices:
            raise StopIteration("No more chunks to load. All samples have been processed.")
        with Pool(processes=self.nproc) as pool:
            mapped_results = pool.map(worker_preprocess_chunk, [
                (self, idx, self.statistics)
                for idx in batch_indices
            ])

        # Separate processed_chunks and coords from mapped_results
        cf_train, coords_train, valid_train, cf_val, coords_val, valid_val = zip(*mapped_results)

        pc_time = time()
        print(f'chunks processed in {pc_time - bi_time} seconds')
        self.chunk_idx += self.num_chunks
        self.current_train_chunks, self.train_coords, self.valid_train = self.concatenate(cf_train, coords_train, valid_train)
        self.current_val_chunks, self.val_coords, self.valid_val = self.concatenate(cf_val, coords_val, valid_val)

        cc_time = time()
        print(f'chunks concatenated in {cc_time - pc_time} seconds')

    def __iter__(self):
        return self

    def compute_time_gaps(self, time_coords):
        """
        Helper method to compute time gaps for a given set of time coordinates.
        Args:
            time_coords (np.ndarray): Array of time coordinates.

        Returns:
            torch.Tensor: Tensor of time gaps between consecutive timestamps.
        """
        if len(time_coords) > 1:
            time_deltas = np.diff(time_coords.astype('datetime64[D]')).astype(int)
            time_gaps = torch.tensor(time_deltas, dtype=torch.int32)  # Ensure tensor format
        else:
            time_gaps = torch.empty((0,), dtype=torch.int32)  # Empty tensor if not enough time points
        return time_gaps

    def reset(self):
        # Set original attributes to None
        self.current_train_chunks = None
        self.train_coords = None
        self.valid_train = None

        self.current_val_chunks = None
        self.val_coords = None
        self.valid_val = None

    def __next__(self):
        """Return the next batch."""

        # Check if current chunk needs to be loaded or concatenated with remaining data
        # Save the remaining data (if any) before loading the next chunk
        if self.current_train_chunks is None and self.current_val_chunks is None:
            self.load_chunk()

        # Compute time gaps for train and validation
        train_time_gaps = self.compute_time_gaps(self.train_coords['time'])
        val_time_gaps = self.compute_time_gaps(self.val_coords['time'])

        print('created megachunk !!')

        return self.current_train_chunks, self.train_coords, train_time_gaps, self.valid_train, self.current_val_chunks, self.val_coords, val_time_gaps, self.valid_val



# 5_9_9
def main():
    data_cube = assign_block_split(
        ds=ds,
        block_size=[("time", 20), ("y", 195), ("x", 195)],
        split=0.7
    )

    with open('stats.pkl', "rb") as f:
        stats = pickle.load(f)


    print('obtain data iterator')
    data_iterator = XrDataset(data_cube, stats, sample_size=[("time", 11), ("y", 15), ("x", 15)])

    with h5py.File("train_data_11_15_15.h5", "w") as train_file, h5py.File("val_data_11_15_15.h5", "w") as val_file:
        # Initialize expandable datasets for train
        example_batch = next(iter(data_iterator))
        train_shape = example_batch[0].shape[1:]  # Train data shape (e.g., (209, 11, 15, 15))
        val_shape = example_batch[4].shape[1:]    # Validation data shape (e.g., (209, 11, 15, 15))
        mask_shape = train_shape  # Masks have the same shape as the data
        coord_time_shape = example_batch[1]["time"].shape[1:]  # (11,)
        coord_x_shape = example_batch[1]["x"].shape[1:]       # (15,)
        coord_y_shape = example_batch[1]["y"].shape[1:]       # (15,)
        time_gap_shape = example_batch[2].shape[1:]           # (10,)

        # Train datasets
        train_data_dset = train_file.create_dataset(
            "data", shape=(0, *train_shape), maxshape=(None, *train_shape), dtype='float32', chunks=True
        )
        train_mask_dset = train_file.create_dataset(
            "mask", shape=(0, *mask_shape), maxshape=(None, *mask_shape), dtype='bool', chunks=True
        )
        train_coord_time_dset = train_file.create_dataset(
            "coord_time", shape=(0, *coord_time_shape), maxshape=(None, *coord_time_shape), dtype='int64', chunks=True
        )
        train_coord_x_dset = train_file.create_dataset(
            "coord_x", shape=(0, *coord_x_shape), maxshape=(None, *coord_x_shape), dtype='float32', chunks=True
        )
        train_coord_y_dset = train_file.create_dataset(
            "coord_y", shape=(0, *coord_y_shape), maxshape=(None, *coord_y_shape), dtype='float32', chunks=True
        )
        train_time_gaps_dset = train_file.create_dataset(
            "time_gaps", shape=(0, *time_gap_shape), maxshape=(None, *time_gap_shape), dtype='int32', chunks=True
        )

        # Validation datasets
        val_data_dset = val_file.create_dataset(
            "data", shape=(0, *val_shape), maxshape=(None, *val_shape), dtype='float32', chunks=True
        )
        val_mask_dset = val_file.create_dataset(
            "mask", shape=(0, *mask_shape), maxshape=(None, *mask_shape), dtype='bool', chunks=True
        )
        val_coord_time_dset = val_file.create_dataset(
            "coord_time", shape=(0, *coord_time_shape), maxshape=(None, *coord_time_shape), dtype='int64', chunks=True
        )
        val_coord_x_dset = val_file.create_dataset(
            "coord_x", shape=(0, *coord_x_shape), maxshape=(None, *coord_x_shape), dtype='float32', chunks=True
        )
        val_coord_y_dset = val_file.create_dataset(
            "coord_y", shape=(0, *coord_y_shape), maxshape=(None, *coord_y_shape), dtype='float32', chunks=True
        )
        val_time_gaps_dset = val_file.create_dataset(
            "time_gaps", shape=(0, *time_gap_shape), maxshape=(None, *time_gap_shape), dtype='int32', chunks=True
        )

        # Keep track of the current size for train and validation datasets
        current_train_size = 0
        current_val_size = 0

        print('iterate chunks')
        for batch_idx, batch in enumerate(data_iterator):
            print(f'Save Chunk {batch_idx} !!!!!!!!!!!!!!!!')

            # Unpack train and validation batches
            train_data, train_coords, train_time_gaps, train_masks, \
                val_data, val_coords, val_time_gaps, val_masks = batch

            train_chunk_size = train_data.shape[0]
            val_chunk_size = val_data.shape[0]

            # Resize and write train datasets
            train_data_dset.resize(current_train_size + train_chunk_size, axis=0)
            train_mask_dset.resize(current_train_size + train_chunk_size, axis=0)
            train_coord_time_dset.resize(current_train_size + train_chunk_size, axis=0)
            train_coord_x_dset.resize(current_train_size + train_chunk_size, axis=0)
            train_coord_y_dset.resize(current_train_size + train_chunk_size, axis=0)
            train_time_gaps_dset.resize(current_train_size + train_chunk_size, axis=0)

            train_data_dset[current_train_size:current_train_size + train_chunk_size] = train_data
            train_mask_dset[current_train_size:current_train_size + train_chunk_size] = train_masks
            train_coord_time_dset[current_train_size:current_train_size + train_chunk_size] = train_coords["time"].astype('datetime64[s]').astype('int64')
            train_coord_x_dset[current_train_size:current_train_size + train_chunk_size] = train_coords["x"]
            train_coord_y_dset[current_train_size:current_train_size + train_chunk_size] = train_coords["y"]
            train_time_gaps_dset[current_train_size:current_train_size + train_chunk_size] = train_time_gaps.numpy()

            current_train_size += train_chunk_size

            # Resize and write validation datasets
            val_data_dset.resize(current_val_size + val_chunk_size, axis=0)
            val_mask_dset.resize(current_val_size + val_chunk_size, axis=0)
            val_coord_time_dset.resize(current_val_size + val_chunk_size, axis=0)
            val_coord_x_dset.resize(current_val_size + val_chunk_size, axis=0)
            val_coord_y_dset.resize(current_val_size + val_chunk_size, axis=0)
            val_time_gaps_dset.resize(current_val_size + val_chunk_size, axis=0)

            val_data_dset[current_val_size:current_val_size + val_chunk_size] = val_data
            val_mask_dset[current_val_size:current_val_size + val_chunk_size] = val_masks
            val_coord_time_dset[current_val_size:current_val_size + val_chunk_size] = val_coords["time"].astype('datetime64[s]').astype('int64')
            val_coord_x_dset[current_val_size:current_val_size + val_chunk_size] = val_coords["x"]
            val_coord_y_dset[current_val_size:current_val_size + val_chunk_size] = val_coords["y"]
            val_time_gaps_dset[current_val_size:current_val_size + val_chunk_size] = val_time_gaps.numpy()

            current_val_size += val_chunk_size

            print(f"Train chunk of size {train_chunk_size} and validation chunk of size {val_chunk_size} written to respective HDF5 files.")

            data_iterator.reset()

    print(f"Processing complete. Train dataset saved to 'train_dataset.h5' and validation dataset saved to 'val_dataset.h5'.")







if __name__ == "__main__":
    main()

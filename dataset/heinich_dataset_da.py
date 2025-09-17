import os
import h5py
import torch
import random
import warnings
import numpy as np
import xarray as xr
from time import time
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Dict, Tuple, List
from itertools import product

from reconstruct import base_path
from si_dataset import ds
from utils.utils import compute_time_gaps


def select_random_timestamps_array(
    data_array: np.ndarray,
    coords: Dict[str, np.ndarray],
    num_timestamps: int = 11,
    seed: int = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Randomly selects a subset of timestamps per sample, reducing the time dimension.
    Args:
        data_array: shape (n_samples, n_features, time, y, x)
        coords: dict with 'time' key of shape (n_samples, time)
        num_timestamps: how many timestamps to keep
        seed: optional, for reproducibility
    Returns:
        reduced_data: shape (n_samples, n_features, num_timestamps, y, x)
        updated_coords: with 'time' key reduced to shape (n_samples, num_timestamps)
    """
    if seed is not None:
        random.seed(seed)

    n_samples, n_features, time_dim, height, width = data_array.shape
    time_coord = coords["time"]
    assert time_coord.shape == (n_samples, time_dim)

    reduced_data = np.empty((n_samples, n_features, num_timestamps, height, width), dtype=data_array.dtype)
    reduced_time = np.empty((n_samples, num_timestamps), dtype=time_coord.dtype)

    for i in range(n_samples):
        selected = sorted(random.sample(range(time_dim), num_timestamps))
        reduced_data[i] = np.take(data_array[i], indices=selected, axis=1)
        reduced_time[i] = time_coord[i, selected]

    updated_coords = coords.copy()
    updated_coords["time"] = reduced_time

    return reduced_data, updated_coords


def split_dataarray_parallel(
    chunk: np.ndarray,
    coords: Dict[str, np.ndarray],
    sample_size: List[Tuple[str, int]],
    overlap: List[Tuple[str, float]] = None,
    n_jobs: int = 24,
    #samples_per_split: int = 20910,
    samples_per_split: int = 209023,
    split_id: int = 0
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Split a 4D xarray.DataArray into overlapping patches, extracting all variables at once.

    Args:
        chunk (xr.DataArray): Input array with shape (index, time, y, x)
        sample_size (List[Tuple[str, int]]): Dimensions and patch sizes (e.g. [('time', 11), ('y', 15), ('x', 15)])
        overlap (List[Tuple[str, float]]): Fractional overlap per dimension
        n_jobs (int): Number of parallel jobs
        samples_per_split (int): Number of samples to return in this split
        split_id (int): Which split to return (used for batching across multiple calls)

    Returns:
        Tuple:
            - patches: np.ndarray with shape (n_samples, index, t, y, x)
            - coords: dict of coordinate values per sample (e.g. {'time': [...], 'y': [...], 'x': [...]})
    """

    # === Step 1: Prepare dimension dictionaries ===
    sample_size_dict = dict(sample_size)
    overlap_dict = dict(overlap) if overlap else {dim: 0.0 for dim in sample_size_dict}
    relevant_coords = [coord for coord, _ in sample_size]

    shape_dict = dict(zip(relevant_coords, chunk.shape[1:]))

    step_sizes = {dim: sample_size_dict[dim] for dim in relevant_coords}
    overlap_steps = {
        dim: int(step_sizes[dim] * overlap_dict.get(dim, 0.0)) if step_sizes[dim] > 1 else 0
        for dim in relevant_coords
    }

    # === Step 2: Compute valid start indices ===
    start_indices = {
        dim: [
            i for i in range(0, shape_dict[dim], step_sizes[dim] - overlap_steps[dim])
            if i + step_sizes[dim] <= shape_dict[dim]
        ]
        for dim in relevant_coords
    }

    # === Step 3: Generate combinations of patch start positions ===
    index_grid = list(product(*[start_indices[dim] for dim in relevant_coords]))
    #print(index_grid[:10])

    # === Step 4: Select the current batch (split) of samples ===
    start = split_id * samples_per_split
    end = min((split_id + 1) * samples_per_split, len(index_grid))
    batch_indices = index_grid[start:end]

    # === Step 5: Extract patches ===
    def extract_patch(idx_vals):
        warnings.filterwarnings('ignore')
        # Convert dictionary of slices into tuple of slices for numpy indexing
        slices = tuple(slice(idx, idx + step_sizes[dim]) for idx, dim in zip(idx_vals, relevant_coords))
        return chunk[:, slices[0], slices[1], slices[2]]

    patches = Parallel(n_jobs=n_jobs)(
        delayed(extract_patch)(idx_vals)
        for idx_vals in tqdm(batch_indices, desc=f"Extracting patches (Split {split_id})")
    )

    # === Step 6: Extract coordinate slices (optional) ===
    coord_result = {}
    for dim in relevant_coords:
        #print(f"======================={dim}===========================")
        coord_array = coords[dim]
        #print(coord_array)
        coord_slices = Parallel(n_jobs=n_jobs)(
            delayed(lambda idx: coord_array[idx:idx + step_sizes[dim]])(start_idx)
            for start_idx in [idx[relevant_coords.index(dim)] for idx in batch_indices]
        )
        #print(coord_slices)
        coord_result[dim] = np.stack(coord_slices)

        #print(relevant_coords)
        #print(len(patches))
        #print(patches)

    return np.stack(patches), coord_result


def assign_block_split_da(da: xr.DataArray, block_size: list, split: float = 0.8, seed: int = 42) -> np.ndarray:
    """
    Assigns each spatial-temporal block in a DataArray to train (1.0) or validation (0.0) based on a split ratio.
    The result is a 1D array of binary values indicating train/validation for each chunk.

    Args:
        da (xr.DataArray): Input array with dims (index, time, y, x).
        block_size (list of tuples): e.g. [('time', 53), ('y', 500), ('x', 500)].
        split (float): Fraction of blocks assigned to training (default 0.8).
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: A 1D array of binary values (1 for train, 0 for val) for each chunk.
    """
    if seed is not None:
        np.random.seed(seed)

    # Drop the 'index' dimension and work with just the spatial-temporal dims
    spatial_da = da.isel(index=0).drop_vars("index", errors="ignore")
    dims = list(spatial_da.dims)
    sizes = [spatial_da.sizes[dim] for dim in dims]

    block_size_dict = dict(block_size)
    block_sizes = [block_size_dict[dim] for dim in dims]

    # Number of blocks per dimension
    num_blocks = [int(np.ceil(size / bsize)) for size, bsize in zip(sizes, block_sizes)]
    total_blocks = np.prod(num_blocks)

    # Generate random assignments per block based on the split ratio
    random_assignments = (np.random.rand(total_blocks) < split).astype(int)  # 1 for training, 0 for validation

    # Flatten the 1D array to represent each block's split status
    return random_assignments


class XrFeatureDataset:
    def __init__(self, data_cube, split_array, sample_size, overlap, block_size):
        self.data_cube = data_cube
        self.split_array = split_array  # This is now a binary array
        self.sample_size = sample_size
        self.overlap = overlap
        self.block_size_dict = dict(block_size)
        self.dims = [dim for dim, _ in block_size]

        self.chunk_idx = 0
        self.split_idx = 0
        self.chunk = None

        # Automatically compute the length of the 'time' dimension
        self.time_len = self.data_cube.sizes['time']
        self.y_len = self.data_cube.sizes['y']
        self.x_len = self.data_cube.sizes['x']
        self.chunk_starts = self._compute_chunk_starts()


    def _compute_chunk_starts(self):
        """
        Computes the starting indices for chunks across all three dimensions (time, y, x).
        """
        # Compute the chunk starts for each dimension (time, y, x)
        time_starts = [
            i for i in range(0, self.time_len, self.block_size_dict['time'])
            if i + self.block_size_dict['time'] <= self.time_len
        ]

        y_starts = [
            i for i in range(0, self.y_len, self.block_size_dict['y'])
            if i + self.block_size_dict['y'] <= self.y_len
        ]

        x_starts = [
            i for i in range(0, self.x_len, self.block_size_dict['x'])
            if i + self.block_size_dict['x'] <= self.x_len
        ]

        # Generate all combinations of (time_start, y_start, x_start)
        chunk_grid = list(product(time_starts, y_starts, x_starts))

        return chunk_grid

    def __iter__(self):
        return self

    def __next__(self):
        # Instead of using `isel`, directly slice the `split_array` using numpy indexing
        if self.split_idx == 1:
            self.chunk_idx += 1
            self.split_idx = 0
            self.chunk = None

        if self.chunk_idx == len(self.chunk_starts):
            raise StopIteration


        if self.chunk is None:
            warnings.filterwarnings('ignore')

            # Get the chunk's starting indices for time, y, x
            t_start, y_start, x_start = self.chunk_starts[self.chunk_idx]
            #print('chunk starting indices: ')
            #print(t_start, y_start, x_start)

            # Use slice directly in the isel call for each dimension:
            chunk = self.data_cube.isel(
                time=slice(t_start, t_start + self.block_size_dict['time']),
                y=slice(y_start, y_start + self.block_size_dict['y']),
                x=slice(x_start, x_start + self.block_size_dict['x'])
            )
            is_val_chunk = self.split_array[self.chunk_idx] == 0  # Validation if it's 0, else it's training (1)

            coords = {k: chunk.coords[k].values for k in chunk.coords if k != 'index'}
            self.chunk = (chunk.values, self.split_array[self.chunk_idx], coords, is_val_chunk)

        chunk_values, split_values, coords, is_val_chunk = self.chunk

        print(f'Splitting chunk {self.chunk_idx}.{self.split_idx}')

        print(chunk_values.shape)
       # print(self.sample_size)
       # print(self.overlap)
       # print(coords)
        patches, patch_coords = split_dataarray_parallel(
            chunk=chunk_values,
            coords=coords,
            sample_size=self.sample_size,
            overlap=self.overlap,
            n_jobs=32,
            samples_per_split=7220,
            split_id=self.split_idx
        )

        patches, patch_coords = select_random_timestamps_array(patches, patch_coords, num_timestamps=11)

        # Remove patches that are completely NaN
        center_time = patches.shape[2] // 2
        center_y = patches.shape[3] // 2
        center_x = patches.shape[4] // 2

        mask_center_nan = np.isnan(patches[:, :, center_time, center_y, center_x]).all(axis=1)
        valid_mask = ~mask_center_nan

        patches = patches[valid_mask]
        patch_coords = {k: v[valid_mask] for k, v in patch_coords.items()}

        valid_mask = ~np.isnan(patches)

        if patches.shape[0] == 0:
            print(f"No valid samples in chunk {self.chunk_idx}, batch {self.split_idx}. Skipping.")
            self.split_idx += 1
            return None

        # Fill remaining NaNs with per-patch means
        patches = np.where(
            np.isnan(patches),
            np.nanmean(patches, axis=(2, 3, 4), keepdims=True),
            patches
        )

        time_gaps = compute_time_gaps(patch_coords["time"])

        # Apply time gap quality filter
        valid_time_gap_mask = (time_gaps <= 80).all(dim=1) & (time_gaps.sum(dim=1) <= 180)

        # Apply mask to all relevant data
        patches = patches[valid_time_gap_mask.numpy()]
        time_gaps = time_gaps[valid_time_gap_mask]
        valid_mask = valid_mask[valid_time_gap_mask.numpy()]

        print(f"✔️ Processed chunk {self.chunk_idx}.{self.split_idx} (val={is_val_chunk})")

        self.split_idx += 1


        return patches, time_gaps, is_val_chunk, valid_mask


def main():
    sample_size = [("time", 20), ("y", 15), ("x", 15)]
    overlap = [("time", 2./20), ("y", 2./15), ("x", 2./15)]
    block_size = [("time", 84), ("y", 500), ("x", 500)]

    print("📦 Assigning block-based train/val split ...")
    split_array = assign_block_split_da(ds, block_size=block_size, split=0.8)
    print("✅ Split assignment complete.")

    dataset = XrFeatureDataset(ds, split_array=split_array, block_size=block_size, sample_size=sample_size, overlap=overlap)

    base_path = '../'

    with h5py.File(os.path.join(base_path, "haininch_train.h5"), "w") as train_file, h5py.File(os.path.join("hainich_val.h5"), "w") as val_file:
        data_shape = (149, 11, 15, 15)
        time_gap_shape = torch.Size([10])



        def create_dsets(file, shape):
            return {
                "data": file.create_dataset("data", shape=(0, *shape), maxshape=(None, *shape), dtype="float32", chunks=True),
                "time_gaps": file.create_dataset("time_gaps", shape=(0, *time_gap_shape), maxshape=(None, *time_gap_shape), dtype="int32", chunks=True),
                "mask": file.create_dataset("mask", shape=(0, *shape), maxshape=(None, *shape), dtype="bool", chunks=True)
            }

        train_dsets = create_dsets(train_file, data_shape)
        val_dsets = create_dsets(val_file, data_shape)
        train_count, val_count = 0, 0

        for batch_idx, batch in enumerate(dataset):
            data, gaps, is_val, valid_mask = batch

            print(f"⏳ Writing batch {batch_idx} to dataset: {is_val}")



            if is_val:
                val_dsets["data"].resize(val_count + data.shape[0], axis=0)
                val_dsets["time_gaps"].resize(val_count + data.shape[0], axis=0)
                val_dsets["mask"].resize(val_count + data.shape[0], axis=0)

                val_dsets["data"][val_count:] = data
                val_dsets["time_gaps"][val_count:] = gaps.numpy()
                val_dsets["mask"][val_count:] = valid_mask

                val_count += data.shape[0]

            else:
                train_dsets["data"].resize(train_count + data.shape[0], axis=0)
                train_dsets["time_gaps"].resize(train_count + data.shape[0], axis=0)
                train_dsets["mask"].resize(train_count + data.shape[0], axis=0)

                print(data.shape)

                train_dsets["data"][train_count:] = data
                train_dsets["time_gaps"][train_count:] = gaps.numpy()
                train_dsets["mask"][train_count:] = valid_mask

                train_count += data.shape[0]

            print(f"✅ Wrote {data.shape[0]} {'val' if is_val else 'train'} samples")

    print("🎉 Done. Saved to 'dataarray_train.h5' and 'dataarray_val.h5'")


if __name__ == "__main__":
    main()
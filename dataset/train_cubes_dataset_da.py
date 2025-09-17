import os
import h5py
import torch
import random
import warnings
import numpy as np
import xarray as xr
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Dict, Tuple, List
from itertools import product
from utils.utils import compute_time_gaps
from prepare_dataarray import prepare_spectral_data

def check_consistency(
        cube_num: str,
        base_path: str,
        patches: np.ndarray,           # (n, 149, 11, 15, 15)
        coords: Dict[str, np.ndarray], # {'time': (n, 11), 'y': (n, 15), 'x': (n, 15)}
        mask: np.ndarray,              # bool, same shape as patches
        max_report: int = 10           # stop detailed logging after this many mismatches
    ) -> None:
    """
    Verifies that every kept value (mask == True) in `patches`
    matches the value extracted from the original cube.

    Parameters
    ----------
    cube_num   : '000174'  (six-digit string)
    base_path  : '/net/data_ssd/deepfeatures/trainingcubes'
    patches    : extracted patches,  (n, bands, 11, 15, 15)
    coords     : coordinate dict returned by `split_dataarray_parallel`
    mask       : bool array, same shape as `patches`
    max_report : max number of mismatches to print in detail
    """
    # ------------------------------------------------------------------
    # 1.  Re-open *exactly* the cube and re-compute spectral indices
    # ------------------------------------------------------------------
    path = os.path.join(base_path, f"{cube_num}.zarr")
    da   = xr.open_zarr(path)
    da   = da.s2l2a.where(da.cloud_mask == 0)
    ds   = prepare_spectral_data(da, to_ds = False)

    mismatches, reported = 0, 0

    # ------------------------------------------------------------------
    # 2.  Iterate over samples and compare values where mask is True
    # ------------------------------------------------------------------
    for i in range(patches.shape[0]):
        # integer indices for faster sel/isel
        t_idx = xr.DataArray(coords["time"][i], dims="time")
        y_idx = xr.DataArray(coords["y"][i],   dims="y")
        x_idx = xr.DataArray(coords["x"][i],   dims="x")

        #print(t_idx, y_idx, x_idx)

        cube_block = ds.sel(time=t_idx, y=y_idx, x=x_idx).values  # (149, 11, 15, 15)

        #print(cube_block[mask[i]])
        #print(patches[i][mask[i]])

        good = np.array_equal(
            cube_block[mask[i]],
            patches[i][mask[i]],
            equal_nan=True  # <- treat NaNs as equal
        )

        if not good:
            mismatches += 1
            if reported < max_report:
                bad = np.where(~np.isclose(
                        cube_block, patches[i], equal_nan=True) & mask[i])
                print(f"⚠️  mismatch in sample {i} — first bad idx {bad[0][:4]} …")
                reported += 1

    # ------------------------------------------------------------------
    # 3.  Final summary
    # ------------------------------------------------------------------
    if mismatches == 0:
        print(f"✅ cube {cube_num}: all {patches.shape[0]} samples match.")
    else:
        print(f"❌ cube {cube_num}: {mismatches}/{patches.shape[0]} samples mismatched.")


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
        coord_array = coords[dim]
        coord_slices = Parallel(n_jobs=n_jobs)(
            delayed(lambda idx: coord_array[idx:idx + step_sizes[dim]])(start_idx)
            for start_idx in [idx[relevant_coords.index(dim)] for idx in batch_indices]
        )
        coord_result[dim] = np.stack(coord_slices)

    return np.stack(patches), coord_result

def process_cube(cube_num, base_path):
    warnings.filterwarnings('ignore')
    path = os.path.join(base_path, f"{cube_num}.zarr")
    print(path)
    da = xr.open_zarr(path)
    da = da.s2l2a.where((da.cloud_mask == 0))

    # Count non-NaN points across spatial dimensions and bands
    valid_data_count = da.notnull().sum(dim=["band", "y", "x"])

    # Select only timestamps with at least 100 valid data points
    da = da.sel(time=valid_data_count > 150)

    #nan_fraction = da.isnull().mean(dim=("band", "y", "x"))
    #da = da.sel(time=nan_fraction < 0.97)
    da = da.chunk({"time": da.sizes["time"], "y": 90, "x": 90})

    ds = prepare_spectral_data(da, to_ds = False)
    data = ds.values
    coords = {dim: ds.coords[dim].values for dim in ds.dims if dim in ["time", "y", "x"]}

    if data.shape[1] < 11:
        return {}, {}, {}, {}

    patches, patch_coords = split_dataarray_parallel(
        chunk=data,
        coords=coords,
        sample_size=[("time", min(20, data.shape[1])), ("y", 15), ("x", 15)],
        overlap=[("time", 4./20), ("y", 6./15), ("x", 6./15)],
        n_jobs=32,
        samples_per_split=17220,
        split_id=0
    )


    patches, patch_coords = select_random_timestamps_array(patches, patch_coords, num_timestamps=11)

    # Remove patches that are completely NaN
    center_time = patches.shape[2] // 2
    center_y = patches.shape[3] // 2
    center_x = patches.shape[4] // 2

    center_nan_mask = np.isnan(patches[:, :, center_time, center_y, center_x]).all(axis=1)
    valid_mask = ~center_nan_mask
    patches = patches[valid_mask]
    patch_coords = {k: v[valid_mask] for k, v in patch_coords.items()}
    valid_mask_arr = ~np.isnan(patches)

    if patches.shape[0] == 0:
        return {}, {}, {}, {}

    patches = np.where(
        np.isnan(patches),
        np.nanmean(patches, axis=(2, 3, 4), keepdims=True),
        patches
    )

    mean_value = np.nanmean(patches, axis=(2, 3, 4), keepdims=True)
    print('================')
    print(mean_value.shape)
    print('~~~~~~~~')

    #patches = np.nan_to_num(patches, nan=0.0)

    time_gaps = compute_time_gaps(patch_coords['time'])

    valid_mask = (time_gaps <= 80).all(dim=1) & (time_gaps.sum(dim=1) <= 185)
    patches = patches[valid_mask.numpy()]
    time_gaps = time_gaps[valid_mask]
    valid_mask_arr = valid_mask_arr[valid_mask.numpy()]
    #valid_mask_arr = np.ones_like(patches, dtype=bool)

    return patches, time_gaps, valid_mask_arr, patch_coords

def divide_mini_cubes(split=0.75):
    random.seed(42)
    numbers = list(range(500))
    selected = random.sample(numbers, int(split * 500))
    remaining = [n for n in numbers if n not in selected]
    random.shuffle(remaining)
    val_split = int(2 / 3 * len(remaining))
    return selected, remaining[:val_split], remaining[val_split:]

base_path = '/net/data_ssd/deepfeatures/trainingcubes'
train, val, test = divide_mini_cubes()
set_to_process = train
six_digit_strings = [f"{num:06d}" for num in set_to_process]

with h5py.File("./da_train_149.h5", "w") as f:
    data_shape = (149, 11, 15, 15)
    time_gap_shape = (10,)

    data_dset = f.create_dataset("data", shape=(0, *data_shape), maxshape=(None, *data_shape), dtype='float32', chunks=True)
    mask_dset = f.create_dataset("mask", shape=(0, *data_shape), maxshape=(None, *data_shape), dtype='bool', chunks=True)
    time_gaps_dset = f.create_dataset("time_gaps", shape=(0, *time_gap_shape), maxshape=(None, *time_gap_shape), dtype='int32', chunks=True)

    current_size = 0

    for cube_num in six_digit_strings:
        data, time_gaps, mask, patch_coords = process_cube(cube_num, base_path)
        #print(patch_coords['time'])
        #print(patch_coords['y'])
        #print(patch_coords['x'])
        print(type(data))
        print(type(time_gaps))
        print(type(mask))


        #check_consistency(cube_num, base_path, data, patch_coords, mask)
        #break
        if isinstance(data, dict):
            continue

        n_samples = data.shape[0]

        data_dset.resize(current_size + n_samples, axis=0)
        mask_dset.resize(current_size + n_samples, axis=0)
        time_gaps_dset.resize(current_size + n_samples, axis=0)

        data_dset[current_size:current_size + n_samples] = data
        mask_dset[current_size:current_size + n_samples] = mask
        time_gaps_dset[current_size:current_size + n_samples] = time_gaps.numpy()

        current_size += n_samples
    print(f"{current_size} samples written")


# val: 10328
# train: 44011
import os
import h5py
import torch
import random
import numpy as np
import xarray as xr
from utils.utils import drop_if_central_point_nan_at_selected_times, concatenate, compute_time_gaps, select_random_timestamps
from ml4xcube.utils import get_chunk_by_index, split_chunk
#from prepare_si_dataset import prepare_cube
from prepare_dataarray import prepare_spectral_data
from sentinel1 import match_sentinel1_to_s2_cube
from ml4xcube.preprocessing import drop_nan_values, fill_nan_values
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd

import numpy as np
import pandas as pd
import xarray as xr

import numpy as np
import pandas as pd
import xarray as xr

import numpy as np
import pandas as pd

def extract_batch_s1_patches(
    s2_coords: dict,
    s1_array: np.ndarray,
    s1_times: list,
    max_time_diff_days: int = 3
):
    """
    Efficient batch extraction of Sentinel-1 patches aligned to S2 samples.

    - Selects 7 S2-aligned timestamps per sample (first, middle, last + 4 random)
    - Matches to closest S1 timestamps (within max_time_diff_days)
    - Extracts all patches
    - Fills missing values (NaNs) with per-sample, per-band mean
    - Returns nan mask and matched indices

    Args:
        s2_coords (dict): with keys:
            - 'time': (N, 7)
            - 'x': (N, P)
            - 'y': (N, P)
        s1_array (np.ndarray): shape (bands, T, H, W)
        s1_times (list): list of datetime.datetime or np.datetime64
        max_time_diff_days (int): threshold in days for matching S1 to S2

    Returns:
        patches: (N, bands, 7, H, W), filled in-place
        nan_mask: same shape, True where NaNs were filled
        matched_indices_all: list of N lists of matched S1 indices or None
        target_timestamps_all: list of N lists of selected S2-aligned timestamps
    """
    s1_times = np.array(s1_times)
    N = s2_coords["time"].shape[0]
    print(N)
    print(s2_coords['time'].shape)
    bands = s1_array.shape[0]

    print(s1_array.shape)

    sample_patches = []
    matched_indices_all = []
    target_timestamps_all = []


    for i in range(N):
        # === Temporal selection (first, middle, last + 4 random) ===
        s2_times = pd.to_datetime(s2_coords["time"][i])
        fixed = [0, 3, 6]
        mid_lower = sorted(np.random.choice([1, 2], size=2, replace=False))
        mid_upper = sorted(np.random.choice([4, 5], size=2, replace=False))
        final_idx = sorted(fixed + mid_lower + mid_upper)
        target_times = [s2_times[j] for j in final_idx]
        target_timestamps_all.append(target_times)

        # === Match S1 timestamps ===
        matched_indices = []
        for t in target_times:
            diffs = np.abs(s1_times - np.datetime64(t))
            min_diff = diffs.min()
            if min_diff <= np.timedelta64(max_time_diff_days, 'D'):
                matched_indices.append(np.argmin(diffs))
            else:
                matched_indices.append(None)
        matched_indices_all.append(matched_indices)
        print(matched_indices_all)

        # === Spatial selection ===
        x_idx = s2_coords["x"][i]
        print(x_idx)
        y_idx = s2_coords["y"][i]
        print(y_idx)
        x_start, x_end = x_idx.min(), x_idx.max() + 1
        y_start, y_end = y_idx.min(), y_idx.max() + 1
        H, W = y_end - y_start, x_end - x_start

        # === Initialise patch ===
        patch = np.full((bands, 7, H, W), np.nan, dtype=np.float32)

        for j, s1_idx in enumerate(matched_indices):
            if s1_idx is not None:
                print('==================')
                print(s1_idx)
                print(y_start, y_end)
                print(x_start, x_end)
                patch[:, j] = s1_array[:, s1_idx, y_start:y_end, x_start:x_end]

        sample_patches.append(patch)

    # === Stack all samples ===
    patches = np.stack(sample_patches, axis=0)  # (N, bands, 7, H, W)

    # === Compute NaN mask in one go ===
    nan_mask = np.isnan(patches)

    # === In-place fill of NaNs using per-sample, per-band mean ===
    means = np.nanmean(patches, axis=(2, 3, 4), keepdims=True)  # (N, bands, 1, 1, 1)
    patches[:] = np.where(nan_mask, means, patches)

    return patches, nan_mask, matched_indices_all, target_timestamps_all



def process_cube(cube_num):
    cube_path = os.path.join(base_path, cube_num + '.zarr')
    print(cube_path)


    da = xr.open_zarr(cube_path)

    s1 = match_sentinel1_to_s2_cube(da)
    #print(s1.backscatter.values.shape)
    #print(s1.band.values.shape)
    #print(s1.time.values)

    da = da.s2l2a.where((da.cloud_mask == 0))

    n_total = da.sizes["band"] * da.sizes["y"] * da.sizes["x"]
    threshold = int(n_total * 0.035)

    # Count non-NaN points per time step
    valid_data_count = da.notnull().sum(dim=["band", "y", "x"])

    # Keep only time steps with at least 3.5% valid data
    da = da.sel(time=valid_data_count >= threshold)

    chunks = {"time": da.sizes["time"], "y": 90, "x": 90}
    da = da.chunk(chunks)


    ds = prepare_spectral_data(da)
    #s1_clean = s1.backscatter.drop_vars(["title", "description"], errors="ignore")
    #ds_s1 = prepare_spectral_data(s1_clean)


    chunk, coords = get_chunk_by_index(ds, 0)
    #chunk_s1, coords_s1 = get_chunk_by_index(ds_s1, 0)

    print('split chunk')
    chunk, coords = split_chunk(chunk, coords, sample_size=[("time", 10), ("y", 15), ("x", 15)], overlap=[("time", 7./10.), ("y", 10./15.), ("x", 10./15.)])

    print('select timestamps')
    chunk, coords = select_random_timestamps(chunk, coords, num_timestamps=7)


    time_gaps = compute_time_gaps(coords['time'])  # (N, 10) tensor of int

    # Early return if empty
    if torch.tensor(time_gaps).ndim != 2 or torch.tensor(time_gaps).shape[0] == 0: #or torch.tensor(time_gaps).shape[1] != 4:
        print(f"Invalid or empty time_gaps: shape {time_gaps.shape}")
        return {}, {}, {}


    # Final valid mask: both conditions must be satisfied
    valid_time_gap_mask = (time_gaps.sum(dim=1) <= 45)

    # Apply the mask to filter valid samples
    chunk = {k: v[valid_time_gap_mask] for k, v in chunk.items()}
    filtered_indices = np.where(valid_time_gap_mask.numpy())[0]
    coords = {k: v[filtered_indices] for k, v in coords.items()}

    vars_train = list(chunk.keys())
    print('drop if central point is not numeric')
    chunk, coords = drop_if_central_point_nan_at_selected_times(chunk, coords, vars=vars_train, required_time_indices=list(range(7)))
    print('create validity mask')
    valid_mask = {var: ~np.isnan(chunk[var]) for var in vars_train}
    print('fill nan values')
    chunk = fill_nan_values(chunk, vars=vars_train, method='sample_mean')
    for var in chunk:
        cnt = np.isnan(chunk[var]).sum()
        if cnt > 0: print(f"NaNs in variable '{var}' after filling: {cnt}")

    s1_patches_np, s1_coords, s1_valid_mask_np = extract_batch_s1_patches(coords, s1.backscatter.values.transpose(1, 0, 2, 3) , s1.time.values)
    #time_gaps_s1 = compute_time_gaps(s1_coords['time'])


    return chunk, coords, valid_mask, s1_patches_np, s1_coords, s1_valid_mask_np


def divide_mini_cubes(split = 0.75):
    # Set the seed for reproducibility
    random.seed(42)

    # Generate a list of numbers from 0 to 499
    numbers = list(range(500))

    # Calculate 80% of 500
    count_to_select = int(split * 500)

    # Randomly select 80% of the numbers
    selected_numbers = random.sample(numbers, count_to_select)

    # Find the remaining numbers
    remaining_numbers = [num for num in numbers if num not in selected_numbers]

    print(f"Selected Numbers ({int(split * 100)}%):", selected_numbers)
    print(f"Remaining Numbers ({int((1-split) * 100)}%):", remaining_numbers)
    return selected_numbers, remaining_numbers


# Call the method
selected_numbers, remaining_numbers = divide_mini_cubes()

# Randomly split remaining_numbers into 2/3 validation and 1/3 test
random.shuffle(remaining_numbers)  # Shuffle for randomness

val_count = int(2 / 3 * len(remaining_numbers))
val_numbers = remaining_numbers[:val_count]
test_numbers = remaining_numbers[val_count:]

print(f"Validation cubes ({len(val_numbers)}):", val_numbers)
print(f"Test cubes ({len(test_numbers)}):", test_numbers)

# Convert to 6-digit strings if needed
val_six_digit_strings = [f"{num:06d}" for num in val_numbers]
test_six_digit_strings = [f"{num:06d}" for num in test_numbers]

# Iterate through the selected numbers and create 6-digit strings
six_digit_strings = [f"{num:06d}" for num in selected_numbers]
#six_digit_strings = val_six_digit_strings

base_path = '/net/data_ssd/deepfeatures/trainingcubes'

#with h5py.File("/net/home/jpeters/h_test_149.h5", "w") as train_file:
with h5py.File("train_test.h5", "w") as train_file:
    # === Dataset shapes ===
    s2_shape = (10, 7, 15, 15)         # e.g., 10 spectral features, 7 time steps
    s2_mask_shape = s2_shape
    s2_time_gap_shape = (6,)          # gaps between 7 timestamps

    s1_shape = (2, 7, 15, 15)          # 2 bands: VV, VH
    s1_mask_shape = s1_shape
    s1_time_gap_shape = (6,)          # 7 timestamps → 6 gaps

    # === Create S2 datasets ===
    train_data_dset = train_file.create_dataset(
        "data", shape=(0, *s2_shape), maxshape=(None, *s2_shape), dtype='float32', chunks=True
    )
    train_mask_dset = train_file.create_dataset(
        "mask", shape=(0, *s2_mask_shape), maxshape=(None, *s2_mask_shape), dtype='bool', chunks=True
    )
    train_time_gaps_dset = train_file.create_dataset(
        "time_gaps", shape=(0, *s2_time_gap_shape), maxshape=(None, *s2_time_gap_shape), dtype='int32', chunks=True
    )

    # === Create S1 datasets ===
    train_s1_data_dset = train_file.create_dataset(
        "s1_data", shape=(0, *s1_shape), maxshape=(None, *s1_shape), dtype='float32', chunks=True
    )
    train_s1_mask_dset = train_file.create_dataset(
        "s1_mask", shape=(0, *s1_mask_shape), maxshape=(None, *s1_mask_shape), dtype='bool', chunks=True
    )
    train_s1_time_gaps_dset = train_file.create_dataset(
        "s1_time_gaps", shape=(0, *s1_time_gap_shape), maxshape=(None, *s1_time_gap_shape), dtype='int32', chunks=True
    )


    current_train_size = 0
    batch_size = 36
    for i in range(0, len(six_digit_strings), batch_size):
        batch = six_digit_strings[i:i + batch_size]
        print(f"Processing batch: {batch}")

        with Pool(processes=batch_size) as pool:
            results = pool.map(process_cube, batch)

        # Further processing for the batch can be added here
        print(f"Finished processing batch: {batch}")

        print('concatenate batch of chunks')
        chunk, coords, valid_mask, s1_data, s1_coords, s1_mask = zip(*results)
        chunk, coords, valid_mask = concatenate(chunk, coords, valid_mask)
        s1_data = np.concatenate(s1_data)
        s1_mask = np.concatenate(s1_mask)

        s1_time = np.concatenate([c["time"] for c in s1_coords])
        s1_x = np.stack([c["x"] for c in s1_coords])
        s1_y = np.stack([c["y"] for c in s1_coords])

        time_gaps = compute_time_gaps(coords['time'])
        time_gaps_s1 = compute_time_gaps(s1_coords['time'])

        # Compute the size of the current batch
        batch_chunk_size = chunk.shape[0]

        if batch_chunk_size > 0:
            # === Resize datasets ===
            train_data_dset.resize(current_train_size + batch_chunk_size, axis=0)
            train_mask_dset.resize(current_train_size + batch_chunk_size, axis=0)
            train_time_gaps_dset.resize(current_train_size + batch_chunk_size, axis=0)

            train_s1_data_dset.resize(current_train_size + batch_chunk_size, axis=0)
            train_s1_mask_dset.resize(current_train_size + batch_chunk_size, axis=0)
            train_s1_time_gaps_dset.resize(current_train_size + batch_chunk_size, axis=0)

            # === Write to datasets ===
            train_data_dset[current_train_size:current_train_size + batch_chunk_size] = chunk
            train_mask_dset[current_train_size:current_train_size + batch_chunk_size] = valid_mask
            train_time_gaps_dset[current_train_size:current_train_size + batch_chunk_size] = time_gaps.numpy()

            train_s1_data_dset[current_train_size:current_train_size + batch_chunk_size] = s1_data
            train_s1_mask_dset[current_train_size:current_train_size + batch_chunk_size] = s1_mask
            train_s1_time_gaps_dset[current_train_size:current_train_size + batch_chunk_size] = s1_time_gaps.numpy()

            current_train_size += batch_chunk_size

        print(f"Train batch of size {batch_chunk_size} written to HDF5 file.")

        print(f"Total samples written: {current_train_size}")
    print(current_train_size)



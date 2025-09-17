import os
import h5py
import torch
import random
import numpy as np
import xarray as xr
from utils.utils import drop_if_central_point_nan_or_inf, drop_if_central_point_nan_at_selected_times,  concatenate, compute_time_gaps, select_random_timestamps, select_timestamps_from_sections
from ml4xcube.utils import get_chunk_by_index, split_chunk
#from prepare_si_dataset import prepare_cube
from prepare_dataarray import prepare_spectral_data

from ml4xcube.preprocessing import drop_nan_values, fill_nan_values
from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np
import xarray as xr
import pandas as pd


def extract_s1_patches_for_coords(s1: xr.Dataset, coords: dict, patch_size: int = 15) -> np.ndarray:
    """
    For each sample in `coords`, extract a (2, 7, patch_size, patch_size) Sentinel-1 patch:
    - Closest S1 timestamp for each timestamp in coords['time'][i]
    - Patch is centered at coords['x'][i], coords['y'][i]

    Args:
        s1 (xr.Dataset): Sentinel-1 cube with dimensions (band, time, y, x)
        coords (dict): Dictionary containing 'time', 'x', and 'y' arrays
        patch_size (int): Size of spatial patch (default: 15)

    Returns:
        np.ndarray: S1 patches of shape (N, 2, 7, patch_size, patch_size)
    """
    s1_times = pd.to_datetime(s1.time.values)
    half = patch_size // 2
    s1_patches = []

    for i in range(len(coords["time"])):
        s2_times = pd.to_datetime(coords["time"][i])  # shape: (7,)
        center_x = coords["x"][i]
        center_y = coords["y"][i]

        # Find the closest S1 timestamp for each S2 timestamp
        selected_s1_times = []
        for s2_t in s2_times:
            diffs = np.abs((s1_times - s2_t).astype('timedelta64[s]'))
            closest_idx = int(np.argmin(diffs))
            selected_s1_times.append(s1_times[closest_idx])

        # Ensure unique timestamps if duplicates occur
        selected_s1_times = pd.to_datetime(selected_s1_times)

        # Extract the patch from S1
        s1_patch = s1.sel(
            time=selected_s1_times,
            x=slice(center_x - half, center_x + half + 1),
            y=slice(center_y - half, center_y + half + 1)
        ).to_array()  # shape: [band, time, y, x]

        # Validate shape
        if (
                s1_patch.sizes["time"] != len(s2_times) or
                s1_patch.sizes["x"] != patch_size or
                s1_patch.sizes["y"] != patch_size
        ):
            print(f"[Warning] Skipping sample {i} due to shape mismatch")
            continue

        s1_patches.append(s1_patch.values)  # shape: [2, 7, 15, 15]

    return np.stack(s1_patches) if s1_patches else np.empty((0, 2, 7, patch_size, patch_size), dtype=np.float32)



def process_cube(cube_num):
    cube_path = os.path.join(base_path, cube_num + '.zarr')
    print(cube_path)


    da = xr.open_zarr(cube_path)

    da = da.s2l2a.where((da.cloud_mask == 0))

    #da = da.dropna(dim="time", how="all")

    # Count non-NaN points across spatial dimensions and bands
    #valid_data_count = da.notnull().sum(dim=["band", "y", "x"])

    # Select only timestamps with at least 100 valid data points
    #da = da.sel(time=valid_data_count >= 150)

    n_total = da.sizes["band"] * da.sizes["y"] * da.sizes["x"]
    threshold = int(n_total * 0.035)

    # Count non-NaN points per time step
    valid_data_count = da.notnull().sum(dim=["band", "y", "x"])

    # Keep only time steps with at least 3.5% valid data
    da = da.sel(time=valid_data_count >= threshold)

    chunks = {"time": da.sizes["time"], "y": 90, "x": 90}
    da = da.chunk(chunks)

    # Create a mask: True where at least one valid value exists per timestep
    #valid_timesteps = da.isnull().all(dim=["band", "y", "x"]) == False

    # Compute the boolean mask
    #valid_timesteps = valid_timesteps.compute()

    #filtered = da.sel(time=da.time[valid_timesteps])

    ds = prepare_spectral_data(da)

    chunk, coords = get_chunk_by_index(ds, 0)

    print('split chunk')
    chunk, coords = split_chunk(chunk, coords, sample_size=[("time", 8), ("y", 15), ("x", 15)], overlap=[("time", 7./8.), ("y", 10./15.), ("x", 10./15.)])

    print('select timestamps')
    chunk, coords = select_random_timestamps(chunk, coords, num_timestamps=5)

    time_gaps = compute_time_gaps(coords['time'])  # (N, 10) tensor of int

    # Early return if empty
    if torch.tensor(time_gaps).ndim != 2 or torch.tensor(time_gaps).shape[0] == 0 or torch.tensor(time_gaps).shape[1] != 4:
        print(f"Invalid or empty time_gaps: shape {time_gaps.shape}")
        return {}, {}, {}

    # Constraint 1: No single gap ≥ 85
    cond1 = (time_gaps < 80).all(dim=1)

    # Constraint 2: Sum of gaps ≤ 180
    cond2 = (time_gaps.sum(dim=1) <= 45)

    # Final valid mask: both conditions must be satisfied
    valid_time_gap_mask = cond1 & cond2

    # Apply the mask to filter valid samples
    chunk = {k: v[valid_time_gap_mask] for k, v in chunk.items()}
    filtered_indices = np.where(valid_time_gap_mask.numpy())[0]
    coords = {k: v[filtered_indices] for k, v in coords.items()}

    vars_train = list(chunk.keys())
    print('drop if central point is not numeric')
    #chunk, coords = drop_if_central_point_nan_or_inf(chunk, coords, vars=vars_train)
    chunk, coords = drop_if_central_point_nan_at_selected_times(chunk, coords, vars=vars_train, required_time_indices=list(range(5)))
    print('create validity mask')
    valid_mask = {var: ~np.isnan(chunk[var]) for var in vars_train}
    print('fill nan values')
    chunk = fill_nan_values(chunk, vars=vars_train, method='sample_mean')
    for var in chunk:
        cnt = np.isnan(chunk[var]).sum()
        if cnt > 0: print(f"NaNs in variable '{var}' after filling: {cnt}")

    return chunk, coords, valid_mask


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
       # Initialize expandable datasets for train
       train_shape = (10, 5, 15, 15)  # Train data shape (e.g., (209, 11, 15, 15))
       mask_shape = train_shape # Masks have the same shape as the data
       #coord_time_shape = (11,)
       #coord_x_shape = (15,)
       #coord_y_shape = (15,)
       time_gap_shape = torch.Size([10])
       # Train datasets
       # Train datasets
       train_data_dset = train_file.create_dataset(
           "data", shape=(0, *train_shape), maxshape=(None, *train_shape), dtype='float32', chunks=True
       )
       train_mask_dset = train_file.create_dataset(
           "mask", shape=(0, *mask_shape), maxshape=(None, *mask_shape), dtype='bool', chunks=True
       )
       train_time_gaps_dset = train_file.create_dataset(
           "time_gaps", shape=(0, *time_gap_shape), maxshape=(None, *time_gap_shape), dtype='int32', chunks=True
       )
       #train_coord_time_dset = train_file.create_dataset(
       #    "coord_time", shape=(0, *coord_time_shape), maxshape=(None, *coord_time_shape), dtype='int64', chunks=True
       #)
       #train_coord_x_dset = train_file.create_dataset(
       #    "coord_x", shape=(0, *coord_x_shape), maxshape=(None, *coord_x_shape), dtype='float32', chunks=True
       #)
       #train_coord_y_dset = train_file.create_dataset(
       #    "coord_y", shape=(0, *coord_y_shape), maxshape=(None, *coord_y_shape), dtype='float32', chunks=True
       #)
       train_time_gaps_dset = train_file.create_dataset(
           "time_gaps", shape=(0, *time_gap_shape), maxshape=(None, *time_gap_shape), dtype='int32', chunks=True
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
           chunk, coords, valid_mask = zip(*results)
           try:
            chunk, coords, valid_mask = concatenate(chunk, coords, valid_mask)
           except Exception as e:
               print(str(e))
               continue
           time_gaps = compute_time_gaps(coords['time'])

           # Compute the size of the current batch
           batch_chunk_size = chunk.shape[0]

           if batch_chunk_size > 0:

               # Resize datasets to accommodate the new batch
               train_data_dset.resize(current_train_size + batch_chunk_size, axis=0)
               train_mask_dset.resize(current_train_size + batch_chunk_size, axis=0)
               #train_coord_time_dset.resize(current_train_size + batch_chunk_size, axis=0)
               #train_coord_x_dset.resize(current_train_size + batch_chunk_size, axis=0)
               #train_coord_y_dset.resize(current_train_size + batch_chunk_size, axis=0)
               train_time_gaps_dset.resize(current_train_size + batch_chunk_size, axis=0)

               # Write the batch data into the HDF5 datasets
               train_data_dset[current_train_size:current_train_size + batch_chunk_size] = chunk
               train_mask_dset[current_train_size:current_train_size + batch_chunk_size] = valid_mask
               #train_coord_time_dset[current_train_size:current_train_size + batch_chunk_size] = coords['time'].astype('datetime64[s]').astype('int64')
               #train_coord_x_dset[current_train_size:current_train_size + batch_chunk_size] = coords['x']
               #train_coord_y_dset[current_train_size:current_train_size + batch_chunk_size] = coords['y']
               train_time_gaps_dset[current_train_size:current_train_size + batch_chunk_size] = time_gaps.numpy()

               # Update the current train size
               current_train_size += batch_chunk_size

           print(f"Train batch of size {batch_chunk_size} written to HDF5 file.")

print(current_train_size)



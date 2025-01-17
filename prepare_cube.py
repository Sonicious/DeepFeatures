import os
import h5py
import torch
import pickle
import random
import numpy as np
import xarray as xr
from utils import drop_if_central_point_nan_or_inf, concatenate, compute_time_gaps, select_random_timestamps
from ml4xcube.utils import get_chunk_by_index, split_chunk
from prepare_si_dataset import prepare_cube
from ml4xcube.preprocessing import drop_nan_values, drop_inf_values, fill_nan_values, standardize
from pathos.multiprocessing import ProcessingPool as Pool

# Load the percentile boundaries
with open("98_percentile_mini_cubes.pkl", "rb") as f:
    mini_cube_bounds = pickle.load(f)

with open("98_percentile.pkl", "rb") as f:
    full_bounds = pickle.load(f)

with open("confidence_intervals.pkl", "rb") as f:
    iqr_bounds = pickle.load(f)

# Combine boundaries by taking the minimal lower bound and maximal upper bound
combined_bounds = {}
for var in set(mini_cube_bounds.keys()).union(full_bounds.keys()):
    mini_bounds = mini_cube_bounds.get(var, [-float("inf"), float("inf")])
    full_bounds_var = full_bounds.get(var, [-float("inf"), float("inf")])
    iqr_bounds_var = iqr_bounds.get(var, [-float("inf"), float("inf")])
    combined_bounds[var] = [
        min(mini_bounds[0], full_bounds_var[0], iqr_bounds_var[0]),  # Minimal lower bound
        max(mini_bounds[1], full_bounds_var[1], iqr_bounds_var[1]),  # Maximal upper bound
    ]


def process_cube(cube_num):
    cube_path = os.path.join(base_path, cube_num + '.zarr')
    print(cube_path)

    da = xr.open_zarr(cube_path)["s2l2a"]

    ds = prepare_cube(da)

    # Apply outlier filtering
    print('remove outliers')
    for var in ds.data_vars:
        if var in combined_bounds:
            lower_bound, upper_bound = combined_bounds[var]
            #print(f"Filtering {var} with bounds: {lower_bound} to {upper_bound}")
            ds[var] = xr.where((ds[var] < lower_bound) | (ds[var] > upper_bound), np.nan, ds[var])
        else:
            print(f"No bounds found for {var}, skipping outlier filtering.")


    chunk, coords = get_chunk_by_index(ds, 0, block_size=[('time', 20), ('x', 90), ('y', 90)])

    print('split chunk')
    chunk, coords = split_chunk(chunk, coords, sample_size=[("time", 20), ("y", 15), ("x", 15)], overlap=[("time", 0.), ("y", 7./15.), ("x", 7./15.)])

    print('select timestamps')
    chunk, coords = select_random_timestamps(chunk, coords, num_timestamps=11)


    vars_train = list(chunk.keys())
    print('drop nan values ')
    chunk, coords = drop_nan_values(chunk, coords, mode='if_all_nan', vars=vars_train)
    print('drop inf values')
    chunk, coords = drop_inf_values(chunk, coords, vars=vars_train)
    print('drop if central point is not numeric')
    chunk, coords = drop_if_central_point_nan_or_inf(chunk, coords, vars=vars_train)
    print('create validity mask')
    valid_mask = {var: ~np.isnan(chunk[var]) for var in vars_train}
    print('fill nan values')
    chunk = fill_nan_values(chunk, vars=vars_train, method='sample_mean')
    for var in chunk:
        cnt = np.isnan(chunk[var]).sum()
        if cnt > 0: print(f"NaNs in variable '{var}' after filling: {cnt}")
    print('standardizing')
    chunk = standardize(chunk, stats)
    for var in chunk:
        cnt = np.isnan(chunk[var]).sum()
        if cnt > 0: print(f"NaNs in variable '{var}' after standardization: {cnt}")

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

# Iterate through the selected numbers and create 6-digit strings
six_digit_strings = [f"{num:06d}" for num in selected_numbers]
#six_digit_strings = [f"{num:06d}" for num in remaining_numbers]

base_path = '/net/data_ssd/deepfeatures/trainingcubes'

with open('mean_std_mini_cubes.pkl', "rb") as f:
    stats = pickle.load(f)


with h5py.File("train_m.h5", "w") as train_file:
       # Initialize expandable datasets for train
       train_shape = (11, 15, 15, 221)  # Train data shape (e.g., (209, 11, 15, 15))
       mask_shape = train_shape  # Masks have the same shape as the data
       coord_time_shape = (11,)
       coord_x_shape = (15,)
       coord_y_shape = (15,)
       time_gap_shape = torch.Size([10])
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
       current_train_size = 0
       batch_size = 24
       for i in range(0, len(six_digit_strings), batch_size):
           batch = six_digit_strings[i:i + batch_size]
           print(f"Processing batch: {batch}")

           with Pool(processes=batch_size) as pool:
               results = pool.map(process_cube, batch)

           # Further processing for the batch can be added here
           print(f"Finished processing batch: {batch}")

           print('concatenate batch of chunks')
           chunk, coords, valid_mask = zip(*results)
           chunk, coords, valid_mask = concatenate(chunk, coords, valid_mask)
           time_gaps = compute_time_gaps(coords['time'])

           # Compute the size of the current batch
           batch_chunk_size = chunk.shape[0]

           if batch_chunk_size > 0:

               # Resize datasets to accommodate the new batch
               train_data_dset.resize(current_train_size + batch_chunk_size, axis=0)
               train_mask_dset.resize(current_train_size + batch_chunk_size, axis=0)
               train_coord_time_dset.resize(current_train_size + batch_chunk_size, axis=0)
               train_coord_x_dset.resize(current_train_size + batch_chunk_size, axis=0)
               train_coord_y_dset.resize(current_train_size + batch_chunk_size, axis=0)
               train_time_gaps_dset.resize(current_train_size + batch_chunk_size, axis=0)

               # Write the batch data into the HDF5 datasets
               train_data_dset[current_train_size:current_train_size + batch_chunk_size] = chunk
               train_mask_dset[current_train_size:current_train_size + batch_chunk_size] = valid_mask
               train_coord_time_dset[current_train_size:current_train_size + batch_chunk_size] = coords['time'].astype('datetime64[s]').astype('int64')
               train_coord_x_dset[current_train_size:current_train_size + batch_chunk_size] = coords['x']
               train_coord_y_dset[current_train_size:current_train_size + batch_chunk_size] = coords['y']
               train_time_gaps_dset[current_train_size:current_train_size + batch_chunk_size] = time_gaps.numpy()

               # Update the current train size
               current_train_size += batch_chunk_size

           print(f"Train batch of size {batch_chunk_size} written to HDF5 file.")





"""
    sample_shape = chunk.shape[1:]
    mask_shape = sample_shape
    time_shape = coords['time'].shape[1:]
    x_shape = coords['x'].shape[1:]
    y_shape = coords['y'].shape[1:]
    gaps_shape = time_gaps.shape[1:]
(11, 15, 15, 221)
(11, 15, 15, 221)
(11,)
(15,)
(15,)
torch.Size([10])"""
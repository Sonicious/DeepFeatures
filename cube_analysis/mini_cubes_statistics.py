import os
import pickle
import xarray as xr
import numpy as np
from tqdm import tqdm
from dataset.prepare_si_dataset import prepare_cube
from concurrent.futures import ProcessPoolExecutor, as_completed

# Path to the directory containing the minicubes
minicube_dir = "/net/data_ssd/deepfeatures/trainingcubes"  # Replace with the actual path

# List all .zarr minicubes in the directory
minicube_paths = [os.path.join(minicube_dir, f) for f in os.listdir(minicube_dir) if f.endswith(".zarr")]

# Load and prepare the first cube to determine variables
first_cube_path = minicube_paths[0]
print(f"Loading the first cube: {first_cube_path}")
prepared_first_ds = prepare_cube(xr.open_zarr(first_cube_path)["s2l2a"])
variables = list(prepared_first_ds.data_vars)
print(f"Variables found: {variables}")

# Function to compute mean and standard deviation incrementally
def compute_mean_std(data_sum, data_sq_sum, count):
    mean = data_sum / count
    variance = (data_sq_sum / count) - (mean ** 2)
    std_dev = np.sqrt(variance)
    return mean, std_dev

# Function to process a single minicube
def process_minicube_for_stats(path, var):
    try:
        da = xr.open_zarr(path)
        da = da.s2l2a.where((da.cloud_mask == 0))
        prepared_ds = prepare_cube(da)

        if var in prepared_ds:
            data = prepared_ds[var].values
            # Apply percentile bounds to exclude outliers
            data = data[~np.isnan(data)]  # Remove NaN values
            data_sum = np.sum(data)
            data_sq_sum = np.sum(data ** 2)
            count = data.size
            return data_sum, data_sq_sum, count
        else:
            print(f"Variable {var} not found in {path}")
    except Exception as e:
        print(f"Failed to load or prepare variable {var} from {path}: {e}")
    return 0, 0, 0

# Load existing results if they exist
output_path = "../mean_std_mini_cubes_rm_cloud.pkl"
if os.path.exists(output_path):
    print(f"Loading existing results from {output_path}")
    with open(output_path, "rb") as f:
        stats_results = pickle.load(f)
else:
    print("No existing results found. Starting fresh.")
    stats_results = {}

# Prepare and compute mean and standard deviation for all variables
for var in variables:
    if var in stats_results:
        print(f"Skipping already processed variable: {var}")
        continue

    #if var not in percentile_bounds:
    #    print(f"No percentile bounds found for variable {var}. Skipping.")
    #    continue
#
    #lower_bound, upper_bound = percentile_bounds[var]
    #print(f"Processing variable: {var} with bounds: lower={lower_bound}, upper={upper_bound}")
    total_sum = 0
    total_sq_sum = 0
    total_count = 0

    # Use multiprocessing to load data
    #max_workers=32
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_minicube_for_stats, path, var): path
            for path in minicube_paths
        }
        with tqdm(total=len(minicube_paths), desc=f"Processing minicubes for {var}") as pbar:
            for future in as_completed(futures):
                data_sum, data_sq_sum, count = future.result()
                total_sum += data_sum
                total_sq_sum += data_sq_sum
                total_count += count
                pbar.update(1)

    # Skip if no data is collected for the variable
    if total_count == 0:
        print(f"No data found for variable {var}. Skipping.")
        continue

    # Compute mean and standard deviation
    try:
        print("Computing mean and standard deviation...")
        mean, std_dev = compute_mean_std(total_sum, total_sq_sum, total_count)
        stats_results[var] = {"mean": mean, "std_dev": std_dev}
        print(f"{var} mean: {mean}, std_dev: {std_dev}")

        # Save progress after each variable
        print(f"Saving progress to {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(stats_results, f)

    except Exception as e:
        print(f"Failed to process variable {var}: {e}")

print(f"Mean and standard deviation saved to: {output_path}")

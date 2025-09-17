import os
import pickle
import xarray as xr
import numpy as np
from tqdm import tqdm
from dataset.prepare_si_dataset import prepare_cube
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

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

# Function to compute bounds for identifying outliers based on percentiles
def compute_bounds(data):
    """
    Compute bounds for identifying outliers based on percentiles.

    Args:
        data (array-like): Input data.

    Returns:
        list: Lower bound (1.5th percentile), upper bound (98.5th percentile).
    """
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values from the input data
    print("Computing percentiles...")
    lower_bound = np.percentile(data, 1.5)
    upper_bound = np.percentile(data, 98.5)
    print(f"Computed percentiles: lower_bound={lower_bound}, upper_bound={upper_bound}")
    return [lower_bound, upper_bound]


# Function to process a single minicube
def process_minicube(path, var):
    try:
        #print(f"Loading cube from: {path}")
        ds = xr.open_zarr(path)
        prepared_ds = prepare_cube(ds["s2l2a"])
        if var in prepared_ds:
            #print(f"Appending data for variable: {var}")
            return prepared_ds[var].values
        else:
            print(f"Variable {var} not found in {path}")
    except Exception as e:
        print(f"Failed to load or prepare variable {var} from {path}: {e}")
    return None

# Load existing results if they exist
output_path = "./98_percentile_mini_cubes.pkl"
if os.path.exists(output_path):
    print(f"Loading existing results from {output_path}")
    with open(output_path, "rb") as f:
        confidence_intervals = pickle.load(f)
else:
    print("No existing results found. Starting fresh.")
    confidence_intervals = {}

# Prepare and compute confidence intervals for all variables
for var in variables:
    if var in confidence_intervals:
        print(f"Skipping already processed variable: {var}")
        continue

    print(f"Processing variable: {var}")
    var_data_list = []

    # Use multiprocessing to load data
    with ProcessPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(process_minicube, path, var): path for path in minicube_paths}
        with tqdm(total=len(minicube_paths), desc=f"Processing minicubes for {var}") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    var_data_list.append(result)
                pbar.update(1)  # Update progress bar after each task completes


    # Skip if no data is collected for the variable
    if not var_data_list:
        print(f"No data found for variable {var}. Skipping.")
        continue

    # Concatenate using NumPy and compute bounds
    try:
        start_time = time.time()
        print("Concatenating data...")
        concatenated_var = np.concatenate(var_data_list, axis=0)  # Use NumPy for concatenation
        print("Flattening data...")
        flat_data = concatenated_var.flatten()  # Flatten the data for computation
        print("Computing bounds...")
        confidence_intervals[var] = compute_bounds(flat_data)
        end_time = time.time()
        print(f"{var} bounds: {confidence_intervals[var]} (Computed in {end_time - start_time:.2f} seconds)")

        # Save progress after each variable
        print(f"Saving progress to {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(confidence_intervals, f)

    except Exception as e:
        print(f"Failed to process variable {var}: {e}")

print(f"Confidence intervals saved to: {output_path}")

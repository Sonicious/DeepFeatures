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

#with open("./min_max_bands_mini_cubes_raw.pkl", "rb") as f:
#    band_min_max = pickle.load(f)

# Function to process a single minicube for min and max values
def process_minicube_for_min_max(path, var):
    #print(f"Processing {path}")

    try:
        da = xr.open_zarr(path)
        da = da.s2l2a.where((da.cloud_mask == 0))
        prepared_ds = prepare_cube(da)

        if var in prepared_ds:
            data = prepared_ds[var]
            # Apply percentile bounds to exclude outliers
            #print('compute min and max values')
            band_min = data.min().compute().item()
            band_max = data.max().compute().item()
            return band_min, band_max
        else:
            print(f"Variable {var} not found in {path}")
    except Exception as e:
        print(f"Failed to load or prepare variable {var} from {path}: {e}")
    return np.inf, -np.inf

# Load existing results if they exist
output_path = "../min_max_mini_cubes_rm_clouds.pkl"
#output_path = "./mini_cube_idx_updated.pkl"
if os.path.exists(output_path):
    print(f"Loading existing results from {output_path}")
    with open(output_path, "rb") as f:
        stats_results = pickle.load(f)
else:
    print("No existing results found. Starting fresh.")
    stats_results = {}


# Prepare and compute min and max for all variables
for var in variables:
    if var in stats_results:
        print(f"Skipping already processed variable: {var} with ranges {stats_results[var]}")
        continue

    global_min = np.inf
    global_max = -np.inf

    # Use multiprocessing to load data
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(process_minicube_for_min_max, path, var): path
            for path in minicube_paths
        }
        with tqdm(total=len(minicube_paths), desc=f"Processing minicubes for {var}") as pbar:
            for future in as_completed(futures):
                local_min, local_max = future.result()
                global_min = min(global_min, local_min)
                global_max = max(global_max, local_max)
                pbar.update(1)

    # Skip if no valid data is found for the variable
    if global_min == np.inf or global_max == -np.inf:
        print(f"No valid data found for variable {var}. Skipping.")
        continue

    # Save the min and max values
    try:
        print(f"Saving min and max for {var}...")
        stats_results[var] = [global_min, global_max]
        print(f"{var} min: {global_min}, max: {global_max}")

        # Save progress after each variable
        print(f"Saving progress to {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(stats_results, f)

    except Exception as e:
        print(f"Failed to process variable {var}: {e}")

print(f"Min and max values saved to: {output_path}")

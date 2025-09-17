import os
import random
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from dataset.prepare_si_dataset import prepare_cube
from utils.utils import drop_if_central_point_nan_or_inf
from ml4xcube.utils import get_chunk_by_index, split_chunk
from ml4xcube.preprocessing import drop_nan_values, fill_nan_values
from statsmodels.stats.outliers_influence import variance_inflation_factor
from concurrent.futures import ProcessPoolExecutor, as_completed

def compute_vif(df):
    df = df.dropna()
    vif_data = pd.DataFrame()
    vif_data["Variable"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data

def extract_center_coordinates(coords):
    center_time = coords['time'][0, coords['time'].shape[1] // 2]
    center_xs = coords['x'][:, coords['x'].shape[1] // 2]
    center_ys = coords['y'][:, coords['y'].shape[1] // 2]
    return center_time, center_xs, center_ys

def create_dataframe_from_cube(ds_path):
    try:
        da = xr.open_zarr(ds_path)
        da = da.s2l2a.where(da.cloud_mask == 0)
        ds = prepare_cube(da)

        # Process cube
        chunk, coords = get_chunk_by_index(ds, 0, block_size=[('time', 20), ('x', 90), ('y', 90)])
        chunk, coords = split_chunk(chunk, coords, sample_size=[("time", 11), ("y", 15), ("x", 15)],
                                    overlap=[("time", 0.), ("y", 14./15.), ("x", 14./15.)])
        vars_train = list(chunk.keys())
        chunk, coords = drop_if_central_point_nan_or_inf(chunk, coords, vars=vars_train)
        chunk, coords = drop_nan_values(chunk, coords, mode='if_all_nan', vars=vars_train)
        valid_mask = {var: ~np.isnan(chunk[var]) for var in vars_train}
        chunk = fill_nan_values(chunk, vars=vars_train, method='sample_mean')

        # Convert to tensor format
        data = np.stack([chunk[var] for var in vars_train], axis=2)  # (samples, time, vars, x, y)
        batch_size, time_len, channels, h, w = data.shape
        central_time = time_len // 2
        central_x = h // 2
        central_y = w // 2
        central_input = data[:, central_time, :, central_x, central_y]

        # Randomly sample 15%
        n_samples = central_input.shape[0]
        n_select = int(0.10 * n_samples)
        if n_select < 2:
            return None  # Skip cubes with insufficient data

        selected_idx = sorted(random.sample(range(n_samples), n_select))
        sampled_data = central_input[selected_idx, :]

        # Create DataFrame for VIF computation
        df = pd.DataFrame(sampled_data, columns=vars_train)
        return compute_vif(df)

    except Exception as e:
        print(f"⚠️ Failed processing {ds_path}: {e}")
        return None


# Set up directory
minicube_dir = "/net/data_ssd/deepfeatures/trainingcubes"
minicube_paths = [os.path.join(minicube_dir, f) for f in os.listdir(minicube_dir) if f.endswith(".zarr")]

# Aggregate all VIF results
vif_accumulator = {}

# Parallel execution
with ProcessPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(create_dataframe_from_cube, path): path for path in minicube_paths}
    with tqdm(total=len(futures), desc="Processing cubes for VIF") as pbar:
        for future in as_completed(futures):
            result = future.result()
            pbar.update(1)
            if result is not None:
                for _, row in result.iterrows():
                    var, vif = row["Variable"], row["VIF"]
                    if var not in vif_accumulator:
                        vif_accumulator[var] = []
                    vif_accumulator[var].append(vif)

# Compute mean VIF per variable
#mean_vif = pd.DataFrame({
#    "Variable": list(vif_accumulator.keys()),
#    "Mean_VIF": [np.mean(vif_accumulator[var]) for var in vif_accumulator]
#}).sort_values(by="Mean_VIF", ascending=True)


mean_vif = pd.DataFrame([
    {
        "Variable": var,
        "Mean_VIF": np.mean([v for v in vifs if not np.isnan(v) and not np.isinf(v)]) if any(not np.isnan(v) and not np.isinf(v) for v in vifs) else np.nan,
        "Num_NaNs": sum(np.isnan(v) for v in vifs),
        "Num_INFs": sum(np.isinf(v) for v in vifs),
    }
    for var, vifs in vif_accumulator.items()
]).sort_values(by="Mean_VIF", ascending=True)


# Save and print
output_path = "mean_vif.csv"
mean_vif.to_csv(output_path, index=False)
print(f"✅ Mean VIFs saved to {output_path}")
print(mean_vif)

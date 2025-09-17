import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
# === Configuration ===
limit_years = True  # ← set to True if you want to keep only 2020–2022

print("Loading original Zarr dataset...")
ds = xr.open_zarr("/net/data_ssd/deepfeatures/sciencecubes_processed/000000_2.zarr", consolidated=True)
print("Dataset loaded.")


# Step 1: Crop 7 pixels from each spatial border (x and y)
print("Cropping spatial borders (x and y)...")
#ds_spatial = ds.isel(
#    x=slice(7, -7),
#    y=slice(7, -7)
#)

y = slice(9, 273)
x = slice(260, 524)
ds_spatial = ds.isel(y=y, x=x)

print(f"New spatial shape: x={ds_spatial.sizes['x']}, y={ds_spatial.sizes['y']}")

# Step 2 (optional): Filter time to 2020–2022
if limit_years:
    print("Filtering time range to 2020-01-01 through 2022-12-31...")
    ds_time = ds_spatial.sel(time=slice("2019-10-16", "2021-03-10"))
    print(f"Time steps after filtering: {ds_time.sizes['time']}")
else:
    print("Skipping time filtering — using full available time range.")
    ds_time = ds_spatial

# Step 3: Remove 5 timesteps from each temporal border
print("Cropping 5 timesteps from each temporal border...")
if ds_time.sizes["time"] > 10:
    ds_final = ds_time.isel(time=slice(5, -5))
    print(f"Final number of time steps: {ds_final.sizes['time']}")
else:
    raise ValueError("Not enough time steps to remove 5 from each end.")

# Step 4: Drop band/variable dims if they exist
drop_dims = [dim for dim in ["band", "bands", "variables", "features"] if dim in ds_final.dims]
if drop_dims:
    print(f"Dropping unnecessary dimension(s): {drop_dims}")
    ds_final = ds_final.isel({dim: 0 for dim in drop_dims})  # Keep first slice
    ds_final = ds_final.drop_vars(drop_dims, errors="ignore")

# Step 5: Create single feature DataArray (7 features, time, y, x)
print("Creating empty feature cube with shape (feature=7, time, y, x)...")
shape = (7, ds_final.sizes["time"], ds_final.sizes["y"], ds_final.sizes["x"])
data = np.full(shape, np.nan, dtype=np.float32)

da = xr.DataArray(
    data,
    dims=("feature", "time", "y", "x"),
    coords={
        "feature": [f"feature_{i}" for i in range(7)],
        "time": ds_final["time"],
        "y": ds_final["y"],
        "x": ds_final["x"]
    }
)

# Step 6: Define chunking (aligned with your write logic)
print("Defining chunking scheme (feature=7, time=1, y=full, x=full)...")
encoding = {
    "features": {
        "chunks": (7, 1, ds_final.sizes["y"], ds_final.sizes["x"])
    }
}

# Step 7: Save to Zarr
zarr_path = "/net/data_ssd/deepfeatures/sciencecubes_processed/000000_feature_cube_small.zarr"
print(f"Saving feature cube to Zarr at: {zarr_path} ...")
ds_out = xr.Dataset({"features": da})
ds_out = ds_out.drop_vars("feature")  # prevent conflict with region writing later

print(ds_out)

ds_out.to_zarr(
    zarr_path,
    mode="w",
    encoding=encoding
)
print("✅ Zarr feature cube saved successfully.")

print("Timestamps in feature cube:")
print(da["time"].values)

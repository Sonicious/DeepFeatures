import pickle
import numpy as np
import xarray as xr
from prepare_si_dataset import prepare_cube

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



da = xr.open_zarr('/net/scratch/mreinhardt/testcube.zarr')["s2l2a"]
ds = prepare_cube(da)


# Clamp each variable in the dataset
for var in ds.data_vars:
    if var in combined_bounds:
        lower_bound, upper_bound = combined_bounds[var]
        print(f"Clamping {var} with bounds: {lower_bound} to {upper_bound}")
        #ds[var] = xr.where(ds[var] < lower_bound, lower_bound,
        #         xr.where(ds[var] > upper_bound, upper_bound, ds[var]))
        ds[var] = xr.where((ds[var] < lower_bound) | (ds[var] > upper_bound), np.nan, ds[var])
    else:
        print(f"No bounds found for {var}, skipping clamping.")
# The dataset `ds` now has values clamped within the specified boundaries.
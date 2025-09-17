import numpy as np
import xarray as xr
#from dataset.prepare_dataarray import compute_spectral_indices
from prepare_dataarray import prepare_spectral_data

"""np.set_printoptions(threshold=np.inf)"""

#ds = xr.open_zarr('/net/data_ssd/deepfeatures/sciencecubes/000000.zarr')
#ds = xr.open_zarr('/net/data_ssd/deepfeatures/sciencecubes_processed/000000.zarr')['s2l2a']
#ds = xr.open_zarr('/net/data_ssd/deepfeatures/sciencecubes_processed/000000_2.zarr')['s2l2a']
ds = xr.open_zarr('/net/data_ssd/deepfeatures/trainingcubes/000174.zarr')['s2l2a']
print(ds)

esa = ds['esa_wc']#.isel(time_esa_wc=0)
print(esa)
#y = slice(9, 273)
#x = slice(260, 524)
#ds_spatial = ds.isel(y=y, x=x)
#print(ds_spatial)
#da = ds_spatial.sel(time='2020-05-21')  # shape: (band=3, y, x)
da = ds.sel(time='2019-06-17')  # shape: (band=3, y, x)
print(ds)
# Count finite values across all bands
num_valid = np.isfinite(da.values).sum()
total = da.size
percent_valid = 100 * num_valid / total

print(f"Valid pixels: {num_valid} / {total} ({percent_valid:.2f}%)")

# === Select RGB bands using names ===
rgb = da.sel(band=["B04", "B03", "B02"])  # Red, Green, Blue
rgb.compute()

print(rgb)

import matplotlib.pyplot as plt
import numpy as np

# === Squeeze time if needed ===
rgb_squeezed = rgb.squeeze("time")  # dims: (band, y, x)

# === Convert to numpy and normalize ===
rgb_np = rgb_squeezed.transpose("y", "x", "band").values
rgb_np = rgb_np / np.nanpercentile(rgb_np, 98)
rgb_np = np.clip(rgb_np, 0, 1)

# === Spatial extent ===
x_coords = rgb.coords["x"].values
y_coords = rgb.coords["y"].values
extent = [x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]]  # [left, right, bottom, top]

# === Plot ===
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(rgb_np, extent=extent)

# === Labels, title, font sizes ===
#ax.set_title("Sentinel-2 RGB", fontsize=15)
#ax.set_xlabel("x", fontsize=13)
#ax.set_ylabel("y", fontsize=13)
#ax.tick_params(axis='both', labelsize=12)
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_xticks([])
ax.set_yticks([])

# === Increase scientific notation fontsize if needed ===
#ax.yaxis.get_offset_text().set_fontsize(12)
#ax.xaxis.get_offset_text().set_fontsize(12)
ax.yaxis.offsetText.set_visible(False)
ax.xaxis.offsetText.set_visible(False)

plt.tight_layout()
plt.show()

#cloud_mask = ds.cloud_mask
#ds = ds.s2l2a.where((ds.cloud_mask == 0))
#nan_fraction = ds.isnull().mean(dim=("band", "y", "x"))
#ds = ds.sel(time=nan_fraction <= 0.999985)
#chunks = {"time": 56, "y": 250, "x": 250}
#ds = ds.chunk(chunks)
#ds.to_zarr('/net/data_ssd/deepfeatures/sciencecubes_processed/000000_2.zarr', mode='w')

# Count non-NaN points across spatial dimensions and bands
#valid_data_count = ds.notnull().sum(dim=["band", "y", "x"])

# Select only timestamps with at least 100 valid data points
#ds = ds.sel(time=valid_data_count >= 150)


ds = prepare_spectral_data(ds, to_ds=True)
#ds = prepare_cube(ds)


#has_nan = ds.isnull().any().compute()
#print(f"Contains NaNs: {has_nan}")

#ds.to_zarr('/net/data_ssd/deepfeatures/sciencecubes_processed/0000002.zarr', mode='w')


# Crop boundaries: remove 7 from start and end of x and y, and 5 from time
#cropped = ds.isel(
#    x=slice(7, -7),
#    y=slice(7, -7),
#    time=slice(5, -5)
#)
#
#print(cropped)

#ds = prepare_cube(ds)

# Narrow to first 86 timesteps
#ds = cropped.isel(time=slice(0, 86))
#
#nan_percentages_per_time = {}
#
## Select the B01 band
#data = ds.sel(band='B01')
#
## Total number of pixels per timestep (x * y)
#total_elements_per_timestep = data.sizes['x'] * data.sizes['y']
#
## Count NaNs per timestep
#nan_counts = xr.where(data.isnull(), 1, 0).sum(dim=['x', 'y']).compute()
#
## Convert to percentage
#percentage_per_timestep = nan_counts / total_elements_per_timestep * 100
#
## Store in dictionary
#nan_percentages_per_time['B01'] = percentage_per_timestep
#
## Print both absolute and percentage values
#nan_array = nan_percentages_per_time['B01']
#for time, nan_count, pct in zip(nan_array['time'].values, nan_counts.values, nan_array.values):
#    print(f"{str(time)}: {int(nan_count):,} NaNs ({pct:.2f}%)")
#
## Optionally: average
#avg_pct = float(percentage_per_timestep.mean())
#print(f"\nAverage % NaNs across timesteps: {avg_pct:.2f}%")
#
#
## Total number of elements across all timesteps
#total_elements = data.sizes['time'] * data.sizes['x'] * data.sizes['y']
#
## Count total NaNs across time, x, y
#total_nan_count = xr.where(data.isnull(), 1, 0).sum(dim=['x', 'y', 'time']).compute()
#
## Compute percentage of non-NaN (available) data
#available_percentage = 100 - (total_nan_count / total_elements * 100)
#
#print(f"\nOverall available data (non-NaN): {available_percentage:.2f}%")
#print(f"\nOverall available data (non-NaN): {total_nan_count:.2f}%")
#print(ds)



# Extract cloud mask and lccs
#lccs_class = da['lccs_class']
#cloud_mask = da.cloud_mask
#da = da.s2l2a.where((da.cloud_mask == 0))
#print(da.chunks)
#print(da.dims)

# Drop time slices where everything is NaN across all bands and pixels
#da = da.dropna(dim="time", how="all")
#chunks = {"time": 53, "y": 250, "x": 250}
#da = da.chunk(chunks)




# Total number of valid pixels (non-NaN)
#total = np.sum(~np.isnan(cloud_mask.values))

# Compute percentages using logical masks
#percentages = {
#    0: 100 * np.sum(cloud_mask.values == 0) / total,
#    1: 100 * np.sum(cloud_mask.values == 1) / total,
#    2: 100 * np.sum(cloud_mask.values == 2) / total,
#    3: 100 * np.sum(cloud_mask.values == 3) / total,
#}

# Optional: round and label
#labels = {
#    0: 'clear',
#    1: 'thick_cloud',
#    2: 'thin_cloud',
#    3: 'cloud_shadow'
#}
#
#for key, pct in percentages.items():
#   print(f"{labels[key]} ({key}): {pct:.2f}%")

# Count non-NaN values after cloud mask was applied (i.e., only clear-sky data remains)
#non_nan_count = ds.notnull().sum().item()
#print(f"Total number of non-NaN values after cloud masking: {non_nan_count:,}")

#ds = prepare_cube(da_cleaned)
#ds = prepare_cube(da['s2l2a'])


# Align LCCS with ds["time"] by repeating each year's label across the full year
"""llccs_years = pd.to_datetime(lccs_class['time_lccs'].values).year
ds_years = pd.to_datetime(ds['time'].values).year

expanded_lccs = []

# Get spatial shape
spatial_shape = lccs_class.shape[1:]  # (y, x)

for year in ds_years:
    idx = np.where(lccs_years == year)[0]
    if len(idx) == 0:
        # Fill with NaNs if the year is not available
        mask = xr.full_like(lccs_class.isel(time_lccs=0), np.nan)
    else:
        mask = lccs_class.isel(time_lccs=idx[0])
    expanded_lccs.append(mask)


# Concatenate to match time axis
expanded_lccs = xr.concat(expanded_lccs, dim='time')
expanded_lccs = expanded_lccs.assign_coords(time=ds['time'])
ds['lccs_class'] = expanded_lccs

# Mapping: original class → simplified one-digit class
ccs_value_remap = {
    10: 1, 11: 1, 12: 1, 20: 1, 30: 1,                   # cropland
    40: 2, 110: 2, 130: 2, 140: 2,                       # grassland
    50: 3, 60: 3, 61: 3, 62: 3, 70: 3, 71: 3, 72: 3,
    80: 3, 81: 3, 82: 3, 90: 3,                          # tree
    100: 4, 120: 4, 121: 4, 122: 4,                      # shrub
    150: 5, 151: 5, 152: 5, 153: 5, 200: 5, 201: 5, 202: 5,  # low vegetation
    160: 6, 170: 6, 180: 6,                              # wetlands
    190: 7,                                              # urban
    210: 8,                                              # water
    220: 9,                                              # snow_and_ice
    0: np.nan                                            # no_data → NaN
}

# Class label mapping (used for metadata)
lccs_label_remap = {
    1: 'cropland',
    2: 'grassland',
    3: 'tree',
    4: 'shrub',
    5: 'low_vegetation',
    6: 'wetlands',
    7: 'urban',
    8: 'water',
    9: 'snow_and_ice'
}

# Apply remapping
ds['lccs_class'] = xr.apply_ufunc(
    np.vectorize(lccs_value_remap.get),
    ds['lccs_class'],
    dask='parallelized',
    output_dtypes=[float]  # float for NaN
)

# Add metadata
ds['lccs_class'].attrs['flag_values'] = np.array(list(lccs_label_remap.keys()))
ds['lccs_class'].attrs['flag_meanings'] = " ".join(lccs_label_remap.values())"""


# Final check
#print('======================')
#print(ds['lccs_class'])
import xarray as xr
import numpy as np

# === CONFIGURATION ===
feature_cube_path = '/net/data_ssd/deepfeatures/sciencecubes_processed/000000_feature_cube.zarr'
s2_cube_path = '/net/data_ssd/deepfeatures/sciencecubes_processed/000000.zarr'
selected_feature = 'feature_0'
selected_band = 'B01'
n_timesteps = 86

# === Load feature cube and extract time ===
print("🔹 Loading feature data cube...")
ds_f = xr.open_zarr(feature_cube_path, chunks=None)
feat = ds_f['features'].isel(time=slice(0, n_timesteps))
feat = feat.assign_coords(feature=[f"feature_{i}" for i in range(ds_f.sizes['feature'])])
feat = feat.sel(feature=selected_feature)
feature_times = feat['time']

# === Load and align Sentinel-2 cube ===
print("🔹 Loading initial Sentinel-2 cube...")
ds_i = xr.open_zarr(s2_cube_path, consolidated=True)
s2 = ds_i['s2l2a'].sel(time=feature_times)
s2 = s2.isel(x=slice(7, -7), y=slice(7, -7)).sel(band=selected_band)

# === Compute valid pixel counts ===
print("🔹 Computing valid pixel counts...")
valid_feat = feat.count(dim=['x', 'y'])
valid_init = s2.count(dim=['x', 'y'])

# === Identify and exclude mismatching timestamps ===
print("🔍 Checking for time mismatches...")
matching_time_mask = []
for idx, (coord_time, val_f, val_i) in enumerate(zip(feature_times.values, valid_feat, valid_init)):
    val_f = val_f.compute()
    val_i = val_i.compute()
    if val_f != val_i:
        print(f"\n#{idx:02d} | coord: {idx} | 📅 {str(coord_time)}")
        print(f"    🔸 feature cube: {val_f:,} valid pixels")
        print(f"    🔸 initial cube: {val_i:,} valid pixels")
        matching_time_mask.append(False)
    else:
        matching_time_mask.append(True)

# Filter data to only matching timestamps
print("🔧 Filtering cubes to aligned timestamps only...")
matching_time_mask = np.array(matching_time_mask)
feat_aligned = feat.isel(time=matching_time_mask)
s2_aligned = s2.isel(time=matching_time_mask)

# === Check valid pixel counts per x and y coordinate ===
print("🔍 Checking valid pixel alignment across coordinates...")

# Count valid pixels per x (aggregated over time and y)
x_valid_feat = feat_aligned.count(dim=['time', 'y']).values
x_valid_s2 = s2_aligned.count(dim=['time', 'y']).values

# Count valid pixels per y (aggregated over time and x)
y_valid_feat = feat_aligned.count(dim=['time', 'x']).values
y_valid_s2 = s2_aligned.count(dim=['time', 'x']).values

# Compare and print mismatches in x
x_diff_mask = x_valid_feat != x_valid_s2
if np.any(x_diff_mask):
    print(f"\n🔸 X mismatch at {np.sum(x_diff_mask)} coordinates:")
    for i, (xf, xs) in enumerate(zip(x_valid_feat, x_valid_s2)):
        if xf != xs:
            print(f"  x={feat_aligned['x'].values[i]}: feature={xf}, initial={xs}")
else:
    print("✅ X coordinates have matching valid pixel counts.")

# Compare and print mismatches in y
y_diff_mask = y_valid_feat != y_valid_s2
if np.any(y_diff_mask):
    print(f"\n🔸 Y mismatch at {np.sum(y_diff_mask)} coordinates:")
    for i, (yf, ys) in enumerate(zip(y_valid_feat, y_valid_s2)):
        if yf != ys:
            print(f"  y={feat_aligned['y'].values[i]}: feature={yf}, initial={ys}")
else:
    print("✅ Y coordinates have matching valid pixel counts.")

print("\n✅ Coordinate-wise validation complete.")

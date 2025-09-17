import xarray as xr
import matplotlib.pyplot as plt
import xarray as xr
from ml4xcube.plotting import plot_slice
from ml4xcube.insights import get_gap_heat_map



# === Open the feature cube ===
zarr_path = "/net/data_ssd/deepfeatures/sciencecubes_processed/000000_feature_cube_small.zarr"
ds = xr.open_zarr(zarr_path, consolidated=True)
print("Time range:")
print(f"  {ds.time.values[0]} to {ds.time.values[-1]}")

print("X range:")
print(f"  {ds.x.values[0]} to {ds.x.values[-1]}")

print("Y range:")
print(f"  {ds.y.values[-1]} to {ds.y.values[0]}")  # Y often decreases top-to-bottom
features = ds["features"]  # shape: (feature, time, y, x)
print("Loaded features with shape:", features.shape)

# === List available dimensions and coordinates ===
print("Feature indices:", features.feature.values)
print("Time range:", str(features.time.values[0]), "to", str(features.time.values[-1]))

# === Choose feature index and time slice ===
feature_idx = 0  # Change to 1, 2, ..., 6 for other features
time_idx = 39    # Change to explore other timestamps

# === Plot spatial map of a feature at a specific time ===
feat_name = f"Feature {feature_idx}"
time_str = str(features.time.values[time_idx])[:10]

da = features.isel(feature=feature_idx, time=time_idx)
da.plot(cmap="viridis")
plt.title(f"{feat_name} at {time_str}")
plt.show()

# === Optional: Plot a time series at a specific pixel ===
y_idx = 100
x_idx = 100

time_series = features.isel(feature=feature_idx, y=y_idx, x=x_idx)
time_series.plot(marker="o")
plt.title(f"{feat_name} Time Series at y={y_idx}, x={x_idx}")
plt.show()

time = 64
feature = 0

# === Loop through all features ===
for feature_idx in range(features.sizes["feature"]):
    frame = features.isel(feature=feature_idx, time=time_idx)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = frame.plot(ax=ax, cmap="viridis", add_colorbar=False)

    # Custom colorbar with min/mid/max only
    vmin = float(im.get_clim()[0])
    vmax = float(im.get_clim()[1])
    vmid = (vmin + vmax) / 2
    cbar = plt.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_ticks([vmin, vmid, vmax])
    cbar.ax.set_yticklabels([f"{vmin:.2f}", f"{vmid:.2f}", f"{vmax:.2f}"], fontsize=12)  # colorbar tick fontsize

    # Titles and labels
    time_str = str(frame.time.values)[:10]
    #ax.set_title(f"Feature {feature_idx} at time index {time_idx} ({time_str})", fontsize=14)
    ax.set_title(f"Feature {feature_idx}", fontsize=15)
    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("y", fontsize=13)

    # Axis tick label font size
    ax.tick_params(axis='both', labelsize=12)

    # === Increase scientific notation fontsize on y-axis ===
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_fontsize(12)

    plt.tight_layout()
    plt.show()


"""import numpy as np
if np.isnan(frame.values).all():
    print(f"⚠️ All values for feature {feature} at time index {time} are NaN.")
else:
    print(f"✅ Feature {feature} at time {time} contains valid data.")


# Extract feature 0 as a DataArray (shape: time, y, x)
feat = ds.isel(feature=0).squeeze(drop=True).compute()
print(feat)

# Check: feat should now be a DataArray with dims ('time', 'y', 'x')

# Compute the gap heat map
gap_map = get_gap_heat_map(feat['features'], count_dim='time')

# Optional: wrap it into a dataset with a name
gap_ds = gap_map.to_dataset(name='feature_0')

# Visualize the result
plot_slice(gap_ds, var_to_plot ='feature_0', xdim='x', ydim='y', color_map='plasma')   # Using ml4xcube
# OR: gap_map.plot.imshow()          # Using plain xarray"""
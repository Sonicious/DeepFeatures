import xarray as xr
import numpy as np

# === Konfiguration ===
limit_years = True  # False: alle Zeitstufen behalten
zarr_path = "/net/data_ssd/deepfeatures/sciencecubes_processed/000000_2.zarr"
output_path = "/net/data_ssd/deepfeatures/sciencecubes_processed/000000_features.zarr"

print("Loading original Zarr dataset...")
ds = xr.open_zarr(zarr_path, consolidated=True)
print("Dataset loaded.")

# === Step 1: Spatial Cropping (7 Pixel an allen Seiten) ===
#ds_spatial = ds.isel(x=slice(7, -7), y=slice(7, -7))
y = slice(9, 273)
x = slice(260, 524)
ds_spatial = ds.isel(y=y, x=x)
print(f"New spatial shape: x={ds_spatial.sizes['x']}, y={ds_spatial.sizes['y']}")

# === Step 2 (optional): Zeitfilter ===
if limit_years:
    ds_time = ds_spatial.sel(time=slice("2019-10-16", "2021-02-18"))
else:
    ds_time = ds_spatial

# === Step 3: Temporaler Zuschnitt (5 Zeitschritte von Anfang & Ende) ===
if ds_time.sizes["time"] > 10:
    ds_final = ds_time.isel(time=slice(5, -5))
else:
    raise ValueError("Nicht genug Zeitstufen, um 5 vorne und hinten abzuschneiden.")

# === Step 4: Koordinaten und Shape extrahieren ===
coords = {dim: ds_final.coords[dim] for dim in ["time", "y", "x"]}
shape = (7, coords["time"].size, coords["y"].size, coords["x"].size)  # 7 Features

# === Step 5: Leeres Feature-Array anlegen ===
print("Creating empty feature DataArray...")
feature_array = xr.DataArray(
    data=np.full(shape, np.nan, dtype=np.float32),
    dims=("feature", "time", "y", "x"),
    coords={
        "feature": [f"feature_{i}" for i in range(7)],
        "time": coords["time"],
        "y": coords["y"],
        "x": coords["x"]
    },
    name="features"
)

# === Step 6: Chunking-Definition ===
encoding = {
    "features": {
        "chunks": (1, 493, 97, 97)
    }
}

# === Step 7: Speichern ===
print(f"Saving feature DataArray to Zarr: {output_path}")
feature_array.to_dataset(name="features").to_zarr(
    output_path,
    mode="w",
    encoding=encoding
)
print("✅ Feature-Cube gespeichert.")

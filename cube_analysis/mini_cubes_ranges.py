import os
import pickle
import numpy as np
import xarray as xr
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from dataset.prepare_dataarray import prepare_spectral_data

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
base_path = "/net/data/deepfeatures/training/0.1.0"
output_path = "./all_ranges_no_clouds.pkl"

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def is_new_cube_id(name: str) -> bool:
    # matches 0000_0.zarr, 1536_1.zarr, etc.
    return name.endswith(".zarr") and "_" in name


def ensure_band(da: xr.DataArray) -> xr.DataArray:
    if "band" in da.dims:
        return da
    if "index" in da.dims:
        return da.rename({"index": "band"})
    raise ValueError(f"No band/index dim found: {da.dims}")


def process_cube_minmax(cube_path: str) -> dict:
    """
    Worker: load ONE cube fully, apply cloud mask, prepare spectral data once,
    then compute per-band min/max. Returns {band: (min, max)}.
    """
    ds = xr.open_zarr(cube_path)
    try:
        da = ds.s2l2a.where(ds.cloud_mask == 0)

        out = prepare_spectral_data(
            da,
            to_ds=False,
            compute_SI=True,
            load_b01b09=True,
        )
        if out is not None:
            da = out

        da = ensure_band(da)

        # Read full cube into memory once (your "cube is small" assumption)
        da = da.load()

        local = {}
        for band in da.band.values:
            arr = da.sel(band=band).values
            if np.all(np.isnan(arr)):
                continue
            local[band] = (float(np.nanmin(arr)), float(np.nanmax(arr)))

        return local

    finally:
        ds.close()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if os.path.exists(output_path):
    with open(output_path, "rb") as f:
        all_ranges_no_clouds = pickle.load(f)
    print(f"✅ Loaded existing {output_path} ({len(all_ranges_no_clouds)} variables)")
    print(all_ranges_no_clouds)

else:
    print("🆕 Creating all_ranges_no_clouds.pkl from NEW cubes only")

    cube_dirs = sorted(
        d for d in os.listdir(base_path)
        if is_new_cube_id(d) and os.path.isdir(os.path.join(base_path, d))
    )
    if len(cube_dirs) == 0:
        raise RuntimeError("No new-style cubes (*_0.zarr, *_1.zarr) found")

    cube_paths = [os.path.join(base_path, d) for d in cube_dirs]

    global_min = defaultdict(lambda: np.inf)
    global_max = defaultdict(lambda: -np.inf)

    max_workers = 24
    print(f"⚙️ Using {max_workers} workers; 1 cube per task")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_cube_minmax, p): p for p in cube_paths}

        with tqdm(total=len(cube_paths), desc="Processing cubes (1 per process)") as pbar:
            for fut in as_completed(futures):
                cube_path = futures[fut]
                try:
                    local = fut.result()
                except Exception as e:
                    print(f"❌ Failed cube {os.path.basename(cube_path)}: {e}")
                    pbar.update(1)
                    continue

                for band, (mn, mx) in local.items():
                    if mn < global_min[band]:
                        global_min[band] = mn
                    if mx > global_max[band]:
                        global_max[band] = mx

                pbar.update(1)

    all_ranges_no_clouds = {
        band: [float(global_min[band]), float(global_max[band])]
        for band in global_min.keys()
    }

    with open(output_path, "wb") as f:
        pickle.dump(all_ranges_no_clouds, f)

    print(f"✅ Created {output_path} with {len(all_ranges_no_clouds)} variables")

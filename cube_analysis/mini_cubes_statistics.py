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
output_path = "./all_mean_std_no_clouds.pkl"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def is_new_cube_id(name: str) -> bool:
    return name.endswith(".zarr") and "_" in name


def ensure_band(da: xr.DataArray) -> xr.DataArray:
    if "band" in da.dims:
        return da
    if "index" in da.dims:
        return da.rename({"index": "band"})
    raise ValueError(f"No band/index dim found: {da.dims}")


def process_cube_mean_std(cube_path: str) -> dict:
    """
    Worker: process ONE cube.
    Returns {band: (sum, sumsq, count)} for valid (non-NaN) pixels.
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

        # Load cube once (consistent with your min/max script)
        da = da.load()

        local = {}

        for band in da.band.values:
            arr = da.sel(band=band).values
            valid = np.isfinite(arr)

            if not np.any(valid):
                continue

            vals = arr[valid]
            local[band] = (
                float(vals.sum()),
                float((vals ** 2).sum()),
                int(vals.size),
            )

        return local

    finally:
        ds.close()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if os.path.exists(output_path):
    with open(output_path, "rb") as f:
        all_mean_std = pickle.load(f)

    print(f"✅ Loaded existing {output_path} ({len(all_mean_std)} variables)")

    # sort by std (descending)
    sorted_items = sorted(
        all_mean_std.items(),
        key=lambda kv: kv[1][1],   # std is at index 1
        reverse=True
    )

    print("📊 Variables sorted by descending std:")
    for var, (mean, std) in sorted_items:
        print(f"{var:20s}  mean={mean:.6f}  std={std:.6f}")


else:
    print("🆕 Computing mean & std for all variables")

    cube_dirs = sorted(
        d for d in os.listdir(base_path)
        if is_new_cube_id(d) and os.path.isdir(os.path.join(base_path, d))
    )
    if len(cube_dirs) == 0:
        raise RuntimeError("No new-style cubes (*_0.zarr, *_1.zarr) found")

    cube_paths = [os.path.join(base_path, d) for d in cube_dirs]

    global_sum = defaultdict(float)
    global_sumsq = defaultdict(float)
    global_count = defaultdict(int)

    max_workers = 24
    print(f"⚙️ Using {max_workers} workers; 1 cube per task")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_cube_mean_std, p): p for p in cube_paths}

        with tqdm(total=len(cube_paths), desc="Processing cubes (1 per process)") as pbar:
            for fut in as_completed(futures):
                cube_path = futures[fut]
                try:
                    local = fut.result()
                except Exception as e:
                    print(f"❌ Failed cube {os.path.basename(cube_path)}: {e}")
                    pbar.update(1)
                    continue

                for band, (s, ss, n) in local.items():
                    global_sum[band] += s
                    global_sumsq[band] += ss
                    global_count[band] += n

                pbar.update(1)

    # ------------------------------------------------------------------
    # Final mean & std
    # ------------------------------------------------------------------
    all_mean_std = {}
    for band in global_count.keys():
        n = global_count[band]
        if n == 0:
            continue

        mean = global_sum[band] / n
        var = global_sumsq[band] / n - mean ** 2
        std = float(np.sqrt(max(var, 0.0)))  # numerical safety

        all_mean_std[band] = [float(mean), std]

    with open(output_path, "wb") as f:
        pickle.dump(all_mean_std, f)

    print(f"✅ Created {output_path} with {len(all_mean_std)} variables")

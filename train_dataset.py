import os
import h5py
import torch
import random
import numpy as np
import xarray as xr
from typing import Dict, List, Tuple
from time import time

from utils.utils import compute_time_gaps
from dataset.prepare_dataarray import prepare_spectral_data
from dataset.preprocess_sentinel import extract_sentinel2_patches


# -------------------------
# Helpers for new cube layout
# -------------------------
def list_cube_ids(base_path: str) -> List[str]:
    """Return cube ids without '.zarr', e.g. ['0000_0', '0000_1', ...]."""
    ids = []
    for name in os.listdir(base_path):
        if name.endswith(".zarr") and os.path.isdir(os.path.join(base_path, name)):
            ids.append(name[:-5])  # strip ".zarr"
    ids.sort()
    return ids


def group_cube_ids(cube_ids: List[str]) -> Dict[str, List[str]]:
    """
    Group by prefix before '_' so 0000_0 and 0000_1 stay together.
    If a cube id has no '_', the full id is treated as group key.
    """
    groups: Dict[str, List[str]] = {}
    for cid in cube_ids:
        key = cid.split("_")[0] if "_" in cid else cid
        groups.setdefault(key, []).append(cid)
    # ensure deterministic ordering inside each group
    for k in groups:
        groups[k].sort()
    return groups


def split_groups(
    groups: Dict[str, List[str]],
    train_split: float = 0.75,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split by group keys, then expand back to cube ids.
    Remaining (1-train) is split into 2/3 val and 1/3 test.
    """
    rng = random.Random(seed)
    keys = sorted(groups.keys())
    rng.shuffle(keys)

    n_train = int(train_split * len(keys))
    train_keys = keys[:n_train]
    rest_keys = keys[n_train:]

    # val/test split on remaining groups
    rng.shuffle(rest_keys)
    n_val = int((2 / 3) * len(rest_keys))
    val_keys = rest_keys[:n_val]
    test_keys = rest_keys[n_val:]

    def expand(keys_subset: List[str]) -> List[str]:
        out = []
        for k in keys_subset:
            out.extend(groups[k])
        out.sort()
        return out

    return expand(train_keys), expand(val_keys), expand(test_keys)


# -------------------------
# Data handling
# -------------------------
def ensure_band(var: xr.DataArray) -> xr.DataArray:
    if "band" in var.dims:
        return var
    if "index" in var.dims:
        return var.rename({"index": "band"})
    raise ValueError(f"No band-like dim in {var.dims}")


def verify_patches_against_cube(
    da: xr.DataArray,
    patches: np.ndarray,
    coords_out: Dict[str, np.ndarray],
    n_samples: int = 5
):
    """
    da: xarray.DataArray with dims (band/index, time, y, x)
    patches: np.ndarray (N, select_t, bands, h, w) OR (N, bands, select_t, h, w) depending on your extractor
    coords_out: dict with keys 'time','y','x' where each entry is (N, select_t) for time and (N,h)/(N,w) or similar
    """
    print(f"\n🔍 Verifying {n_samples} random patches...")

    da = ensure_band(da)

    N = patches.shape[0]
    sample_ids = np.random.choice(N, size=min(n_samples, N), replace=False)

    for idx in sample_ids:
        t_coords = coords_out["time"][idx]
        y_coords = coords_out["y"][idx]
        x_coords = coords_out["x"][idx]

        da_patch = da.sel(
            time=xr.DataArray(t_coords, dims="time"),
            y=xr.DataArray(y_coords, dims="y"),
            x=xr.DataArray(x_coords, dims="x"),
        ).transpose("time", "band", "y", "x")

        da_patch_np = da_patch.values

        extracted_np = patches[idx]

        # If your extractor returns (bands, time, h, w), align it
        if extracted_np.shape == (da_patch_np.shape[1], da_patch_np.shape[0], *da_patch_np.shape[2:]):
            # (band, time, y, x) -> (time, band, y, x)
            extracted_np = np.transpose(extracted_np, (1, 0, 2, 3))

        is_equal = np.allclose(da_patch_np, extracted_np, equal_nan=True)
        print(f"🧪 Patch {idx}: {'✅ MATCH' if is_equal else '❌ MISMATCH'} | da={da_patch_np.shape} ext={extracted_np.shape}")

    print("🔁 Verification complete.\n")


def _pick_s2_dataarray(ds: xr.Dataset) -> xr.DataArray:
    """
    Try common variable names; fall back to first data_var.
    Adjust this if your dataset naming changed.
    """
    for cand in ["s2l2a", "s2", "S2", "data", "reflectance"]:
        if cand in ds.data_vars:
            return ds[cand]
    # fallback: first variable
    first = list(ds.data_vars)[0]
    return ds[first]


def _pick_cloud_mask(ds: xr.Dataset):
    """Return cloud mask DataArray or None if not present."""
    for cand in ["cloud_mask", "cloudmask", "s2_cloud_mask", "mask", "qa_cloud"]:
        if cand in ds.data_vars:
            return ds[cand]
    return None


def process_cube(cube_id: str, base_path: str, sentinel_set: str = "si"):
    """
    cube_id like '0000_0' (NO '.zarr').
    """
    cube_path = os.path.join(base_path, cube_id + ".zarr")
    print(f"→ {cube_path}")

    ds = xr.open_zarr(cube_path)

    da = _pick_s2_dataarray(ds)
    cloud = _pick_cloud_mask(ds)
    if cloud is not None:
        da = da.where(cloud == 0)

    # time filtering based on valid fraction
    da = ensure_band(da)

    n_total = da.sizes["band"] * da.sizes["y"] * da.sizes["x"]
    threshold = int(n_total * 0.03)
    valid_data_count = da.notnull().sum(dim=["band", "y", "x"])
    da = da.sel(time=valid_data_count >= threshold)

    if da.sizes.get("time", 0) == 0:
        return None, None, None, None

    da = da.chunk({"time": da.sizes["time"], "y": 90, "x": 90})

    if sentinel_set == "si":
        da = prepare_spectral_data(da, to_ds=False, compute_SI=True, load_b01b09=True)
    else:
        da = prepare_spectral_data(da, to_ds=False)

    # coords for patch extraction
    coords = {dim: da.coords[dim].values for dim in da.dims if dim in ["time", "y", "x"]}

    patches, coords_out, valid_mask, _ = extract_sentinel2_patches(
        da.values, coords["time"], coords["y"], coords["x"]
    )

    # make numpy
    if isinstance(patches, torch.Tensor):
        patches = patches.cpu().numpy()
    if isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.cpu().numpy()

    if patches.shape[0] == 0:
        return None, None, None, None

    return patches, coords_out, valid_mask, da


# -------------------------
# Main
# -------------------------
base_path = "/net/data/deepfeatures/training/0.1.0"   # <-- set this to your new folder

sentinel_ds = "si"   # 'si' or 's2' etc.
dataset = "train" # 'train' / 'validate' / 'test'

cube_ids = list_cube_ids(base_path)
print("cube ids received")
groups = group_cube_ids(cube_ids)
print("cube ids grouped")
train_ids, val_ids, test_ids = split_groups(groups, train_split=0.85, seed=42)
print(test_ids)
print(val_ids)
print(train_ids)

if dataset == "validate":
    file_name = f"val_{sentinel_ds}_final.h5"
    selected_ids = val_ids
elif dataset == "test":
    file_name = f"test_{sentinel_ds}_final.h5"
    selected_ids = test_ids
elif dataset == "train":
    file_name = f"train_{sentinel_ds}_final.h5"
    selected_ids = train_ids
else:
    file_name = "file.h5"
    selected_ids = []

print(f"#cubes total={len(cube_ids)} | train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")
print(f"Writing: {file_name} | dataset={dataset} | n_cubes={len(selected_ids)}")

with h5py.File(file_name, "w") as train_file:
    if sentinel_ds == "s2":
        train_shape = (11, 12, 15, 15)
    elif sentinel_ds == "si":
        train_shape = (11, 147, 15, 15)
    else:
        raise ValueError(f"Unknown sentinel_ds={sentinel_ds}")

    mask_shape = train_shape
    time_gap_shape = (10,)

    train_data_dset = train_file.create_dataset(
        "data", shape=(0, *train_shape), maxshape=(None, *train_shape),
        dtype="float32", chunks=True
    )
    train_mask_dset = train_file.create_dataset(
        "mask", shape=(0, *mask_shape), maxshape=(None, *mask_shape),
        dtype="bool", chunks=True
    )
    train_time_gaps_dset = train_file.create_dataset(
        "time_gaps", shape=(0, *time_gap_shape), maxshape=(None, *time_gap_shape),
        dtype="int32", chunks=True
    )

    current_train_size = 0
    batch_size = 1

    for i in range(0, len(selected_ids), batch_size):
        batch = selected_ids[i:i + batch_size]
        print(f"\nProcessing batch: {batch}")

        start = time()
        try:
            sentinel_patches, coords_out, sentinel_mask, da = process_cube(
                batch[0], base_path=base_path, sentinel_set=sentinel_ds
            )
        except Exception as e:
            print(f"⚠️ Skipping {batch[0]} due to error: {e}")
            continue

        if sentinel_patches is None:
            print("⚠️ No patches, skipping.")
            continue

        print(f"time taken: {time() - start:.2f}s to process cube {batch[0]}")
        MAX_PER_CUBE = 40
        batch_chunk_size = sentinel_patches.shape[0]
        if batch_chunk_size <= 0:
            continue

        if batch_chunk_size > MAX_PER_CUBE:
            keep_idx = np.random.choice(batch_chunk_size, size=MAX_PER_CUBE, replace=False)

            sentinel_patches = sentinel_patches[keep_idx]
            sentinel_mask = sentinel_mask[keep_idx]

            # coords_out is a dict of arrays; downsample each entry on axis=0
            coords_out = {k: v[keep_idx] for k, v in coords_out.items()}

            batch_chunk_size = MAX_PER_CUBE
            print(f"🎲 Downsampled batch to {MAX_PER_CUBE} random samples")

        # Resize
        train_data_dset.resize(current_train_size + batch_chunk_size, axis=0)
        train_mask_dset.resize(current_train_size + batch_chunk_size, axis=0)
        train_time_gaps_dset.resize(current_train_size + batch_chunk_size, axis=0)

        # Write
        train_data_dset[current_train_size:current_train_size + batch_chunk_size] = sentinel_patches
        train_mask_dset[current_train_size:current_train_size + batch_chunk_size] = sentinel_mask

        time_gaps = compute_time_gaps(coords_out["time"])
        if isinstance(time_gaps, torch.Tensor):
            time_gaps = time_gaps.cpu().numpy()

        train_time_gaps_dset[current_train_size:current_train_size + batch_chunk_size] = time_gaps

        current_train_size += batch_chunk_size
        print(f"✅ Wrote batch_size={batch_chunk_size} | total={current_train_size}")

    print(f"\nDONE. total samples written: {current_train_size}")

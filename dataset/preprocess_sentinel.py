import torch
import numpy as np
import xarray as xr
from typing import Tuple, Dict

import xarray as xr

import numpy as np
import pandas as pd
from typing import Tuple, Dict


mport numpy as np
import xarray as xr
import pandas as pd

import stackstac
from pystac_client import Client
import planetary_computer as pc
from rasterio.enums import Resampling

def utm_zone_to_epsg(utm: str) -> int:
    zone = int(utm[:-1])
    band = utm[-1].upper()
    return 32600 + zone if band >= 'N' else 32700 + zone

def download_sentinel1_stack(
        bbox_deg,
        start_date,
        end_date,
        collection="sentinel-1-rtc",
        resolution=10,
        utm='35U',
        max_items=5000,
        bands=["vv", "vh"],
        resampling_method=Resampling.bilinear
) -> xr.Dataset:

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    search = catalog.search(
        collections=[collection],
        bbox=bbox_deg,
        datetime=f"{start_date}/{end_date}",
        max_items=max_items
    )

    items = list(search.get_items())
    if not items:
        raise ValueError("No Sentinel-1 items found for the given search parameters.")

    epsg = utm_zone_to_epsg(utm)
    signed_items = [pc.sign(item) for item in items]

    stacked = stackstac.stack(
        signed_items,
        assets=bands,
        resolution=resolution,
        bounds_latlon=bbox_deg,
        xy_coords='center',
        epsg=epsg,
        resampling=resampling_method,
    )

    ds = stacked.to_dataset(name="backscatter")
    ds.attrs = {}
    for var in ds.data_vars:
        ds[var].attrs = {}

    return ds

def match_sentinel1_to_s2_cube(s2: xr.Dataset) -> xr.Dataset:
    """
    Given an S2 cube, download a matching S1 cube with identical x/y coordinates
    and a time range ±7 days around the S2 cube's time extent.
    """
    # Extract spatial and temporal info from s2
    bbox = s2.bbox_wgs84
    s2_x = s2.x.values
    s2_y = s2.y.values

    utm_zone = s2.attrs.get("utm_zone")

    start_time = pd.to_datetime(s2.time.values[0])
    end_time = pd.to_datetime(s2.time.values[-1])
    start_date = (start_time - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (end_time + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    # Download S1 stack
    s1 = download_sentinel1_stack(
        bbox_deg=bbox,
        start_date=start_date,
        end_date=end_date,
        utm=utm_zone
    )

    # Ensure exact match on x/y coordinates
    #common_x = np.intersect1d(s1.x.values, s2_x)
    #common_y = np.intersect1d(s1.y.values, s2_y)

    s1_matched = s1.sel(x=s2_x, y=s2_y)

    # --- Transform to dB scale ---
    s1_db = 10 * np.log10(s1_matched)  # avoid log(0)

    # --- Clip to typical S1 range (-30 to +5 dB) ---
    s1_db_clipped = s1_db.clip(min=-30, max=5)

    # --- Normalize to [0,1] ---
    s1_norm = (s1_db_clipped + 30) / 35  # -30→0, +5→1

    return s1_norm


def extract_s1_patches(
    s2_coords: dict,
    s1_array: np.ndarray,      # (bands, T, H, W)
    s1_times: list,             # length T, datetime-like
    s1_x: list,                 # full x coordinate vector of S1 cube
    s1_y: list,                 # full y coordinate vector of S1 cube
    max_time_diff_days: int = 3
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Extract Sentinel-1 patches aligned to selected Sentinel-2 sample times.

    Args:
        s2_coords: dict with keys:
            - 'time': (N, K) timestamps per S2 sample
            - 'x':    (N, P) coordinate values (x) for each S2 patch
            - 'y':    (N, P) coordinate values (y) for each S2 patch
        s1_array: (bands, T, H, W) S1 data cube
        s1_times: list/array of datetime-like, length T
        s1_x: full x coordinate vector of S1 cube (list)
        s1_y: full y coordinate vector of S1 cube (list)
        max_time_diff_days: tolerance in days for S1↔S2 matching

    Returns:
        s1_patches_np:  (N, bands, time, H_sel, W_sel) NaNs filled
        s1_coords:      dict with 'time', 'x', 'y'
        s1_valid_mask_np: bool array, same shape as s1_patches_np (True where valid data)
    """
    # Convert to np.array for fast comparison
    s1_x = np.array(s1_x)
    s1_y = np.array(s1_y)
    s1_times = np.array(s1_times, dtype='datetime64[ns]')

    N = s2_coords["time"].shape[0]
    bands, T, H_full, W_full = s1_array.shape

    patches = []
    coords_time = []
    coords_x = []
    coords_y = []
    valid_masks = []

    for i in range(N):
        # --- Map coords from S2 → indices in S1 ---
        x_idx = np.array([np.where(s1_x == val)[0][0] for val in s2_coords["x"][i]], dtype=int)
        y_idx = np.array([np.where(s1_y == val)[0][0] for val in s2_coords["y"][i]], dtype=int)

        # print(x_idx, y_idx)

        # --- Temporal selection: match each S2 time to closest S1 time ---
        target_times = pd.to_datetime(s2_coords["time"][i])
        target_times = np.array(target_times, dtype='datetime64[ns]')

        matched_indices = []
        matched_times = []

        for t in target_times:
            diffs = np.abs(s1_times - t)
            min_pos = int(np.argmin(diffs))
            min_diff = diffs[min_pos]
            if min_diff <= np.timedelta64(max_time_diff_days, 'D'):
                matched_indices.append(min_pos)
                matched_times.append(s1_times[min_pos])
            else:
                matched_indices.append(None)
                matched_times.append(np.datetime64('NaT'))

        # --- Extract patch ---
        H_sel = len(y_idx)
        W_sel = len(x_idx)
        patch = np.full((bands, len(target_times), H_sel, W_sel), np.nan, dtype=np.float32)

        for j, s1_idx in enumerate(matched_indices):
            if s1_idx is not None:
                tile = s1_array[:, s1_idx, :, :]
                tile = np.take(tile, y_idx, axis=1)
                tile = np.take(tile, x_idx, axis=2)
                patch[:, j] = tile

        # --- Valid mask before fill ---
        valid_mask = ~np.isnan(patch)


        # --- Fill NaNs with per-sample, per-band mean ---
        means = np.nanmean(patch, axis=(1, 2, 3), keepdims=True)
        means = np.where(np.isnan(means), 0.0, means)
        patch = np.where(valid_mask, patch, means)

        patches.append(patch)
        coords_time.append(np.array(matched_times))
        coords_x.append(s2_coords["x"][i])
        coords_y.append(s2_coords["y"][i])
        valid_masks.append(valid_mask)

    s1_patches_np = np.stack(patches, axis=0)           # (N, bands, T_sel, H, W)
    s1_valid_mask_np = np.stack(valid_masks, axis=0)    # same shape

    print(f's1 valid mask: {s1_valid_mask_np.shape}')
    s1_coords = {
        "time": np.stack(coords_time, axis=0),          # (N, T_sel)
        "x": np.stack(coords_x, axis=0),                # (N, W)
        "y": np.stack(coords_y, axis=0)                 # (N, H)
    }

    return s1_patches_np, s1_coords, s1_valid_mask_np





def verify_patches_against_cube(
    da: xr.DataArray,
    patches: torch.Tensor,
    coords_out: Dict[str, np.ndarray],
    n_samples: int = 5
):
    """
    Verifies extracted patches against the original xarray cube.

    Args:
        da: xarray.DataArray with dims (band, time, y, x)
        patches: torch.Tensor of shape (N, bands, select_t, h, w)
        coords_out: dict with keys 'time', 'y', 'x'
        n_samples: number of random samples to verify
    """
    print(f"\n🔍 Verifying {n_samples} random patches...")

    N = patches.shape[0]
    sample_ids = np.random.choice(N, size=n_samples, replace=False)

    for idx in sample_ids:
        t_coords = coords_out["time"][idx]
        y_coords = coords_out["y"][idx]
        x_coords = coords_out["x"][idx]

        # Select original patch using coordinate values
        da_patch = da.sel(
            time=xr.DataArray(t_coords, dims="time"),
            y=xr.DataArray(y_coords, dims="y"),
            x=xr.DataArray(x_coords, dims="x")
        ).transpose("time", "index", "y", "x")

        print(f'{da_patch.shape}')

        da_patch_np = da_patch.values
        #da_patch_np = da_patch_np[5, 0, :, :]
        extracted_np = patches[idx].numpy()
        #extracted_np = extracted_np[5, 0, :, :]

        vec1 = np.sort(da_patch_np.flatten())
        vec2 = np.sort(extracted_np.flatten())

        is_equal = np.allclose(vec1, vec2, equal_nan=True)

        #is_equal = np.allclose(da_patch_np, extracted_np, equal_nan=True)

        print(f"🧪 Patch {idx}: {'✅ MATCH' if is_equal else '❌ MISMATCH'}")

    print("🔁 Verification complete.\n")

def extract_s2_patches(
    s2_array: np.ndarray,
    time_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    time_win: int = 20,
    h_win: int = 15,
    w_win: int = 15,
    time_stride: int = 3,
    h_stride: int = 5,
    w_stride: int = 5,
    select_t: int = 11
) -> Tuple[torch.Tensor, Dict[str, np.ndarray], torch.Tensor]:
    """
    Extract spatiotemporal patches from Sentinel-2 array using torch,
    randomly select `select_t` of `time_win` timesteps per patch,
    and return a validity mask (True = valid data).

    Args:
        s2_array: np.ndarray (bands, time, height, width)
        time_coords: np.ndarray (T,)
        y_coords: np.ndarray (H,)
        x_coords: np.ndarray (W,)
        time_win: temporal patch size
        h_win: spatial height of patch
        w_win: spatial width of patch
        *_stride: stride for each dimension
        select_t: number of timestamps to keep (≤ time_win)

    Returns:
        patches: (N, bands, select_t, h_win, w_win) torch.Tensor
        coords: dict with keys 'time' (N, select_t), 'y' (N, h_win), 'x' (N, w_win)
        valid_mask: torch.BoolTensor, True where data is valid, shape same as patches
    """
    assert select_t <= time_win, "Cannot select more timestamps than available."

    print("🔧 Converting input array to torch tensor...")
    tensor = torch.from_numpy(s2_array).unsqueeze(0)  # (1, bands, T, H, W)
    bands, T, H, W = s2_array.shape
    print(f"✅ Input shape: bands={bands}, time={T}, height={H}, width={W}")
    Nt = (T - time_win) // time_stride + 1
    Ny = (H - h_win) // h_stride + 1
    Nx = (W - w_win) // w_stride + 1
    print(f"📦 Extracting patches: Nt={Nt}, Ny={Ny}, Nx={Nx}")
    # Extract all patches using unfold
    patches = tensor.unfold(2, time_win, time_stride) \
        .unfold(3, h_win, h_stride) \
        .unfold(4, w_win, w_stride)  # (1, bands, Nt, Ny, Nx, time_win, h_win, w_win)


    patches = patches.squeeze(0).permute(1, 2, 3, 0, 4, 5, 6)  # (bands, Nt, Ny, Nx, time, h, w)
    patches = patches.reshape(-1, bands, time_win, h_win, w_win)  # (N, bands, time, h, w)

    N = patches.shape[0]
    print(f"🧩 Total patches extracted: {N}")
    # Randomly select select_t of time_win timesteps per patch
    print(f"🎲 Selecting {select_t} random timesteps from each 10-frame patch...")
    #random_idx = np.array([np.random.choice(time_win, select_t, replace=False) for _ in range(N)])
    random_idx = np.array([
        np.sort(np.random.choice(time_win, select_t, replace=False))
        for _ in range(N)
    ])
    random_idx_torch = torch.tensor(random_idx, dtype=torch.long)

    # Select corresponding temporal slices using advanced indexing
    idx_batch = torch.arange(N).unsqueeze(1)  # (N, 1)
    selected_patches = patches[idx_batch, :, random_idx_torch]  # (N, bands, select_t, h, w)
    selected_patches = selected_patches.permute(0, 2, 1, 3, 4)


    # === Compute valid mask: True if NOT NaN ===
    print(f"✅ Patch shape after random selection: {selected_patches.shape}")
    valid_mask = ~torch.isnan(selected_patches)  # shape: (N, bands, select_t, h, w)
    print("🧼 Validity mask computed.")

    # === Filter out low-quality patches BEFORE filling ===
    print("🔍 Filtering out low-quality patches...")

    # -- total valid pixels in full patch --
    #valid_per_patch = valid_mask.sum(dim=(2, 3, 4))  # (N, bands)
    #total_pixels = valid_mask.shape[2] * valid_mask.shape[3] * valid_mask.shape[4]
    #enough_valid_total = valid_per_patch >= (0.4 * total_pixels)

    # -- valid center region (3x3 in space, across all timesteps) --
    h_center = valid_mask.shape[3] // 2
    w_center = valid_mask.shape[4] // 2
    t_center = valid_mask.shape[2] // 2
    #centre_region = valid_mask[:, :, :, h_center - 1:h_center + 2,
    #                w_center - 1:w_center + 2]  # (N, bands, select_t, 3, 3)
    #valid_pixel_count = centre_region.sum(dim=(2, 3, 4))  # (N, bands)
    #enough_valid_center = valid_pixel_count >= 10
#
    ## -- check if the most central spatial pixel is valid in at least 2 timesteps --
#
    #central_valid = valid_mask[:, :, :, h_center, w_center]  # (N, bands, select_t)
    #print(f'valid mask shape: {valid_mask.shape}')
    #valid_central_count = central_valid.sum(dim=2)  # (N, bands)
    #enough_valid_central_pixel = valid_central_count >= 2  # (N, bands)
    #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # -- combined valid patch mask --
    #valid_patch_mask = (enough_valid_total & enough_valid_center & enough_valid_central_pixel).any(dim=1)
    valid_patch_mask = valid_mask[:, :, t_center, h_center, w_center].any(dim=1)  # (N, bands)

    # -- apply mask BEFORE filling NaNs --
    selected_patches = selected_patches[valid_patch_mask]
    valid_mask = valid_mask[valid_patch_mask]
    random_idx = random_idx[valid_patch_mask.cpu().numpy()]
    print(f"🧹 Removed {(~valid_patch_mask).sum().item()} invalid patches.")
    print(f"✅ Remaining patches: {selected_patches.shape[0]}")

    # === Fill NaNs only if needed ===
    if not valid_mask.all():
        print("🧪 NaNs detected – filling missing values with sample mean per patch and band...")
        sum_valid = torch.nan_to_num(selected_patches, nan=0.0).sum(dim=(2, 3, 4), keepdim=True)  # (N, bands, 1, 1, 1)
        count_valid = valid_mask.sum(dim=(2, 3, 4), keepdim=True).clamp(min=1)  # avoid division by 0
        mean_per_patch_band = sum_valid / count_valid  # (N, bands, 1, 1, 1)

        # Fill NaNs in-place without cloning
        selected_patches = torch.where(valid_mask, selected_patches, mean_per_patch_band)
        print("✅ NaNs filled with patch-band means.")
    else:
        print("✅ No NaNs found – skipping filling.")

    # === Compute corresponding coordinates ===
    print("🧭 Computing coordinate ranges for all patches...")
    # Get full 3D index grid (Nt, Ny, Nx)
    t_idx, y_idx, x_idx = np.meshgrid(
        np.arange(Nt), np.arange(Ny), np.arange(Nx), indexing='ij'
    )

    # Flatten in the same order as patches were reshaped
    t0_all = (t_idx * time_stride).reshape(-1)[valid_patch_mask.cpu().numpy()]  # (N_valid,)
    y0_all = (y_idx * h_stride).reshape(-1)[valid_patch_mask.cpu().numpy()]
    x0_all = (x_idx * w_stride).reshape(-1)[valid_patch_mask.cpu().numpy()]

    # Now guaranteed to match reshape(patches, bands, -1, ...)
    time_ranges = np.stack([time_coords[t0: t0 + time_win] for t0 in t0_all])
    #for t_range in time_ranges:
    #    print(t_range)
    #print('jfaskdfhsdkufhsduifhseukfhukisdhfukshfkusdjfn')
    selected_time_coords = np.take_along_axis(time_ranges, random_idx, axis=1)
    #for sel_time in selected_time_coords:
    #    print(sel_time)
    #print('jfaskdfhsdkufhsduifhseukfhukisdhfukshfkusdjfn')


    print(f's2 valid mask: {valid_mask.shape}')


    y_ranges = np.stack([y_coords[y0: y0 + h_win] for y0 in y0_all])
    x_ranges = np.stack([x_coords[x0: x0 + w_win] for x0 in x0_all])
    coords = {
        "time": selected_time_coords,  # (N, select_t)
        "y": y_ranges,                 # (N, h_win)
        "x": x_ranges                 # (N, w_win)
    }
    print("🚀 Extraction complete.")
    return selected_patches, coords, valid_mask


# === Dummy data (3 bands, 4 time, 4x4 spatial) ===
"""np.random.seed(42)
data = np.random.rand(2, 4, 2, 3).astype(np.float32)

# === Create xarray.DataArray ===
time = np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], dtype="datetime64")
y = np.array([10, 11])
x = np.array([20, 21, 22])
band = np.array(["red", "green", "blue"])
band = np.array(["red", "green"])

import xarray as xr
da = xr.DataArray(
    data,
    dims=("index", "time", "y", "x"),
    coords={"index": band, "time": time, "y": y, "x": x},
    name="s2"
)

print(da)

# === Extract data and coords ===
s2_array = da.values  # shape (3, 4, 4, 4)
time_coords = da.coords["time"].values
y_coords = da.coords["y"].values
x_coords = da.coords["x"].values

# === Call your function ===
patches, coords, valid_mask = extract_random_temporal_patches(
    s2_array=s2_array,
    time_coords=time_coords,
    y_coords=y_coords,
    x_coords=x_coords,
    time_win=2,
    h_win=2,
    w_win=2,
    time_stride=1,
    h_stride=1,
    w_stride=1,
    select_t=2
)

# === Inspect output ===
print("\n📦 Patch shape:", patches.shape)
print("🧭 First patch time coords:", coords["time"][0])
print("🧭 First patch y coords:", coords["y"][0])
print("🧭 First patch x coords:", coords["x"][0])
print("✅ Valid mask shape:", valid_mask.shape)

verify_patches_against_cube(da, patches, coords)"""
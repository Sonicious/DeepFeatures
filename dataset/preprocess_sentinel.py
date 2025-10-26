import torch
import numpy as np
import xarray as xr
from typing import Tuple, Dict, List
from utils.utils import compute_time_gaps


def ensure_band(var: xr.DataArray) -> xr.DataArray:
    if "band" in var.dims:
        return var
    if "index" in var.dims:
        return var.rename({"index": "band"})
    raise ValueError(f"No band-like dim in {var.dims}")


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

        #y0, y1 = (y_min, y_max) if da.y[0] <= da.y[-1] else (y_max, y_min)
        #x0, x1 = (x_min, x_max) if da.x[0] <= da.x[-1] else (x_max, x_min)
        y0, y1 = y_coords[0], y_coords[-1]
        x0, x1 = x_coords[0], x_coords[-1]
        # Select original patch using coordinate values
        da_patch = da.sel(
            time=xr.DataArray(t_coords, dims="time"),
            y=slice(y0, y1),
            x=slice(x0, x1),
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


def extract_sentinel2_patches(
    s2_array: np.ndarray,
    time_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    time_win: int = 20,
    h_win: int = 15,
    w_win: int = 15,
    time_stride: int = 17,
    h_stride: int = 9,
    w_stride: int = 9,
    select_t: int = 11,
    layout: str = 'BTYX' # TBYX
) -> Tuple[torch.Tensor, Dict[str, np.ndarray], torch.Tensor, bool]:
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
        patches: (N, select_t, bands, h_win, w_win) torch.Tensor
        coords: dict with keys 'time' (N, select_t), 'y' (N, h_win), 'x' (N, w_win)
        valid_mask: torch.BoolTensor, True where data is valid, shape same as patches
    """
    assert select_t <= time_win, "Cannot select more timestamps than available."

    print("🔧 Converting input array to torch tensor...")
    tensor = torch.from_numpy(s2_array).unsqueeze(0)  # (1, bands, T, H, W)
    if layout == 'BTYX': bands, T, H, W = s2_array.shape
    else: T, bands, H, W = s2_array.shape
    if time_win > T: time_win = T

    rm_unvalid = False

    print(f"✅ Input shape: bands={bands}, time={T}, height={H}, width={W}")
    Nt = (T - time_win) // time_stride + 1
    Ny = (H - h_win) // h_stride + 1
    Nx = (W - w_win) // w_stride + 1
    print(f"📦 Extracting patches: Nt={Nt}, Ny={Ny}, Nx={Nx}")
    # Extract all patches using unfold
    if layout == 'BTYX': patches = tensor.unfold(2, time_win, time_stride)
    else: patches = tensor.unfold(1, time_win, time_stride)
    patches = patches.unfold(3, h_win, h_stride) \
        .unfold(4, w_win, w_stride)  # (1, bands, Nt, Ny, Nx, time_win, h_win, w_win) # (1, Nt, bands, Ny, Nx, time_win, h_win, w_win)

    if layout == 'BTYX': patches = patches.squeeze(0).permute(1, 2, 3, 0, 4, 5, 6)  # (bands, Nt, Ny, Nx, time, h, w)
    else: patches = patches.squeeze(0).permute(0, 2, 3, 1, 4, 5, 6)  # (Nt, Ny, Nx, bands, time, h, w)

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


    # === Compute valid mask: True if NOT NaN ===
    print(f"✅ Patch shape after random selection: {selected_patches.shape}")
    valid_mask = ~torch.isnan(selected_patches)  # shape: (N, bands, select_t, h, w)
    print("🧼 Validity mask computed.")

    # === Filter out low-quality patches BEFORE filling ===
    print("🔍 Filtering out low-quality patches...")

    h_center = valid_mask.shape[3] // 2
    w_center = valid_mask.shape[4] // 2
    t_center = valid_mask.shape[1] // 2

    valid_patch_mask = valid_mask[:, t_center, :10, h_center, w_center].any(dim=1)  # (N, bands)

    # -- apply mask BEFORE filling NaNs --
    selected_patches = selected_patches[valid_patch_mask]
    valid_mask = valid_mask[valid_patch_mask]
    random_idx = random_idx[valid_patch_mask.cpu().numpy()]
    print(f"🧹 Removed {(~valid_patch_mask).sum().item()} invalid patches.")
    print(f"✅ Remaining patches: {selected_patches.shape[0]}")

    # === Fill NaNs only if needed ===
    if not valid_mask.all():
        print("🧪 NaNs detected – filling missing values with sample mean per patch and band...")
        sum_valid = torch.nan_to_num(selected_patches, nan=0.0, posinf=0.0, neginf=0.0).sum(dim=(1, 3, 4), keepdim=True)
        count_valid = valid_mask.sum(dim=(1, 3, 4), keepdim=True).clamp(min=1)
        mean_per_patch_band = sum_valid / count_valid  # (N, 1, bands, 1, 1)
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

    selected_time_coords = np.take_along_axis(time_ranges, random_idx, axis=1)


    y_ranges = np.stack([y_coords[y0: y0 + h_win] for y0 in y0_all])
    x_ranges = np.stack([x_coords[x0: x0 + w_win] for x0 in x0_all])
    coords = {
        "time": selected_time_coords,  # (N, select_t)
        "y": y_ranges,                 # (N, h_win)
        "x": x_ranges                 # (N, w_win)
    }

    time_gaps = compute_time_gaps(selected_time_coords)  # (N, 10)
    #time_gaps = torch.where(time_gaps > 2, torch.tensor(2, dtype=time_gaps.dtype), time_gaps)

    #assert time_gaps.ndim == 2 and time_gaps.shape[1] == 10, f"{time_gaps.shape=}"
    gap_mask = (time_gaps.sum(dim=1) < 200)  # (N,)


    removed = (~gap_mask).sum().item()
    if removed:
        print(f"⏱️ Removing {removed} samples with total gaps > 180")
        rm_unvalid = True

    # apply mask to tensors
    selected_patches = selected_patches[gap_mask]
    valid_mask = valid_mask[gap_mask]

    # apply mask to numpy arrays
    idx_np = gap_mask.cpu().numpy()
    coords = {
        "time": coords["time"][idx_np],
        "y": coords["y"][idx_np],
        "x": coords["x"][idx_np],
    }

    print("🚀 Extraction complete.")
    return selected_patches, coords, valid_mask, rm_unvalid





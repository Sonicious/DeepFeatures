import time
import torch
import numpy as np
import xarray as xr
import logging
from typing import Tuple, Dict, List, Optional
from utils.utils import compute_time_gaps

logger = logging.getLogger(__name__)


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
    logger_name: Optional[str] = None,
    time_win: int = 20,
    h_win: int = 15,
    w_win: int = 15,
    time_stride: int = 20,
    h_stride: int = 15,
    w_stride: int = 15,
    select_t: int = 11,
    layout: str = 'BTYX', # TBYX
    max_total_gap = 185,
    inference: bool = False,
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

    log = logging.getLogger(logger_name) if logger_name else logger
    if log.getEffectiveLevel() == logging.INFO:
        log.setLevel(logging.INFO)

    time_start_patch = time.time()
    log.debug("Converting input array to torch tensor")
    tensor = torch.from_numpy(s2_array).unsqueeze(0)  # (1, bands, T, H, W)
    if layout == 'BTYX': bands, T, H, W = s2_array.shape
    else: T, bands, H, W = s2_array.shape
    if time_win > T: time_win = T

    rm_unvalid = False

    log.debug("Input shape: bands=%s time=%s height=%s width=%s", bands, T, H, W)
    Nt = (T - time_win) // time_stride + 1
    Ny = (H - h_win) // h_stride + 1
    Nx = (W - w_win) // w_stride + 1
    log.debug("Extracting patches Nt=%s Ny=%s Nx=%s", Nt, Ny, Nx)
    # Extract all patches using unfold
    if layout == 'BTYX': patches = tensor.unfold(2, time_win, time_stride)
    else: patches = tensor.unfold(1, time_win, time_stride)
    patches = patches.unfold(3, h_win, h_stride) \
        .unfold(4, w_win, w_stride)  # (1, bands, Nt, Ny, Nx, time_win, h_win, w_win) # (1, Nt, bands, Ny, Nx, time_win, h_win, w_win)

    if layout == 'BTYX': patches = patches.squeeze(0).permute(1, 2, 3, 0, 4, 5, 6)  # (bands, Nt, Ny, Nx, time, h, w)
    else: patches = patches.squeeze(0).permute(0, 2, 3, 1, 4, 5, 6)  # (Nt, Ny, Nx, bands, time, h, w)

    patches = patches.reshape(-1, bands, time_win, h_win, w_win)  # (N, bands, time, h, w)

    N = patches.shape[0]
    log.debug(f"Total patches extracted: {N} ({time.time()-time_start_patch:.3f}s)")
    # Randomly select select_t of time_win timesteps per patch
    time_rand_sel = time.time()
    # --- Always build random_idx (identity if select_t == time_win) ---
    if select_t == time_win:
        selected_patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    else:
        random_idx = np.array([
            np.sort(np.random.choice(time_win, select_t, replace=False))
            for _ in range(N)
        ])

        random_idx_torch = torch.from_numpy(random_idx).long()
        idx_batch = torch.arange(N).unsqueeze(1)  # (N, 1)

        # --- ALWAYS take the same code path ---
        selected_patches = patches[idx_batch, :, random_idx_torch]  # (N, bands, select_t, h, w)


    # === Compute valid mask: True if NOT NaN ===
    log.debug(
        "Final patch shape %s elapsed_seconds=%.3f",
        selected_patches.shape,
        time.time() - time_rand_sel,
    )
    time_val = time.time()
    valid_mask = ~torch.isnan(selected_patches)  # shape: (N, bands, select_t, h, w)
    log.debug("Validity mask computed (%.3fs)", time.time() - time_val)

    # === Filter out low-quality patches BEFORE filling ===
    log.debug("Filtering out low low-quality patches")
    time_low = time.time()
    h_center = valid_mask.shape[3] // 2
    w_center = valid_mask.shape[4] // 2
    t_center = valid_mask.shape[1] // 2

    valid_patch_mask = valid_mask[:, t_center, :10, h_center, w_center].any(dim=1)  # (N, bands)

    # -- apply mask BEFORE filling NaNs --
    selected_patches = selected_patches[valid_patch_mask]
    valid_mask = valid_mask[valid_patch_mask]
    if select_t < time_win: random_idx = random_idx[valid_patch_mask.cpu().numpy()]
    log.debug(
        "Removed %s invalid patches in %.3fs",
        (~valid_patch_mask).sum().item(),
        time.time() - time_low,
    )
    log.debug("Remaining patches: %s", selected_patches.shape[0])

    # === Fill NaNs only if needed ===
    time_fill = time.time()
    if not valid_mask.all():
        log.debug(" NaNs detected – filling missing values with sample mean per patch and band...")
        sum_valid = torch.nan_to_num(selected_patches, nan=0.0, posinf=0.0, neginf=0.0).sum(dim=(1, 3, 4), keepdim=True)
        count_valid = valid_mask.sum(dim=(1, 3, 4), keepdim=True).clamp(min=1)
        mean_per_patch_band = sum_valid / count_valid  # (N, 1, bands, 1, 1)
        # Fill NaNs in-place without cloning
        selected_patches = torch.where(valid_mask, selected_patches, mean_per_patch_band)
        log.debug("NaNs filled with patch-band means (%.3fs).", time.time() - time_fill)
    else:
        log.debug("No NaNs found – skipping filling (%.3f).", time.time() - time_fill)

    # === Compute corresponding coordinates ===
    log.debug("Computing coordinate ranges for all patches...")
    time_range = time.time()
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
    if select_t == time_win:selected_time_coords = time_ranges
    else: selected_time_coords = np.take_along_axis(time_ranges, random_idx, axis=1)


    y_ranges = np.stack([y_coords[y0: y0 + h_win] for y0 in y0_all])
    x_ranges = np.stack([x_coords[x0: x0 + w_win] for x0 in x0_all])
    coords = {
        "time": selected_time_coords,  # (N, select_t)
        "y": y_ranges,                 # (N, h_win)
        "x": x_ranges                 # (N, w_win)
    }
    log.debug("Coordinate ranges computed after %.3fs", time.time() - time_range)
    time_gaps_start = time.time()
    time_gaps = compute_time_gaps(selected_time_coords)  # (N, 10)

    gap_mask = (time_gaps.sum(dim=1) < max_total_gap)  # (N,)
    if inference:

        bad_mask = ~gap_mask
        if bad_mask.any():
            time_gaps[bad_mask] = torch.ones_like(time_gaps[bad_mask])
        coords = {
            "time": coords["time"],
            "y": coords["y"],
            "x": coords["x"],
        }

    else:
        removed = (~gap_mask).sum().item()
        if removed:
            log.debug("Removing %s samples with total gaps > %s", removed, max_total_gap)
            rm_unvalid = True

        # apply mask to tensors
        selected_patches = selected_patches[gap_mask]
        valid_mask = valid_mask[gap_mask]
        idx_np = gap_mask.cpu().numpy()
        coords = {
            "time": coords["time"][idx_np],
            "y": coords["y"][idx_np],
            "x": coords["x"][idx_np],
        }
    log.debug("Time gaps computed after %.3f", time.time() - time_gaps_start)

    log.debug("Extraction complete")
    return selected_patches, coords, valid_mask, rm_unvalid


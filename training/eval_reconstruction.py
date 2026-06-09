import pathlib
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dataset.prepare_dataarray import prepare_spectral_data
from dataset.preprocess_sentinel import extract_sentinel2_patches
from model.model import TransformerAE
from utils.utils import compute_time_gaps, extract_center_coordinates


matplotlib.use("Agg")


COLOR_MAP = LinearSegmentedColormap.from_list(
    "",
    [(0.0, "#add8e6"), (0.5, "#4682b4"), (1.0, "#010138")],
)

BASE_PATH = Path("/net/data/deepfeatures/science/0.1.0")
SOURCE_CUBE_ID = "003"
TARGET_TIMESTAMP = np.datetime64("2018-07-05T11:00:31.025000000")
CUDA_DEVICE = "cuda:2"
BATCH_SIZE = 256
FIGURE_DIR = Path(__file__).resolve().parent / "figures" / "feature_evaluation"
X_INDEX_RANGE = (100, 200)
Y_INDEX_RANGE_FROM_BOTTOM = (200, 300)
SPATIAL_PATCH_RADIUS = 7

MODEL_CONFIGS = {
    "si": {
        "label": "with spectral indices",
        "checkpoint_path": Path("../checkpoints/ae-epoch=141-val_loss=4.383e-03.ckpt"),
        "compute_si": True,
        "channels": 147,
        "dbottleneck": 6,
    },
    "no_si": {
        "label": "without spectral indices",
        "checkpoint_path": Path("../checkpoints/no_si3/ae-epoch=87-val_loss=4.342e-03.ckpt"),
        "compute_si": False,
        "channels": 12,
        "dbottleneck": 6,
    },
}

PHYSICAL_BAND_PATTERN = re.compile(r"^B\d{2}A?$")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def feature_roi_to_source_slices(
    source_y_size: int,
    x_index_range,
    y_index_range_from_bottom,
    feature_border: int = SPATIAL_PATCH_RADIUS,
    halo: int = SPATIAL_PATCH_RADIUS,
):
    x0, x1 = x_index_range
    y0_bottom, y1_bottom = y_index_range_from_bottom

    feature_y_size = source_y_size - 2 * feature_border
    if feature_y_size <= 0:
        raise ValueError(f"Source height {source_y_size} is too small for border {feature_border}")

    feature_y_start = feature_y_size - y1_bottom
    feature_y_end = feature_y_size - y0_bottom

    src_x_start = x0 + feature_border - halo
    src_x_end = x1 + feature_border + halo
    src_y_start = feature_y_start + feature_border - halo
    src_y_end = feature_y_end + feature_border + halo

    return slice(src_x_start, src_x_end), slice(src_y_start, src_y_end)


def coord_to_idx(vals, mapping, axis_vals):
    vals = np.asarray(vals)
    idxs = np.empty(vals.shape, dtype=np.int64)
    for j, v in enumerate(vals):
        fv = float(v)
        if fv in mapping:
            idxs[j] = mapping[fv]
        else:
            idxs[j] = int(np.argmin(np.abs(axis_vals - v)))
    return idxs


def create_empty_band_dataset(band_names, xs, ys, target_time, fill_value=np.nan, dtype=np.float32):
    coords = {
        "band": np.asarray(band_names),
        "time": [np.datetime64(target_time)],
        "y": np.asarray(ys),
        "x": np.asarray(xs),
    }
    data = np.full(
        (len(coords["band"]), len(coords["time"]), len(coords["y"]), len(coords["x"])),
        fill_value,
        dtype=dtype,
    )
    da = xr.DataArray(data, coords=coords, dims=("band", "time", "y", "x"), name="bands")
    return xr.Dataset({"bands": da})


def load_source_cube(cube_id: str, *, compute_si: bool) -> xr.DataArray:
    ds = xr.open_zarr(BASE_PATH / f"{cube_id}.zarr")
    da = ds.s2l2a.where(ds.cloud_mask == 0)

    x_slice, y_slice = feature_roi_to_source_slices(
        source_y_size=da.sizes["y"],
        x_index_range=X_INDEX_RANGE,
        y_index_range_from_bottom=Y_INDEX_RANGE_FROM_BOTTOM,
    )
    da = da.isel(y=y_slice, x=x_slice)
    da = da.chunk({"time": 1, "y": 1000, "x": 1000})
    da = prepare_spectral_data(da, to_ds=False, compute_SI=compute_si, load_b01b09=True)
    if "index" in da.dims:
        da = da.rename({"index": "band"})
    return da


def get_target_window(da: xr.DataArray, target_time: np.datetime64) -> xr.DataArray:
    times_ns = da.time.values.astype("datetime64[ns]")
    target_time = np.datetime64(target_time, "ns")
    matches = np.where(times_ns == target_time)[0]
    if len(matches) == 0:
        raise ValueError(f"Timestamp {target_time} not found in source cube")

    target_idx = int(matches[0])
    if target_idx < 5 or target_idx + 5 >= len(times_ns):
        raise ValueError("Target timestamp does not have 5 timesteps before and after")

    return da.isel(time=slice(target_idx - 5, target_idx + 6))


def extract_target_patches(window: xr.DataArray):
    coords = {dim: window.coords[dim].values for dim in ("time", "y", "x")}
    patches_all, coords_all, valid_mask_all, _ = extract_sentinel2_patches(
        window.values,
        coords["time"],
        coords["y"],
        coords["x"],
        time_win=11,
        time_stride=1,
        h_stride=1,
        w_stride=1,
        max_total_gap=195,
        inference=True,
    )
    if patches_all.shape[0] == 0:
        raise ValueError("No valid patches found for the selected timestamp")
    return patches_all, coords_all, valid_mask_all


def load_model(model_cfg: dict, device: torch.device) -> TransformerAE:
    model = TransformerAE(
        dbottleneck=model_cfg["dbottleneck"],
        channels=model_cfg["channels"],
    ).eval()
    checkpoint = torch.load(model_cfg["checkpoint_path"], map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def get_physical_band_names(channel_names) -> list[str]:
    return [str(name) for name in channel_names if PHYSICAL_BAND_PATTERN.match(str(name))]


def reconstruct_bands(model_cfg: dict, cube_id: str, device: torch.device):
    da_source = load_source_cube(cube_id, compute_si=model_cfg["compute_si"])
    channel_names = [str(name) for name in da_source.band.values]
    physical_bands = get_physical_band_names(channel_names)
    band_indices = [channel_names.index(name) for name in physical_bands]

    global_xs = da_source.x.values[SPATIAL_PATCH_RADIUS:-SPATIAL_PATCH_RADIUS]
    global_ys = da_source.y.values[SPATIAL_PATCH_RADIUS:-SPATIAL_PATCH_RADIUS]
    x_to_idx = {float(v): i for i, v in enumerate(global_xs)}
    y_to_idx = {float(v): i for i, v in enumerate(global_ys)}

    window = get_target_window(da_source, TARGET_TIMESTAMP)
    processed_data, coords, valid_mask = extract_target_patches(window)
    model = load_model(model_cfg, device)

    ds_pred = create_empty_band_dataset(physical_bands, global_xs, global_ys, TARGET_TIMESTAMP)
    ds_ref = create_empty_band_dataset(physical_bands, global_xs, global_ys, TARGET_TIMESTAMP)

    _, center_xs, center_ys = extract_center_coordinates(coords)
    time_gaps_s2 = compute_time_gaps(coords["time"])
    n_samples = processed_data.shape[0]

    for start in tqdm(range(0, n_samples, BATCH_SIZE), desc=f"Reconstructing {model_cfg['label']}", unit="batch"):
        end = min(start + BATCH_SIZE, n_samples)

        batch_processed = processed_data[start:end].to(device, dtype=torch.float32)
        batch_mask = valid_mask[start:end].to(device, dtype=torch.bool)
        batch_s2 = time_gaps_s2[start:end].to(device, dtype=torch.int32)

        with torch.no_grad():
            y_all, _ = model(batch_processed, batch_s2)

        t_steps = batch_processed.shape[1]
        height = batch_processed.shape[3]
        width = batch_processed.shape[4]
        ct, cx, cy = t_steps // 2, height // 2, width // 2

        central_in = batch_processed[:, ct, :, cx, cy][:, band_indices]
        central_out = y_all[:, ct, :, cx, cy][:, band_indices]
        central_mask = batch_mask[:, ct, :, cx, cy][:, band_indices]

        bx = center_xs[start:end]
        by = center_ys[start:end]
        x_idx = coord_to_idx(bx, x_to_idx, global_xs)
        y_idx = coord_to_idx(by, y_to_idx, global_ys)

        pred_np = central_out.detach().cpu().numpy().astype(np.float32)
        ref_np = central_in.detach().cpu().numpy().astype(np.float32)
        mask_np = central_mask.detach().cpu().numpy().astype(bool)
        pred_np[~mask_np] = np.nan
        ref_np[~mask_np] = np.nan

        ds_pred["bands"].values[:, 0, y_idx, x_idx] = pred_np.T
        ds_ref["bands"].values[:, 0, y_idx, x_idx] = ref_np.T

    return ds_pred, ds_ref


def compute_per_band_metrics(ds_pred: xr.Dataset, ds_ref: xr.Dataset, eps: float = 1e-6) -> dict[str, dict]:
    metrics = {}
    for band_name in ds_ref.band.values:
        pred = np.asarray(ds_pred["bands"].sel(band=band_name).isel(time=0).values, dtype=float)
        ref = np.asarray(ds_ref["bands"].sel(band=band_name).isel(time=0).values, dtype=float)
        mask = np.isfinite(pred) & np.isfinite(ref)
        if not mask.any():
            metrics[str(band_name)] = {"mae": np.nan, "mape": np.nan}
            continue
        diff = np.abs(pred[mask] - ref[mask])
        mae = float(diff.mean())
        mape = float((diff / np.clip(np.abs(ref[mask]), eps, None)).mean() * 100.0)
        metrics[str(band_name)] = {"mae": mae, "mape": mape}
    return metrics


def safe_metric_token(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4f}".replace(".", "p")


def save_band_figures(ds_si_pred, ds_no_si_pred, ds_ref, metrics_si, metrics_no_si, cube_id: str):
    day_tag = str(np.datetime_as_string(TARGET_TIMESTAMP, unit="D"))
    for band_name in ds_ref.band.values:
        band_name = str(band_name)
        si_img = ds_si_pred["bands"].sel(band=band_name).isel(time=0)
        no_si_img = ds_no_si_pred["bands"].sel(band=band_name).isel(time=0)
        ref_img = ds_ref["bands"].sel(band=band_name).isel(time=0)

        finite_vals = np.concatenate(
            [
                si_img.values[np.isfinite(si_img.values)],
                no_si_img.values[np.isfinite(no_si_img.values)],
                ref_img.values[np.isfinite(ref_img.values)],
            ]
        )
        if finite_vals.size:
            vmin = float(np.min(finite_vals))
            vmax = float(np.max(finite_vals))
        else:
            vmin, vmax = 0.0, 1.0

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

        im0 = axes[0].imshow(
            si_img,
            origin="upper",
            cmap=COLOR_MAP,
            vmin=vmin,
            vmax=vmax,
        )
        axes[0].set_title(
            f"{band_name} recon SI\nMAE={metrics_si[band_name]['mae']:.5f}, "
            f"MAPE={metrics_si[band_name]['mape']:.2f}%"
        )
        axes[0].axis("off")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(
            no_si_img,
            origin="upper",
            cmap=COLOR_MAP,
            vmin=vmin,
            vmax=vmax,
        )
        axes[1].set_title(
            f"{band_name} recon no-SI\nMAE={metrics_no_si[band_name]['mae']:.5f}, "
            f"MAPE={metrics_no_si[band_name]['mape']:.2f}%"
        )
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(
            ref_img,
            origin="upper",
            cmap=COLOR_MAP,
            vmin=vmin,
            vmax=vmax,
        )
        axes[2].set_title(f"{band_name} original")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        out_file = FIGURE_DIR / (
            f"{cube_id}_{day_tag}_{band_name}"
            f"__si_mae_{safe_metric_token(metrics_si[band_name]['mae'])}"
            f"__si_mape_{safe_metric_token(metrics_si[band_name]['mape'])}"
            f"__no_si_mae_{safe_metric_token(metrics_no_si[band_name]['mae'])}"
            f"__no_si_mape_{safe_metric_token(metrics_no_si[band_name]['mape'])}.png"
        )
        fig.savefig(out_file, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {out_file}")


def main():
    cube_id = SOURCE_CUBE_ID
    device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else "cpu")

    print(f"Source cube id: {cube_id}")
    print(f"Source path:    {BASE_PATH / f'{cube_id}.zarr'}")
    print(f"SI checkpoint:  {MODEL_CONFIGS['si']['checkpoint_path']}")
    print(f"No-SI checkpoint: {MODEL_CONFIGS['no_si']['checkpoint_path']}")
    print(f"Timestamp:      {np.datetime_as_string(TARGET_TIMESTAMP, unit='ns')}")
    print(f"Device:         {device}")
    print(f"Figure dir:     {FIGURE_DIR}")
    print(f"ROI x indices:  {X_INDEX_RANGE[0]}:{X_INDEX_RANGE[1]}")
    print(
        "ROI y indices:  "
        f"{Y_INDEX_RANGE_FROM_BOTTOM[0]}:{Y_INDEX_RANGE_FROM_BOTTOM[1]} from bottom "
        "(converted from reversed visual axis)"
    )

    ds_si_pred, ds_ref = reconstruct_bands(MODEL_CONFIGS["si"], cube_id, device)
    ds_no_si_pred, ds_ref_no_si = reconstruct_bands(MODEL_CONFIGS["no_si"], cube_id, device)

    if not np.array_equal(ds_ref.band.values, ds_ref_no_si.band.values):
        raise ValueError("Band coordinates differ between SI and no-SI runs")
    if not np.array_equal(ds_ref.x.values, ds_ref_no_si.x.values) or not np.array_equal(ds_ref.y.values, ds_ref_no_si.y.values):
        raise ValueError("Spatial coordinates differ between SI and no-SI runs")

    metrics_si = compute_per_band_metrics(ds_si_pred, ds_ref)
    metrics_no_si = compute_per_band_metrics(ds_no_si_pred, ds_ref_no_si)

    print("\nPer-band metrics:")
    for band_name in ds_ref.band.values:
        band_name = str(band_name)
        print(
            f"{band_name}: "
            f"SI MAE={metrics_si[band_name]['mae']:.6f}, SI MAPE={metrics_si[band_name]['mape']:.4f}% | "
            f"no-SI MAE={metrics_no_si[band_name]['mae']:.6f}, no-SI MAPE={metrics_no_si[band_name]['mape']:.4f}%"
        )

    save_band_figures(ds_si_pred, ds_no_si_pred, ds_ref, metrics_si, metrics_no_si, cube_id)


if __name__ == "__main__":
    main()

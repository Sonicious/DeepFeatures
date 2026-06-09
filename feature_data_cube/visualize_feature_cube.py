import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize a feature cube")
    parser.add_argument(
        "--zarr-path",
        default="/net/data/deepfeatures/feature/003.zarr",
        help="Path to the feature cube Zarr store",
    )
    parser.add_argument(
        "--timestamp",
        default="2018-07-05T11:00:31.025000000",
        help="Timestamp to visualize",
    )
    parser.add_argument(
        "--science-zarr-path",
        default="/net/data/deepfeatures/science/0.1.0/003.zarr",
        help="Path to the original Sentinel-2 science cube",
    )
    parser.add_argument(
        "--feature-idx",
        type=int,
        default=0,
        help="Feature index for the single-feature map and time series",
    )
    parser.add_argument(
        "--x-idx",
        type=int,
        default=100,
        help="Pixel x index for the time series plot",
    )
    parser.add_argument(
        "--y-idx",
        type=int,
        default=100,
        help="Pixel y index for the time series plot",
    )
    return parser.parse_args()


def open_feature_cube(zarr_path: Path) -> xr.Dataset:
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path does not exist: {zarr_path}")

    try:
        return xr.open_zarr(zarr_path, consolidated=True)
    except KeyError as exc:
        if exc.args != (".zmetadata",):
            raise
        return xr.open_zarr(zarr_path, consolidated=False)


def open_science_cube(zarr_path: Path) -> xr.Dataset:
    if not zarr_path.exists():
        raise FileNotFoundError(f"Science cube path does not exist: {zarr_path}")

    try:
        return xr.open_zarr(zarr_path, consolidated=True)
    except KeyError as exc:
        if exc.args != (".zmetadata",):
            raise
        return xr.open_zarr(zarr_path, consolidated=False)


def resolve_time_index(times: np.ndarray, target_timestamp: str, *, exact: bool = False) -> int:
    target = np.datetime64(target_timestamp, "ns")
    matches = np.where(times == target)[0]
    if matches.size:
        return int(matches[0])

    if exact:
        raise KeyError(f"Timestamp {target_timestamp} not found in cube")

    nearest_idx = int(np.argmin(np.abs(times - target)))
    print(
        "Requested timestamp not found exactly.",
        f"Using nearest timestamp {times[nearest_idx]} at index {nearest_idx}.",
    )
    return nearest_idx


def stretch_to_unit_interval(channel: np.ndarray) -> np.ndarray:
    valid = np.isfinite(channel)
    if not np.any(valid):
        return np.zeros_like(channel, dtype=np.float32)

    lo, hi = np.percentile(channel[valid], [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.zeros_like(channel, dtype=np.float32)
        out[valid] = 0.5
        return out

    scaled = (channel - lo) / (hi - lo)
    scaled = np.clip(scaled, 0, 1)
    scaled[~valid] = 0
    return scaled.astype(np.float32)


def build_pca_rgb(frame: xr.DataArray):
    cube = np.moveaxis(frame.values, 0, -1)  # (y, x, feature)
    y_size, x_size, feature_count = cube.shape
    pixels = cube.reshape(-1, feature_count)

    valid_mask = np.isfinite(pixels).all(axis=1)
    valid_pixels = pixels[valid_mask]
    if valid_pixels.shape[0] < 3:
        raise ValueError("Not enough valid pixels for PCA.")

    mean = valid_pixels.mean(axis=0, keepdims=True)
    centered = valid_pixels - mean

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3]
    scores = centered @ components.T

    explained_variance = (singular_values ** 2) / max(valid_pixels.shape[0] - 1, 1)
    explained_ratio = explained_variance / explained_variance.sum()

    rgb_flat = np.zeros((pixels.shape[0], 3), dtype=np.float32)
    for idx in range(3):
        rgb_flat[valid_mask, idx] = stretch_to_unit_interval(scores[:, idx])

    rgb = rgb_flat.reshape(y_size, x_size, 3)
    return rgb, explained_ratio[:3]


def build_true_color_rgb(science_ds: xr.Dataset, timestamp: str, feature_ds: xr.Dataset) -> np.ndarray:
    science_time_idx = resolve_time_index(science_ds.time.values, timestamp, exact=True)
    selected_time = science_ds.time.values[science_time_idx]
    print(f"Science cube timestamp: {selected_time} (index {science_time_idx})")

    s2 = science_ds.s2l2a.where(science_ds.cloud_mask == 0).isel(time=science_time_idx)

    # Match the feature cube footprint created in feature_data_cube.py via [7:-7] border crop.
    s2 = s2.sel(y=feature_ds.y, x=feature_ds.x)

    red = s2.sel(band="B04").values
    green = s2.sel(band="B03").values
    blue = s2.sel(band="B02").values

    rgb = np.stack(
        [
            stretch_to_unit_interval(red),
            stretch_to_unit_interval(green),
            stretch_to_unit_interval(blue),
        ],
        axis=-1,
    )
    return rgb


def main():
    args = parse_args()
    ds = open_feature_cube(Path(args.zarr_path))
    science_ds = open_science_cube(Path(args.science_zarr_path))

    print("Time range:")
    print(f"  {ds.time.values[0]} to {ds.time.values[-1]}")
    print("X range:")
    print(f"  {ds.x.values[0]} to {ds.x.values[-1]}")
    print("Y range:")
    print(f"  {ds.y.values[-1]} to {ds.y.values[0]}")

    features = ds["features"]
    print("Loaded features with shape:", features.shape)
    print("Feature indices:", features.feature.values)

    time_idx = resolve_time_index(features.time.values, args.timestamp)
    selected_time = features.time.values[time_idx]
    print(f"Selected timestamp: {selected_time} (index {time_idx})")

    feature_idx = args.feature_idx
    if not 0 <= feature_idx < features.sizes["feature"]:
        raise IndexError(f"feature_idx {feature_idx} out of bounds for {features.sizes['feature']} features")

    y_idx = min(args.y_idx, features.sizes["y"] - 1)
    x_idx = min(args.x_idx, features.sizes["x"] - 1)

    single_feature = features.isel(feature=feature_idx, time=time_idx)
    single_feature.plot(cmap="viridis")
    plt.title(f"Feature {feature_idx} at {str(selected_time)[:19]}")
    plt.show()

    time_series = features.isel(feature=feature_idx, y=y_idx, x=x_idx)
    time_series.plot(marker="o")
    plt.title(f"Feature {feature_idx} Time Series at y={y_idx}, x={x_idx}")
    plt.show()

    for current_feature_idx in range(features.sizes["feature"]):
        frame = features.isel(feature=current_feature_idx, time=time_idx)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = frame.plot(ax=ax, cmap="viridis", add_colorbar=False)

        vmin, vmax = [float(v) for v in im.get_clim()]
        vmid = (vmin + vmax) / 2
        cbar = plt.colorbar(im, ax=ax, orientation="vertical")
        cbar.set_ticks([vmin, vmid, vmax])
        cbar.ax.set_yticklabels([f"{vmin:.2f}", f"{vmid:.2f}", f"{vmax:.2f}"], fontsize=12)

        ax.set_title(f"Feature {current_feature_idx} at {str(selected_time)[:19]}", fontsize=15)
        ax.set_xlabel("x", fontsize=13)
        ax.set_ylabel("y", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.yaxis.get_offset_text().set_fontsize(12)

        plt.tight_layout()
        plt.show()

    pca_frame = features.isel(time=time_idx)
    rgb_image, explained_ratio = build_pca_rgb(pca_frame)
    print(
        "PCA explained variance ratio:",
        ", ".join(f"PC{i + 1}={ratio:.4f}" for i, ratio in enumerate(explained_ratio)),
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb_image, origin="upper")
    ax.set_title(f"PCA RGB Composite at {str(selected_time)[:19]}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()

    true_color_rgb = build_true_color_rgb(science_ds, args.timestamp, ds)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(true_color_rgb, origin="upper")
    ax.set_title(f"Sentinel-2 RGB at {args.timestamp[:19]}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

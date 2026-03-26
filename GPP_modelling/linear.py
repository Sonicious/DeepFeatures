#!/usr/bin/env python3
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Set
from sites import sites_dict  # unchanged from your code
try:
    from .config import CUBE_IDS, IN_DIR, OUT_DIR
except ImportError:
    from config import CUBE_IDS, IN_DIR, OUT_DIR

ROOT_DIR = Path("/net/data/Fluxnet/")
IN_DIR = Path(IN_DIR)
OUT_DIR = Path(OUT_DIR)

# ---------------------------------------------------------------------
# detect FLUX years (unchanged helper)
# ---------------------------------------------------------------------
def detect_flux_years_for_site(site: str, root: Path) -> Set[int]:
    years: Set[int] = set()
    ww_dir = root / "FLUXNET2020-ICOS-WarmWinter"
    if ww_dir.exists():
        for p in ww_dir.glob("FLX_*_FLUXNET2015_FULLSET_DD_*_beta-3.csv"):
            if site in p.name:
                years.update({2017, 2018, 2019, 2020})
                break
    for dpat in ["ICOS_2021_I", "ICOS_2022_I", "ICOS_2023_I", "ICOS_2024_I"]:
        d = root / dpat
        if not d.exists():
            continue
        y = int(dpat.split("_")[1])
        for p in d.glob("ICOSETC_*_FLUXNET_DD_01.csv"):
            if site in p.name:
                years.add(y)
                break
    return {y for y in years if 2017 <= y <= 2020}


def _safe(da: xr.DataArray) -> xr.DataArray:
    return xr.where(np.isfinite(da), da, np.nan)


def _select_central_window(da: xr.DataArray, size: int = 100) -> xr.DataArray:
    """Restrict to the centered spatial window before averaging."""
    if "y" not in da.dims or "x" not in da.dims:
        return da

    y_len = int(da.sizes["y"])
    x_len = int(da.sizes["x"])
    win_y = min(size, y_len)
    win_x = min(size, x_len)

    y0 = max(0, (y_len - win_y) // 2)
    x0 = max(0, (x_len - win_x) // 2)

    return da.isel(y=slice(y0, y0 + win_y), x=slice(x0, x0 + win_x))


def _filter_times_by_min_coverage(da: xr.DataArray, min_fraction: float = 0.01) -> xr.DataArray:
    """Keep only timestamps with at least `min_fraction` valid pixels."""
    spatial_dims = [d for d in ("y", "x") if d in da.dims]
    if not spatial_dims or "time" not in da.dims:
        return da

    valid_any_feature = da.notnull().any(dim="feature")
    frac_valid = valid_any_feature.mean(dim=spatial_dims)
    keep_mask = (frac_valid >= min_fraction).values
    if not np.any(keep_mask):
        return da.isel(time=slice(0, 0))
    return da.isel(time=np.where(keep_mask)[0])

# ---------------------------------------------------------------------
# Fill a single feature for a given year – LINEAR ONLY
# ---------------------------------------------------------------------
def linear_fill_one_year(ts: pd.Series, year: int) -> pd.Series:
    """Daily index → linear interpolate only → no climatology or UCM."""
    # restrict to year
    ts_year = ts[ts.index.year == year]
    if len(ts_year) == 0:
        # return empty year
        idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        return pd.Series(np.nan, index=idx)

    # aggregate duplicates to 1/day
    ts_day = ts_year.groupby(ts_year.index.normalize()).mean()

    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    daily = ts_day.reindex(idx)

    # linear-only interpolation
    filled = daily.interpolate("linear", limit_direction="both")
    return filled

# ---------------------------------------------------------------------
# Fill for all features for a year
# ---------------------------------------------------------------------
def fill_feature_stats_one_year(stat_da: xr.DataArray, year: int) -> xr.DataArray:
    feats = stat_da["feature"].values
    full_idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    out = []
    for f in feats:
        ts = stat_da.sel(feature=f).to_series()
        filled = linear_fill_one_year(ts, year)
        out.append(
            xr.DataArray(
                filled.values.astype("float32"),
                coords={"time": full_idx},
                dims=["time"]
            )
        )
    return xr.concat(out, dim="feature").assign_coords(feature=feats)

for cid in CUBE_IDS:
    print(f"\n→ processing cube {cid}")
    in_path  = IN_DIR / f"{cid}.zarr"
    out_path = OUT_DIR / f"{cid}_linear.zarr"

    ds = xr.open_zarr(in_path, consolidated=True)
    da = _safe(ds["features"]).sortby("time")  # (feature, time, y, x)
    da = _select_central_window(da, size=100)
    da = _filter_times_by_min_coverage(da, min_fraction=0.01)

    if da.sizes.get("time", 0) == 0:
        print("   no timestamps with >=1% valid data in central 100x100 window — skipping.")
        continue

    # spatial stats over the central 100x100 pixels only
    mean_da = da.mean(dim=("y", "x"), skipna=True)
    std_da = da.std(dim=("y", "x"), skipna=True)

    # detect FLUX years
    site_code = sites_dict[cid][0]
    flux_years = detect_flux_years_for_site(site_code, ROOT_DIR)
    years_have = set(int(y) for y in np.unique(mean_da.time.dt.year))
    target_years = sorted(flux_years & years_have)

    if not target_years:
        print("   no matching years — skipping.")
        continue

    # restrict
    mean_da = mean_da.sel(time=mean_da.time.dt.year.isin(target_years))
    std_da = std_da.sel(time=std_da.time.dt.year.isin(target_years))
    print(f"   target years: {target_years}")

    # fill
    yearly_mean = [fill_feature_stats_one_year(mean_da, y) for y in target_years]
    yearly_std = [fill_feature_stats_one_year(std_da, y) for y in target_years]
    filled_mean = xr.concat(yearly_mean, dim="time").sortby("time")
    filled_std = xr.concat(yearly_std, dim="time").sortby("time")

    # optionally sanitize
    filled_mean = xr.where(np.isfinite(filled_mean), filled_mean, np.nan)
    filled_std = xr.where(np.isfinite(filled_std), filled_std, np.nan)

    # write
    ds_out = xr.Dataset({
        "feature_mean_linear": filled_mean,
        "feature_std_linear": filled_std,
    })
    enc = {
        "feature_mean_linear": {"chunks": (64, 180)},
        "feature_std_linear": {"chunks": (64, 180)},
    }
    ds_out.to_zarr(out_path, mode="w", consolidated=True, encoding=enc)

    print(f"   saved: {out_path}")

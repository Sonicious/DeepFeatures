#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from sites import sites_dict

try:
    from .config import (
        CUBE_IDS,
        INCLUDE_STD_FEATURES,
        IN_DIR,
        LOSO_VAL_SITES,
        OUT_DIR,
    )
except ImportError:
    from config import (
        CUBE_IDS,
        INCLUDE_STD_FEATURES,
        IN_DIR,
        LOSO_VAL_SITES,
        OUT_DIR,
    )

# ------------------------
# Paths / config
# ------------------------
ROOT_DIR = Path("/net/data/Fluxnet/")
IN_DIR = Path(IN_DIR)
OUT_DIR = Path(OUT_DIR)

# Feature source to use for GPP dataset creation.
# "linear" reads the interpolated mean/std products created by
# GPP_modelling/linear.py.
FEATURE_SOURCE = "linear"

FEATURE_SOURCE_CONFIG = {
    "linear": {
        "path_template": "{cid}_linear.zarr",
        "mean_var_name": "feature_mean_linear",
        "std_var_name": "feature_std_linear",
        "output_tag": "linear",
    },
}
WINDOW = 90
OVERLAP = 80
assert 0 <= OVERLAP < WINDOW, "OVERLAP muss in [0, WINDOW) liegen"
STRIDE = WINDOW - OVERLAP

# --- radiation handling ---
# Options:
#   "include" -> keep radiation feature and standardize it with RAD_MEAN/RAD_STD
#   "exclude" -> drop radiation feature entirely from inputs
RADIATION_MODE = "exclude"

# --- global standardization constants (used if RADIATION_MODE == "include") ---
RAD_MEAN = 28.8545
RAD_STD = 6.8393

# --- target (GPP) standardization ---
GPP_MEAN = 4.042
GPP_STD = 4.386

# QC threshold (%). Set to 70.0 for >=70% valid, 0.0 to accept all.
QC_THRESH = 70.0


# ------------------------
# Helper functions
# ------------------------
def _site_in_filename(site: str, name: str) -> bool:
    return site in name


def detect_flux_years_for_site(site: str, root: Path) -> Set[int]:
    years: Set[int] = set()
    ww_dir = root / "FLUXNET2020-ICOS-WarmWinter"
    if ww_dir.exists():
        for p in ww_dir.glob("FLX_*_FLUXNET2015_FULLSET_DD_*_beta-3.csv"):
            if _site_in_filename(site, p.name):
                years.update({2017, 2018, 2019, 2020})
                break
    return {y for y in years if 2017 <= y <= 2020}


def _safe(da: xr.DataArray) -> xr.DataArray:
    return xr.where(np.isfinite(da), da, np.nan)


def _open_cube_da(cid: str) -> Tuple[xr.DataArray, Optional[int]]:
    """Open interpolated feature cube and return a clean (feature,time) DataArray."""
    if FEATURE_SOURCE not in FEATURE_SOURCE_CONFIG:
        raise ValueError(f"Unknown FEATURE_SOURCE: {FEATURE_SOURCE}")
    cfg = FEATURE_SOURCE_CONFIG[FEATURE_SOURCE]
    z = IN_DIR / cfg["path_template"].format(cid=cid)
    if not z.exists():
        raise FileNotFoundError(z)
    ds = xr.open_zarr(z, consolidated=True)
    mean_var_name = cfg["mean_var_name"]
    std_var_name = cfg["std_var_name"]
    if mean_var_name not in ds:
        raise KeyError(f"{mean_var_name} not in {z}")
    mean_da = _safe(ds[mean_var_name]).sortby("time")

    if INCLUDE_STD_FEATURES:
        if std_var_name not in ds:
            raise KeyError(f"{std_var_name} not in {z}")
        std_da = _safe(ds[std_var_name]).sortby("time")
        mean_labels = [f"mean_{int(v)}" for v in mean_da["feature"].values]
        std_labels = [f"std_{int(v)}" for v in std_da["feature"].values]
        mean_da = mean_da.assign_coords(feature=mean_labels)
        std_da = std_da.assign_coords(feature=std_labels)
        da = xr.concat([mean_da, std_da], dim="feature")
    else:
        da = mean_da

    expected_dims = {"feature", "time"}
    if set(da.dims) != expected_dims:
        raise ValueError(f"Expected dims {expected_dims} after preprocessing, got {da.dims} for {z}")

    ridx = mean_da.attrs.get("radiation_feature_index", None)
    return da, int(ridx) if ridx is not None else None


def _parse_date_col(df: pd.DataFrame) -> pd.Series:
    cols = [c.strip().lstrip("\ufeff") for c in df.columns]
    candidates = [c for c in cols if c.upper().startswith("TIMESTAMP")]
    if not candidates:
        raise ValueError("No TIMESTAMP column")
    col = candidates[0]
    s = df[col].astype(str)

    m8 = s.str.match(r"^\d{8}$")
    m12 = s.str.match(r"^\d{12}$")
    if m8.any():
        s.loc[m8] = pd.to_datetime(s[m8], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    if m12.any():
        s.loc[m12] = pd.to_datetime(s[m12], format="%Y%m%d%H%M", errors="coerce").dt.strftime("%Y-%m-%d")

    dt = pd.to_datetime(s, errors="coerce", utc=False)
    return dt.dt.normalize()


def _choose_gpp_column(cols: List[str]) -> Optional[str]:
    pref = [c for c in cols if c.upper() == "GPP_NT_VUT_REF"]
    if pref:
        return pref[0]
    alt = [c for c in cols if re.match(r"(?i)^GPP($|[_])", c)]
    return alt[0] if alt else None


def _load_fluxnet_daily_gpp(site: str) -> pd.Series:
    """
    Load daily GPP (umol CO2 m-2 s-1) for a given site.
    Includes only days with QC >= QC_THRESH.
    """
    files: List[Path] = []
    ww_dir = ROOT_DIR / "FLUXNET2020-ICOS-WarmWinter"
    if ww_dir.exists():
        files += list(ww_dir.glob(f"FLX_*{site}*_FLUXNET2015_FULLSET_DD_*_beta-3.csv"))
    if not files:
        raise FileNotFoundError(f"No FLUXNET WarmWinter DD files for site {site}")

    parts: List[pd.Series] = []

    for f in files:
        df = pd.read_csv(f, low_memory=False, encoding="utf-8-sig")
        dt = _parse_date_col(df)

        gcol = _choose_gpp_column(df.columns.tolist())
        if not gcol or gcol not in df.columns:
            continue

        qc_candidates = [c for c in df.columns if "QC" in c.upper()]
        if not qc_candidates:
            continue
        qc_col = qc_candidates[0]

        gpp = pd.Series(df[gcol].astype(float).values, index=dt)
        qc = pd.Series(df[qc_col].astype(float).values, index=dt)

        if qc.max(skipna=True) <= 1.1:
            qc = qc * 100.0

        valid_mask = qc >= QC_THRESH
        valid_days = gpp[valid_mask & gpp.notna() & np.isfinite(gpp)]

        if not valid_days.empty:
            parts.append(valid_days)

    if not parts:
        raise ValueError(f"No valid GPP data (QC >= {QC_THRESH}%) for {site}")

    df_stack = pd.concat(parts, axis=1)
    gpp = df_stack.bfill(axis=1).iloc[:, 0]
    gpp = gpp.sort_index()
    gpp.name = "GPP"

    return gpp


def _standardize_radiation(da_ft: xr.DataArray, ridx: Optional[int]) -> xr.DataArray:
    if ridx is None:
        return da_ft
    da = da_ft.copy()
    rad = da.isel(feature=ridx)
    da.loc[dict(feature=ridx)] = (rad - RAD_MEAN) / (RAD_STD or 1.0)
    return da


def _apply_radiation_mode(da_ft: xr.DataArray, ridx: Optional[int]) -> xr.DataArray:
    mode = RADIATION_MODE.lower()
    if mode not in {"include", "exclude"}:
        raise ValueError(f"Invalid RADIATION_MODE: {RADIATION_MODE}")

    if ridx is None:
        return da_ft

    if mode == "include":
        return _standardize_radiation(da_ft, ridx)

    try:
        return da_ft.drop_isel(feature=[ridx])
    except Exception:
        sel = np.ones(da_ft.sizes["feature"], dtype=bool)
        sel[ridx] = False
        return da_ft.isel(feature=np.where(sel)[0])


def _trim_to_years(da_ft: xr.DataArray, years: Set[int]) -> xr.DataArray:
    if not years:
        return da_ft.isel(time=slice(0, 0))
    tyears = pd.to_datetime(da_ft.time.values).year
    mask = np.isin(tyears, sorted(years))
    if not mask.any():
        return da_ft.isel(time=slice(0, 0))
    return da_ft.isel(time=np.where(mask)[0])


def _make_windows(
    da_ft: xr.DataArray,
    gpp: pd.Series,
    cid: str,
    site: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build WINDOW-day windows with STRIDE and standardized GPP target on last day."""
    times = pd.to_datetime(da_ft["time"].values).normalize()
    f_count = da_ft.sizes["feature"]

    xs, ys, meta = [], [], []
    for end_idx in range(WINDOW - 1, len(times), STRIDE):
        start_idx = end_idx - (WINDOW - 1)
        win_times = times[start_idx:end_idx + 1]
        end_day = win_times[-1]

        tgt = gpp.get(end_day, np.nan)
        if pd.isna(tgt):
            continue

        tgt_std = (tgt - GPP_MEAN) / (GPP_STD or 1.0)

        win = da_ft.isel(time=slice(start_idx, end_idx + 1))
        arr = np.asarray(win.values, np.float32)
        if not np.isfinite(arr).all():
            continue

        xs.append(arr)
        ys.append(float(tgt_std))
        meta.append({
            "cube_id": cid,
            "site": site,
            "end_date": str(end_day.date()),
        })

    if not xs:
        return (
            np.empty((0, f_count, WINDOW), np.float32),
            np.empty((0,), np.float32),
            pd.DataFrame(columns=["cube_id", "site", "end_date"]),
        )

    x = np.stack(xs, axis=0)
    y = np.array(ys, np.float32)
    return x, y, pd.DataFrame(meta)


def _build_site_lookup(cube_ids: List[str]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for cid in cube_ids:
        site = sites_dict.get(cid, [None])[0]
        if site is not None:
            lookup[cid] = site
    return lookup


def _build_split_definitions(cube_ids: List[str]) -> List[Dict[str, object]]:
    site_lookup = _build_site_lookup(cube_ids)

    available_sites = [site_lookup[cid] for cid in cube_ids if cid in site_lookup]
    holdout_sites = list(LOSO_VAL_SITES) if LOSO_VAL_SITES else available_sites
    folds: List[Dict[str, object]] = []

    for holdout_site in holdout_sites:
        if holdout_site not in available_sites:
            print(f"⚠️  LOSO hold-out site {holdout_site} is not covered by CUBE_IDS - skip")
            continue
        train_sites = {s for s in available_sites if s != holdout_site}
        if not train_sites:
            print(f"⚠️  LOSO hold-out site {holdout_site} leaves no training sites - skip")
            continue
        folds.append({
            "name": f"loso_{holdout_site}",
            "train_sites": train_sites,
            "val_sites": {holdout_site},
            "dataset_tag_train": f"sitesTrain_excl_{holdout_site}",
            "dataset_tag_val": f"sitesVal_{holdout_site}",
            "summary": f"train sites {sorted(train_sites)}, val site [{holdout_site}]",
        })

    return folds


def _concat_or_empty(
    x_list: List[np.ndarray],
    y_list: List[np.ndarray],
    meta_list: List[pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if not x_list:
        return (
            np.empty((0, 0, WINDOW), np.float32),
            np.empty((0,), np.float32),
            pd.DataFrame(columns=["cube_id", "site", "end_date"]),
        )
    return (
        np.concatenate(x_list, axis=0),
        np.concatenate(y_list, axis=0),
        pd.concat(meta_list, ignore_index=True),
    )


def _save_split_outputs(
    split_name: str,
    x_tr_list: List[np.ndarray],
    y_tr_list: List[np.ndarray],
    meta_tr_list: List[pd.DataFrame],
    x_va_list: List[np.ndarray],
    y_va_list: List[np.ndarray],
    meta_va_list: List[pd.DataFrame],
    dataset_tag_train: str,
    dataset_tag_val: str,
) -> None:
    rad_tag = "rad" if RADIATION_MODE.lower() == "include" else "noRad"
    source_tag = FEATURE_SOURCE_CONFIG[FEATURE_SOURCE]["output_tag"]
    feature_tag = "meanstd" if INCLUDE_STD_FEATURES else "mean"

    x_tr, y_tr, meta_tr = _concat_or_empty(x_tr_list, y_tr_list, meta_tr_list)
    x_va, y_va, meta_va = _concat_or_empty(x_va_list, y_va_list, meta_va_list)

    if len(y_tr):
        npz_tr = OUT_DIR / (
            f"gpp_{WINDOW}day_samples_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}_"
            f"{dataset_tag_train}_{source_tag}_{feature_tag}_{rad_tag}_gppstd_train.npz"
        )
        csv_tr = OUT_DIR / (
            f"gpp_{WINDOW}day_samples_meta_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}_"
            f"{dataset_tag_train}_{source_tag}_{feature_tag}_{rad_tag}_train.csv"
        )
        np.savez_compressed(npz_tr, X=x_tr, y=y_tr)
        meta_tr.to_csv(csv_tr, index=False)
        print(f"\n✅ Saved TRAIN for {split_name}:")
        print(f"   → {npz_tr}")
        print(f"   → {csv_tr}")
        print(f"   Shapes: X={x_tr.shape}, y={y_tr.shape}")
    else:
        print(f"\n⚠️  No TRAIN samples produced for {split_name}.")

    if len(y_va):
        npz_va = OUT_DIR / (
            f"gpp_{WINDOW}day_samples_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}_"
            f"{dataset_tag_val}_{source_tag}_{feature_tag}_{rad_tag}_gppstd_val.npz"
        )
        csv_va = OUT_DIR / (
            f"gpp_{WINDOW}day_samples_meta_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}_"
            f"{dataset_tag_val}_{source_tag}_{feature_tag}_{rad_tag}_val.csv"
        )
        np.savez_compressed(npz_va, X=x_va, y=y_va)
        meta_va.to_csv(csv_va, index=False)
        print(f"\n✅ Saved VAL for {split_name}:")
        print(f"   → {npz_va}")
        print(f"   → {csv_va}")
        print(f"   Shapes: X={x_va.shape}, y={y_va.shape}")
    else:
        print(f"\n⚠️  No VAL samples produced for {split_name}.")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split_defs = _build_split_definitions(CUBE_IDS)
    if not split_defs:
        raise RuntimeError("No valid split definitions were created.")

    print("Using split method: site_loso")
    for split_def in split_defs:
        print(f"  - {split_def['name']}: {split_def['summary']}")

    for split_def in split_defs:
        x_tr_list, y_tr_list, meta_tr_list = [], [], []
        x_va_list, y_va_list, meta_va_list = [], [], []

        train_sites = split_def["train_sites"]
        val_sites = split_def["val_sites"]

        print(f"\n=== Building split {split_def['name']} ===")

        for cid in CUBE_IDS:
            site = sites_dict.get(cid, [None])[0]
            if site is None:
                print(f"⚠️  Missing site mapping for cube {cid}")
                continue

            flux_years = detect_flux_years_for_site(site, ROOT_DIR)
            active_years = sorted(flux_years)

            if not active_years:
                print(f"→ {cid} ({site}): no relevant years - skip")
                continue

            try:
                da_ft, ridx = _open_cube_da(cid)
            except Exception as e:
                print(f"⚠️  Skip {cid}: {e}")
                continue

            da_ft = _trim_to_years(da_ft, set(active_years))
            if da_ft.sizes.get("time", 0) < WINDOW:
                print(f"→ {cid} ({site}): too few days after trim - skip")
                continue

            da_ft = _apply_radiation_mode(da_ft, ridx)

            try:
                gpp = _load_fluxnet_daily_gpp(site)
            except Exception as e:
                print(f"⚠️  GPP load failed for {site}: {e}")
                continue
            gpp = gpp[gpp.index.year.isin(active_years)]

            x, y, meta = _make_windows(da_ft, gpp, cid, site)
            if len(y) == 0:
                print(f"→ {cid} ({site}): 0 samples after windowing - skip")
                continue

            end_sites = meta["site"].values
            tr_mask = np.isin(end_sites, list(train_sites))
            va_mask = np.isin(end_sites, list(val_sites))
            train_desc = f"sites {sorted(train_sites)}"
            val_desc = f"sites {sorted(val_sites)}"

            n_tr = int(tr_mask.sum())
            n_va = int(va_mask.sum())

            if n_tr:
                x_tr_list.append(x[tr_mask])
                y_tr_list.append(y[tr_mask])
                meta_tr_list.append(meta.loc[tr_mask].reset_index(drop=True))
            if n_va:
                x_va_list.append(x[va_mask])
                y_va_list.append(y[va_mask])
                meta_va_list.append(meta.loc[va_mask].reset_index(drop=True))

            print(
                f"→ {cid} ({site}): total={len(y)}, train={n_tr} ({train_desc}), "
                f"val={n_va} ({val_desc}), stride={STRIDE}, overlap={OVERLAP}, "
                f"features={da_ft.sizes['feature']} (mode={RADIATION_MODE})"
            )

        _save_split_outputs(
            split_name=split_def["name"],
            x_tr_list=x_tr_list,
            y_tr_list=y_tr_list,
            meta_tr_list=meta_tr_list,
            x_va_list=x_va_list,
            y_va_list=y_va_list,
            meta_va_list=meta_va_list,
            dataset_tag_train=split_def["dataset_tag_train"],
            dataset_tag_val=split_def["dataset_tag_val"],
        )


if __name__ == "__main__":
    main()

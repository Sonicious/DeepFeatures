#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xarray as xr
from matplotlib.lines import Line2D

from model import GPPTemporalTransformer
from sites import sites_dict

try:
    from .config import (
        CUBE_IDS,
        INCLUDE_STD_FEATURES,
        IN_DIR,
        LOSO_VAL_SITES,
        PLOT_CHECKPOINT_HINT,
        SPLIT_METHOD,
        TRAIN_YEARS,
        VAL_YEARS,
    )
except ImportError:
    from config import (
        CUBE_IDS,
        INCLUDE_STD_FEATURES,
        IN_DIR,
        LOSO_VAL_SITES,
        PLOT_CHECKPOINT_HINT,
        SPLIT_METHOD,
        TRAIN_YEARS,
        VAL_YEARS,
    )

ROOT_DIR = Path("/net/data/Fluxnet/")
IN_DIR = Path(IN_DIR)
OUT_DIR = Path("./gpp_compare_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SOURCE = "linear"
FEATURE_SOURCE_CONFIG = {
    "linear": {
        "path_template": "{cid}_linear.zarr",
        "mean_var_name": "feature_mean_linear",
        "std_var_name": "feature_std_linear",
    },
}

WINDOW = 90
OVERLAP = 80
assert 0 <= OVERLAP < WINDOW, "OVERLAP must be in [0, WINDOW)"
STRIDE = WINDOW - OVERLAP

RAD_MEAN = 28.8545
RAD_STD = 6.8393
GPP_MEAN = 4.042
GPP_STD = 4.386
QC_THRESH = None

FEATURE_TAG = "meanstd" if INCLUDE_STD_FEATURES else "mean"
FEATURE_SOURCE = "linear"
RADIATION_MODE = "noRad"


def _dataset_variant_tag() -> str:
    return str(PLOT_CHECKPOINT_HINT).strip().lower()


def _normalize_split_method() -> str:
    method = SPLIT_METHOD.strip().lower()
    aliases = {
        "year": "year",
        "years": "year",
        "site": "site_loso",
        "loso": "site_loso",
        "site_loso": "site_loso",
        "leave_one_site_out": "site_loso",
    }
    if method not in aliases:
        raise ValueError(f"Unknown SPLIT_METHOD: {SPLIT_METHOD}")
    return aliases[method]


def _available_sites() -> List[str]:
    out = []
    for cid in CUBE_IDS:
        site = sites_dict.get(cid, [None])[0]
        if site is not None:
            out.append(site)
    return out


def _build_fold_defs() -> List[Dict[str, object]]:
    method = _normalize_split_method()
    variant_tag = _dataset_variant_tag()
    if method == "year":
        train_tag = f"years{min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}"
        val_tag = f"years{'-'.join(str(y) for y in sorted(VAL_YEARS))}"
        run_tag = f"{FEATURE_SOURCE}_{FEATURE_TAG}_{RADIATION_MODE}_{variant_tag}_{train_tag}_to_{val_tag}"
        return [{
            "name": "year_split",
            "mode": "year",
            "run_tag": run_tag,
            "train_sites": set(_available_sites()),
            "val_sites": set(_available_sites()),
            "train_years": set(TRAIN_YEARS),
            "val_years": set(VAL_YEARS),
        }]

    sites = _available_sites()
    holdout_sites = list(LOSO_VAL_SITES) if LOSO_VAL_SITES else sites
    folds = []
    for holdout_site in holdout_sites:
        if holdout_site not in sites:
            print(f"⚠️  LOSO hold-out site {holdout_site} is not covered by CUBE_IDS - skip")
            continue
        folds.append({
            "name": f"loso_{holdout_site}",
            "mode": "site_loso",
            "run_tag": f"{FEATURE_SOURCE}_{FEATURE_TAG}_{RADIATION_MODE}_{variant_tag}_loso_{holdout_site}",
            "holdout_site": holdout_site,
            "train_sites": {s for s in sites if s != holdout_site},
            "val_sites": {holdout_site},
            "train_years": None,
            "val_years": None,
        })
    return folds


def _latest_checkpoint_from_results(run_tag: str) -> Path:
    results_csv = Path(f"grid_results_{run_tag}.csv")
    if not results_csv.exists():
        raise FileNotFoundError(f"Results file not found: {results_csv}")

    df = pd.read_csv(results_csv)
    if df.empty:
        raise ValueError(f"No rows in results file: {results_csv}")

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError(f"No successful runs found in {results_csv}")

    ok = ok[ok["best_ckpt"].notna()].copy()
    ok["best_ckpt"] = ok["best_ckpt"].astype(str).str.strip()
    ok = ok[ok["best_ckpt"] != ""]
    if ok.empty:
        raise ValueError(f"No checkpoint paths found in successful rows of {results_csv}")

    hint = str(PLOT_CHECKPOINT_HINT).strip() if PLOT_CHECKPOINT_HINT is not None else ""
    if hint:
        mask = (
            ok["best_ckpt"].str.contains(hint, case=False, na=False)
            | ok["run_name"].astype(str).str.contains(hint, case=False, na=False)
        )
        hinted = ok[mask]
        if not hinted.empty:
            ok = hinted
        else:
            print(f"⚠️  No checkpoint rows matched PLOT_CHECKPOINT_HINT='{hint}' in {results_csv}; using latest successful row.")

    latest_row = ok.iloc[-1]
    ckpt = Path(str(latest_row["best_ckpt"]))
    if not ckpt.exists():
        raise FileNotFoundError(f"Latest checkpoint listed in {results_csv} does not exist: {ckpt}")
    return ckpt


def _site_to_fold(fold_defs: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    mapping: Dict[str, Dict[str, object]] = {}
    for fold in fold_defs:
        for site in fold["val_sites"]:
            mapping[site] = fold
    return mapping


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
    ridx = mean_da.attrs.get("radiation_feature_index", None)
    return da, int(ridx) if ridx is not None else None


def _parse_date_col(df: pd.DataFrame) -> pd.Series:
    candidates = [c for c in df.columns if c.upper().startswith("TIMESTAMP")]
    if not candidates:
        raise ValueError("No TIMESTAMP column")
    col = candidates[0]
    s = df[col].astype(str)
    m8 = s.str.match(r"^\d{8}$")
    if m8.any():
        s.loc[m8] = pd.to_datetime(s[m8], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    m12 = s.str.match(r"^\d{12}$")
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


def _load_fluxnet_daily_gpp(site: str, qc_thresh: Optional[float] = QC_THRESH) -> pd.Series:
    files: List[Path] = []
    ww_dir = ROOT_DIR / "FLUXNET2020-ICOS-WarmWinter"
    if ww_dir.exists():
        files += list(ww_dir.glob(f"FLX_*{site}*_FLUXNET2015_FULLSET_DD_*_beta-3.csv"))
    if not files:
        raise FileNotFoundError(f"No FLUXNET DD files for site {site}")

    parts: List[pd.Series] = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        dt = _parse_date_col(df)
        gcol = _choose_gpp_column(df.columns.tolist())
        if not gcol or gcol not in df.columns:
            continue
        qc_candidates = [c for c in df.columns if "QC" in c.upper()]
        if not qc_candidates:
            continue
        qc_col = qc_candidates[0]

        gpp = pd.Series(pd.to_numeric(df[gcol], errors="coerce").values, index=dt)
        qc = pd.Series(pd.to_numeric(df[qc_col], errors="coerce").values, index=dt)

        if qc.max(skipna=True) <= 1.1:
            qc = qc * 100.0

        if qc_thresh is None:
            valid_days = gpp[gpp.notna() & np.isfinite(gpp)]
        else:
            valid_mask = qc >= qc_thresh
            valid_days = gpp[valid_mask & gpp.notna() & np.isfinite(gpp)]
        if not valid_days.empty:
            parts.append(valid_days)

    if not parts:
        if qc_thresh is None:
            raise ValueError(f"No valid GPP data for {site}")
        raise ValueError(f"No valid GPP data (QC >= {qc_thresh}%) for {site}")

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


def _apply_radiation_mode(da_ft: xr.DataArray, ridx: Optional[int], mode: str) -> xr.DataArray:
    mode = mode.lower()
    if mode not in {"include", "exclude"}:
        raise ValueError(f"Invalid radiation mode: {mode}")
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


def _trim_to_flux_years(da_ft: xr.DataArray, years: List[int]) -> xr.DataArray:
    if not years:
        return da_ft.isel(time=slice(0, 0))
    mask = np.isin(pd.to_datetime(da_ft.time.values).year, years)
    if not mask.any():
        return da_ft.isel(time=slice(0, 0))
    return da_ft.isel(time=np.where(mask)[0])


def _make_windows(
    da_ft: xr.DataArray,
    gpp: pd.Series,
    cid: str,
    site: str,
) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    times = pd.to_datetime(da_ft["time"].values).normalize()
    f_count = da_ft.sizes["feature"]
    xs, ys, ts = [], [], []
    for end_idx in range(WINDOW - 1, len(times), STRIDE):
        start_idx = end_idx - (WINDOW - 1)
        end_day = times[end_idx]
        tgt = gpp.get(end_day, np.nan)
        if pd.isna(tgt):
            continue
        tgt_std = (tgt - GPP_MEAN) / (GPP_STD or 1.0)
        win = da_ft.isel(time=slice(start_idx, end_idx + 1))
        arr = np.asarray(win.values, np.float32)
        if not np.isfinite(arr).all():
            continue
        xs.append(arr.T)
        ys.append(float(tgt_std))
        ts.append(pd.Timestamp(end_day))
    if not xs:
        return (
            np.empty((0, WINDOW, f_count), np.float32),
            np.empty((0,), np.float32),
            [],
        )
    x = np.stack(xs, axis=0)
    y = np.array(ys, np.float32)
    return x, y, ts


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    r = rmse(pred, true)
    denom = float(np.max(true) - np.min(true)) if len(true) > 0 else np.nan
    return r / denom if denom > 0 else np.nan


def overall_nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    mask = np.isfinite(pred) & np.isfinite(true)
    if not mask.any():
        return np.nan
    return nrmse(pred[mask], true[mask])


def _infer_model_feature_count(model: nn.Module) -> int:
    nf = getattr(getattr(model, "hparams", object()), "num_features", None)
    if isinstance(nf, (int, np.integer)):
        return int(nf)
    lin = getattr(model, "input_proj", None)
    if isinstance(lin, nn.Linear):
        return int(lin.in_features)
    raise RuntimeError("Cannot infer model's expected num_features.")


def load_model(ckpt_path: Path, device: torch.device) -> Tuple[GPPTemporalTransformer, int, str]:
    model = GPPTemporalTransformer.load_from_checkpoint(str(ckpt_path), map_location=device)
    model.eval().to(device)
    f_model = _infer_model_feature_count(model)
    ckpt_label = ckpt_path.name
    print(f"[model] {ckpt_label} expects num_features = {f_model}")
    time_first = getattr(model.hparams, "time_first", True)
    assert time_first, "Model was trained with time_first=True; expected (B,T,F) inputs."
    return model, f_model, ckpt_label


def align_X_to_model(X: np.ndarray, f_model: int, ridx: Optional[int]) -> np.ndarray:
    f_data = X.shape[-1]
    if f_data == f_model:
        return X
    if f_data == f_model + 1:
        drop_idx = ridx if ridx is not None else (f_data - 1)
        keep = [i for i in range(f_data) if i != drop_idx]
        return X[..., keep]
    if f_data + 1 == f_model:
        insert_idx = ridx if ridx is not None else f_data
        zeros = np.zeros((*X.shape[:-1], 1), dtype=X.dtype)
        return np.concatenate([X[..., :insert_idx], zeros, X[..., insert_idx:]], axis=-1)
    raise ValueError(f"Cannot align features: data has {f_data}, model expects {f_model}.")


def plot_per_cube(
    dates: List[pd.Timestamp],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cid: str,
    site: str,
    out_dir: Path,
    label: str,
):
    nrmse_val = nrmse(y_pred, y_true)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(dates, y_true, color="lightgray", s=15, label="True GPP (daily)", zorder=2)
    ax.plot(dates, y_pred, color="#1f77b4", linewidth=1.4, label=label, zorder=3)
    ax.text(
        0.02,
        0.96,
        f"{label}: NRMSE={nrmse_val:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"),
    )
    ax.set_title(f"{site}", fontsize=12, weight="semibold")
    ax.set_xlabel("Date")
    ax.set_ylabel("GPP")
    ax.grid(True, linewidth=0.5, alpha=0.5)

    out_png = out_dir / f"gpp_compare_{cid}_{site.replace(' ', '_')}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"   saved: {out_png}")

    return {
        "cid": cid,
        "site": site,
        "dates": dates,
        "y_true": y_true,
        "y": y_pred,
        "nrmse": nrmse_val,
    }


def plot_combined(packs: List[Dict], out_dir: Path, label: str):
    n = len(packs)
    cols, rows = 1, n
    fig, axes = plt.subplots(rows, cols, figsize=(7.5, 2.4 * rows), squeeze=False)
    axes = axes.flatten()

    for i, p in enumerate(packs):
        ax = axes[i]
        ax.scatter(p["dates"], p["y_true"], color="lightgray", s=10, zorder=2)
        ax.plot(p["dates"], p["y"], color="#1f77b4", linewidth=1.2, zorder=3)
        ax.set_title(f"{p['site']}", fontsize=11, weight="semibold")
        ax.set_ylabel("GPP", fontsize=11)
        ax.set_xlabel("")
        ax.grid(True, linewidth=0.5, alpha=0.5)
        ax.label_outer()
        if i < n - 1:
            ax.set_xticklabels([])

    legend_elements = [
        Line2D([0], [0], color="lightgray", lw=0, marker="o", markersize=6, label="True GPP (daily)"),
        Line2D([0], [0], color="#1f77b4", lw=1.6, label=label),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
        fontsize=11,
        ncol=2,
        columnspacing=1.2,
        handlelength=2.2,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_png = out_dir / "gpp_compare_combined_bottomlegend.png"
    out_pdf = out_dir / "gpp_compare_combined_bottomlegend.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"   combined saved: {out_png}")
    print(f"   combined saved: {out_pdf}")


def _predict(model: nn.Module, device: torch.device, x_in: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        xt = torch.tensor(x_in, dtype=torch.float32, device=device)
        y_std_pred = model(xt).detach().cpu().numpy()
    return y_std_pred * GPP_STD + GPP_MEAN


def _compute_split_masks(dates: List[pd.Timestamp], site: str, fold_def: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = fold_def["mode"]
    years = np.array([d.year for d in dates]) if dates else np.array([], dtype=int)
    sites = np.array([site] * len(dates), dtype=object)

    if mode == "year":
        train_mask = np.isin(years, list(fold_def["train_years"]))
        val_mask = np.isin(years, list(fold_def["val_years"]))
    else:
        train_mask = np.isin(sites, list(fold_def["train_sites"]))
        val_mask = np.isin(sites, list(fold_def["val_sites"]))
    other_mask = ~(train_mask | val_mask)
    return train_mask, val_mask, other_mask


def _append_metric_row(rows: List[Dict[str, object]], split_name: str, agg: str, pred: np.ndarray, true: np.ndarray) -> None:
    if len(true) == 0:
        rows.append({"split": split_name, "aggregation": agg, "rmse": np.nan, "nrmse": np.nan})
        return
    rows.append({
        "split": split_name,
        "aggregation": agg,
        "rmse": rmse(pred, true),
        "nrmse": overall_nrmse(pred, true),
    })


def main():
    import matplotlib

    matplotlib.use("Agg")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    fold_defs = _build_fold_defs()
    if not fold_defs:
        raise RuntimeError("No valid folds found for plotting.")

    fold_models: Dict[str, Dict[str, object]] = {}
    for fold_def in fold_defs:
        ckpt_path = _latest_checkpoint_from_results(fold_def["run_tag"])
        model, f_model, ckpt_label = load_model(ckpt_path, device)
        fold_models[fold_def["name"]] = {
            "fold_def": fold_def,
            "model": model,
            "f_model": f_model,
            "ckpt_label": ckpt_label,
        }
        print(f"Using checkpoint for {fold_def['name']}: {ckpt_path}")

    site_to_fold = _site_to_fold(fold_defs)
    label = "prediction"
    if _normalize_split_method() == "site_loso":
        label = "LOSO prediction"

    packs: List[Dict] = []
    pooled = {
        "train_true": [],
        "train_pred": [],
        "val_true": [],
        "val_pred": [],
    }
    site_metric_rows: List[Dict[str, object]] = []

    for cid in CUBE_IDS:
        site = sites_dict.get(cid, [None])[0]
        if site is None:
            print(f"⚠️ Missing site mapping for cube {cid} - skip")
            continue
        if site not in site_to_fold:
            print(f"⚠️ No fold configured for site {site} - skip")
            continue

        fold_def = site_to_fold[site]
        fold_state = fold_models[fold_def["name"]]
        model = fold_state["model"]
        f_model = fold_state["f_model"]

        flux_years = sorted(detect_flux_years_for_site(site, ROOT_DIR))
        if not flux_years:
            print(f"→ {cid} ({site}): no flux years - skip")
            continue

        try:
            da_ft, ridx = _open_cube_da(cid)
        except Exception as e:
            print(f"⚠️ Skip {cid}: {e}")
            continue

        da_ft = _trim_to_flux_years(da_ft, flux_years)
        if da_ft.sizes.get("time", 0) < WINDOW:
            print(f"→ {cid} ({site}): too few days after trim - skip")
            continue

        try:
            gpp_series = _load_fluxnet_daily_gpp(site, qc_thresh=QC_THRESH)
        except Exception as e:
            print(f"⚠️  GPP load failed for {site}: {e}")
            continue
        gpp_series = gpp_series[gpp_series.index.year.isin(flux_years)]

        x_base, _, ts = _make_windows(da_ft, gpp_series, cid, site)
        if x_base.shape[0] == 0:
            print(f"→ {cid} ({site}): no valid windows - skip")
            continue

        x_in = align_X_to_model(x_base, f_model=f_model, ridx=ridx)
        y_pred = _predict(model, device, x_in)

        y_true = np.array([gpp_series.get(pd.Timestamp(t).normalize(), np.nan) for t in ts], dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        dates = [t for t, m in zip(ts, mask) if m]
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            print(f"→ {cid} ({site}): no comparable points after masking - skip")
            continue

        train_mask, val_mask, _ = _compute_split_masks(dates, site, fold_def)

        if train_mask.any():
            pooled["train_true"].append(y_true[train_mask])
            pooled["train_pred"].append(y_pred[train_mask])
            site_metric_rows.append({
                "site": site,
                "cube_id": cid,
                "fold": fold_def["name"],
                "split": "train",
                "rmse": rmse(y_pred[train_mask], y_true[train_mask]),
                "nrmse": overall_nrmse(y_pred[train_mask], y_true[train_mask]),
                "n_samples": int(train_mask.sum()),
            })

        if val_mask.any():
            pooled["val_true"].append(y_true[val_mask])
            pooled["val_pred"].append(y_pred[val_mask])
            site_metric_rows.append({
                "site": site,
                "cube_id": cid,
                "fold": fold_def["name"],
                "split": "val",
                "rmse": rmse(y_pred[val_mask], y_true[val_mask]),
                "nrmse": overall_nrmse(y_pred[val_mask], y_true[val_mask]),
                "n_samples": int(val_mask.sum()),
            })

        if train_mask.any():
            print(f"   ↳ Train NRMSE: {overall_nrmse(y_pred[train_mask], y_true[train_mask]):.3f}")
        if val_mask.any():
            print(f"   ↳ Val NRMSE: {overall_nrmse(y_pred[val_mask], y_true[val_mask]):.3f}")

        pack = plot_per_cube(
            dates=dates,
            y_true=y_true,
            y_pred=y_pred,
            cid=cid,
            site=site,
            out_dir=OUT_DIR,
            label=label,
        )
        print(f"[{cid} {site}] NRMSE: {pack['nrmse']:.3f} using {fold_def['name']}")

        split_col = np.where(val_mask, "val", np.where(train_mask, "train", "other"))
        df_out = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "year": [d.year for d in dates],
            "site": site,
            "fold": fold_def["name"],
            "split": split_col,
            "gpp_true": y_true,
            "gpp_pred": y_pred,
        })
        df_out = df_out[df_out["split"].isin(["train", "val"])]
        out_csv = OUT_DIR / f"gpp_compare_cube_{cid}.csv"
        df_out.to_csv(out_csv, index=False)
        print(f"   saved: {out_csv}")

        packs.append(pack)

    if packs:
        plot_combined(packs, OUT_DIR, label=label)
        overall_true = np.concatenate([p["y_true"] for p in packs], axis=0)
        overall_pred = np.concatenate([p["y"] for p in packs], axis=0)
        overall = overall_nrmse(overall_pred, overall_true)
        print(f"Overall NRMSE: {overall:.4f}")
    else:
        print("No plots to combine.")

    def _concat(lst: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(lst, axis=0) if lst else np.empty((0,), dtype=float)

    tr_true = _concat(pooled["train_true"])
    tr_pred = _concat(pooled["train_pred"])
    va_true = _concat(pooled["val_true"])
    va_pred = _concat(pooled["val_pred"])

    site_metrics_df = pd.DataFrame(site_metric_rows)
    site_metrics_path = OUT_DIR / "site_split_metrics.csv"
    site_metrics_df.to_csv(site_metrics_path, index=False)
    summary_rows: List[Dict[str, object]] = []

    if not site_metrics_df.empty:
        mean_by_split = (
            site_metrics_df.groupby("split", as_index=False)[["rmse", "nrmse"]]
            .mean(numeric_only=True)
        )
        for row in mean_by_split.to_dict("records"):
            split_name = str(row["split"])
            print(f"MEAN {split_name.capitalize()} NRMSE across sites: {row['nrmse']:.4f}")
            print(f"MEAN {split_name.capitalize()} RMSE across sites: {row['rmse']:.4f}")
        for row in mean_by_split.to_dict("records"):
            summary_rows.append({
                "split": row["split"],
                "aggregation": "mean_across_sites",
                "rmse": row["rmse"],
                "nrmse": row["nrmse"],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "global_split_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved site metrics to {site_metrics_path}")
    print(f"Saved summary metrics to {summary_path}")


if __name__ == "__main__":
    main()

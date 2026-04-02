import csv
import itertools
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger

from GPP_loader import make_loaders
from model import GPPTemporalTransformer

try:
    from .config import (
        CUBE_IDS,
        INCLUDE_STD_FEATURES,
        LOSO_VAL_SITES,
        OUT_DIR,
        PLOT_CHECKPOINT_HINT,
        SPLIT_METHOD,
        TRAIN_YEARS,
        VAL_YEARS,
    )
except ImportError:
    from config import (
        CUBE_IDS,
        INCLUDE_STD_FEATURES,
        LOSO_VAL_SITES,
        OUT_DIR,
        PLOT_CHECKPOINT_HINT,
        SPLIT_METHOD,
        TRAIN_YEARS,
        VAL_YEARS,
    )

from sites import sites_dict

SEED = 42
DEVICE_ID = 1
MAX_EPOCHS = 150
MAX_TRIALS = 180


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
torch.set_float32_matmul_precision("high")

BASE = OUT_DIR

WINDOW = 90
OVERLAP = 80
STRIDE = WINDOW - OVERLAP
QC_THRESH = 70.0
FEATURE_SOURCE = "linear"
RADIATION_MODE = "noRad"

FEATURE_SOURCE_TAGS = {
    "linear": "linear",
    "ucm_flux": "ucm_flux",
}
RADIATION_MODE_TAGS = {
    "include": "rad",
    "exclude": "noRad",
    "rad": "rad",
    "noRad": "noRad",
}

FEATURE_TAG = "meanstd" if INCLUDE_STD_FEATURES else "mean"
NUM_FEATURES = 12 if INCLUDE_STD_FEATURES else 6


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

SPACE = {
    "num_features": [NUM_FEATURES],
    "batch_size": [12],
    "d_model": [96],
    "nhead": [16],
    "num_layers": [3],
    "dim_ff": [1024],
    "dropout": [0.05],
    "pool": ["last"],
    "lr": [1e-4],
    "weight_decay": [1e-6],
    "warmup_steps": [200],
    "reduce_pat": [7],
    "reduce_factor": [0.1],
}


def _available_sites() -> List[str]:
    sites = []
    for cid in CUBE_IDS:
        site = sites_dict.get(cid, [None])[0]
        if site is not None:
            sites.append(site)
    return sites


def _build_fold_defs() -> List[Dict[str, str]]:
    variant_tag = _dataset_variant_tag()
    method = _normalize_split_method()
    if method == "year":
        train_tag = f"years{min(TRAIN_YEARS)}-{max(TRAIN_YEARS)}"
        val_tag = f"years{'-'.join(str(y) for y in sorted(VAL_YEARS))}"
        return [{
            "name": "year_split",
            "train_dataset_tag": train_tag,
            "val_dataset_tag": val_tag,
            "run_tag": f"{FEATURE_SOURCE}_{FEATURE_TAG}_{RADIATION_MODE}_{variant_tag}_{train_tag}_to_{val_tag}",
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
            "train_dataset_tag": f"sitesTrain_excl_{holdout_site}",
            "val_dataset_tag": f"sitesVal_{holdout_site}",
            "run_tag": f"{FEATURE_SOURCE}_{FEATURE_TAG}_{RADIATION_MODE}_{variant_tag}_loso_{holdout_site}",
        })
    return folds


def build_dataset_paths(base: str, train_dataset_tag: str, val_dataset_tag: str):
    if FEATURE_SOURCE not in FEATURE_SOURCE_TAGS:
        raise ValueError(f"Unknown FEATURE_SOURCE: {FEATURE_SOURCE}")
    if RADIATION_MODE not in RADIATION_MODE_TAGS:
        raise ValueError(f"Unknown RADIATION_MODE: {RADIATION_MODE}")

    source_tag = FEATURE_SOURCE_TAGS[FEATURE_SOURCE]
    rad_tag = RADIATION_MODE_TAGS[RADIATION_MODE]

    train_npz = (
        f"{base}/gpp_{WINDOW}day_samples_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}_"
        f"{train_dataset_tag}_{source_tag}_{FEATURE_TAG}_{rad_tag}_gppstd_train.npz"
    )
    train_meta = (
        f"{base}/gpp_{WINDOW}day_samples_meta_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}_"
        f"{train_dataset_tag}_{source_tag}_{FEATURE_TAG}_{rad_tag}_train.csv"
    )
    val_npz = (
        f"{base}/gpp_{WINDOW}day_samples_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}_"
        f"{val_dataset_tag}_{source_tag}_{FEATURE_TAG}_{rad_tag}_gppstd_val.npz"
    )
    val_meta = (
        f"{base}/gpp_{WINDOW}day_samples_meta_stride{STRIDE}_overlap{OVERLAP}_qc{QC_THRESH}_"
        f"{val_dataset_tag}_{source_tag}_{FEATURE_TAG}_{rad_tag}_val.csv"
    )
    return train_npz, train_meta, val_npz, val_meta


def build_search_space():
    keys = list(SPACE.keys())
    all_combos = (dict(zip(keys, vals)) for vals in itertools.product(*(SPACE[k] for k in keys)))
    return [c for c in all_combos if c["d_model"] % c["nhead"] == 0]


def run_once(params, fold_def: Dict[str, str], max_epochs: int = MAX_EPOCHS):
    """Train one config for one fold; return metrics dict."""
    set_seed(SEED)

    train_npz, train_meta, val_npz, val_meta = build_dataset_paths(
        BASE,
        train_dataset_tag=fold_def["train_dataset_tag"],
        val_dataset_tag=fold_def["val_dataset_tag"],
    )

    for path in [train_npz, train_meta, val_npz, val_meta]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing dataset file for {fold_def['name']}: {path}")

    loaders = make_loaders(
        train_npz,
        train_meta,
        val_npz,
        val_meta,
        batch_size=params["batch_size"],
        time_first=True,
        num_workers=8,
        pin_memory=True,
    )

    model = GPPTemporalTransformer(
        num_features=params["num_features"],
        seq_len=90,
        d_model=params["d_model"],
        nhead=params["nhead"],
        num_layers=params["num_layers"],
        dim_feedforward=params["dim_ff"],
        dropout=params["dropout"],
        pool=params["pool"],
        learning_rate=params["lr"],
        weight_decay=params["weight_decay"],
        warmup_steps=params["warmup_steps"],
        reduce_patience=params["reduce_pat"],
        reduce_factor=params["reduce_factor"],
    )
    model.criterion = nn.L1Loss()

    run_name = (
        f"{fold_def['name']}_bs{params['batch_size']}_dm{params['d_model']}_h{params['nhead']}"
        f"_L{params['num_layers']}_ff{params['dim_ff']}_do{params['dropout']}"
        f"_{params['pool']}_lr{params['lr']}_wd{params['weight_decay']}"
        f"_wu{params['warmup_steps']}_rp{params['reduce_pat']}_rf{params['reduce_factor']}"
    ).replace(".", "p")

    log_dir = f"grid_logs_2_{fold_def['run_tag']}"
    logger = CSVLogger(save_dir=log_dir, name=run_name, version="")
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
    )
    callbacks = [
        ckpt_cb,
        EarlyStopping(monitor="val_loss", mode="min", patience=16, verbose=True),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[DEVICE_ID],
        precision="16-mixed",
        max_epochs=max_epochs,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=logger,
    )

    t0 = time.time()
    status = "ok"
    best_val = math.inf
    best_path = ""
    try:
        trainer.fit(model, train_dataloaders=loaders["train"], val_dataloaders=loaders["val"])
        if ckpt_cb.best_model_score is not None:
            best_val = float(ckpt_cb.best_model_score.cpu().item())
        best_path = ckpt_cb.best_model_path or ""
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            status = "oom"
        else:
            status = f"runtime_error: {e.__class__.__name__}"
    except Exception as e:
        status = f"error: {e.__class__.__name__}"
    wall = time.time() - t0

    return {
        "fold": fold_def["name"],
        "status": status,
        "best_val_loss": best_val,
        "best_ckpt": best_path,
        "run_name": run_name,
        "wall_time_sec": round(wall, 2),
        **params,
    }


def main():
    set_seed(SEED)

    fold_defs = _build_fold_defs()
    if not fold_defs:
        raise RuntimeError("No valid training folds were created.")

    space = build_search_space()
    random.shuffle(space)
    if MAX_TRIALS > 0:
        space = space[:MAX_TRIALS]

    for fold_def in fold_defs:
        results_path = Path(f"grid_results_{fold_def['run_tag']}.csv")
        write_header = not results_path.exists()
        results_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Training fold {fold_def['name']} ===")
        with results_path.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "fold",
                    "status",
                    "best_val_loss",
                    "best_ckpt",
                    "run_name",
                    "wall_time_sec",
                    "num_features",
                    "batch_size",
                    "d_model",
                    "nhead",
                    "num_layers",
                    "dim_ff",
                    "dropout",
                    "pool",
                    "lr",
                    "weight_decay",
                    "warmup_steps",
                    "reduce_pat",
                    "reduce_factor",
                ],
            )
            if write_header:
                writer.writeheader()

            total = len(space)
            for i, params in enumerate(space, 1):
                print(f"\n=== Fold {fold_def['name']} | Trial {i}/{total} ===")
                print(params)
                metrics = run_once(params, fold_def=fold_def)
                writer.writerow(metrics)
                f.flush()
                print(
                    f"→ status: {metrics['status']}, best val: {metrics['best_val_loss']:.6f}, "
                    f"ckpt: {metrics['best_ckpt']}"
                )

        print(f"Results saved to {results_path.resolve()}")


if __name__ == "__main__":
    main()

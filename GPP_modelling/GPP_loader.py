# gpp_dataloaders.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Sequence, Dict, Any
try:
    from .config import INCLUDE_STD_FEATURES, OUT_DIR
except ImportError:
    from config import INCLUDE_STD_FEATURES, OUT_DIR

class GPPSlidingWindowDataset(Dataset):
    """
    Loads 90-day GPP window dataset from NPZ/CSV files.
    Each sample:
        X: (F, 90) standardized input features
        y: (1,) standardized GPP target
    """
    def __init__(
        self,
        npz_path: str,
        meta_csv: Optional[str] = None,
        time_first: bool = False,     # if True -> (T, F)
        dtype: torch.dtype = torch.float32,
        return_meta: bool = False,
        device: Optional[torch.device] = None,
        feature_slice: Optional[Sequence[int]] = None,  # e.g. [0,1,2,7]
    ):
        data = np.load(npz_path)
        self.X = data["X"]             # (N, F, 90)
        self.y = data["y"]             # (N,)
        self.meta = pd.read_csv(meta_csv) if meta_csv else None
        if feature_slice is not None:
            self.X = self.X[:, feature_slice, :]
        self.time_first = time_first
        self.dtype = dtype
        self.return_meta = return_meta
        self.device = device

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.X[idx], dtype=self.dtype)
        y = torch.tensor(self.y[idx], dtype=self.dtype)
        if self.time_first:
            x = x.transpose(0, 1)  # -> (90, F)
        if self.device is not None:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
        if self.return_meta and self.meta is not None:
            row = self.meta.iloc[int(idx)]
            m = {
                "cube_id": row.get("cube_id"),
                "site": row.get("site"),
                "end_date": row.get("end_date"),
                "index": int(idx),
            }
            return x, y, m
        return x, y


def _default_collate(batch):
    """Keeps metadata as list when present."""
    if len(batch[0]) == 3:
        xs, ys, metas = zip(*batch)
        return torch.stack(xs), torch.stack(ys), list(metas)
    else:
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)


def make_loaders(
    train_npz: str,
    train_meta: str,
    val_npz: Optional[str] = None,
    val_meta: Optional[str] = None,
    batch_size: int = 64,
    time_first: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    return_meta: bool = True,
    device: Optional[str] = None,
    feature_slice: Optional[Sequence[int]] = None,
) -> Dict[str, DataLoader]:
    """Creates PyTorch DataLoaders for train (+optional val)."""
    dev = torch.device(device) if device else None

    train_ds = GPPSlidingWindowDataset(
        train_npz, train_meta,
        time_first=time_first, return_meta=return_meta,
        device=dev, feature_slice=feature_slice,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=_default_collate, drop_last=False,
    )

    loaders = {"train": train_loader}

    if val_npz and val_meta:
        val_ds = GPPSlidingWindowDataset(
            val_npz, val_meta,
            time_first=time_first, return_meta=return_meta,
            device=dev, feature_slice=feature_slice,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            collate_fn=_default_collate, drop_last=False,
        )
        loaders["val"] = val_loader

    return loaders

PATHS = OUT_DIR
WINDOW = 90
OVERLAP = 80
STRIDE = WINDOW - OVERLAP
QC_THRESH = 70.0
TRAIN_YEARS_TAG = "2017-2019"
VAL_YEARS_TAG = "2020"
SOURCE_TAG = "linear"
RADIATION_TAG = "noRad"
FEATURE_TAG = "meanstd" if INCLUDE_STD_FEATURES else "mean"

train_npz = (
    f"{PATHS}/gpp_{WINDOW}day_samples_stride{STRIDE}_overlap{OVERLAP}_"
    f"qc{QC_THRESH}_years{TRAIN_YEARS_TAG}_{SOURCE_TAG}_{FEATURE_TAG}_{RADIATION_TAG}_gppstd_train.npz"
)
train_meta = (
    f"{PATHS}/gpp_{WINDOW}day_samples_meta_stride{STRIDE}_overlap{OVERLAP}_"
    f"qc{QC_THRESH}_years{TRAIN_YEARS_TAG}_{SOURCE_TAG}_{FEATURE_TAG}_{RADIATION_TAG}_train.csv"
)
val_npz = (
    f"{PATHS}/gpp_{WINDOW}day_samples_stride{STRIDE}_overlap{OVERLAP}_"
    f"qc{QC_THRESH}_years{VAL_YEARS_TAG}_{SOURCE_TAG}_{FEATURE_TAG}_{RADIATION_TAG}_gppstd_val.npz"
)
val_meta = (
    f"{PATHS}/gpp_{WINDOW}day_samples_meta_stride{STRIDE}_overlap{OVERLAP}_"
    f"qc{QC_THRESH}_years{VAL_YEARS_TAG}_{SOURCE_TAG}_{FEATURE_TAG}_{RADIATION_TAG}_val.csv"
)

feature_slice = None
if not INCLUDE_STD_FEATURES:
    feature_slice = list(range(6))

loaders = make_loaders(
    train_npz, train_meta,
    val_npz, val_meta,
    batch_size=128,
    time_first=True,   # (B, 90, F) for Transformer models
    num_workers=8,
    feature_slice=feature_slice,
)

#for batch in loaders["train"]:
#    X, y, meta = batch   # X: (B, 90, F), y: (B,)
#    print(X.shape, y.shape, len(meta))
#    break
#

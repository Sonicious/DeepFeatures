#!/usr/bin/env python3

# Shared configuration for the GPP modelling pipeline.

IN_DIR = "/net/data/deepfeatures/feature"
OUT_DIR = IN_DIR

CUBE_IDS = ["003", "004", "005"]

# False -> use interpolated mean features only (6 features)
# True  -> use interpolated mean + std features (12 features)
INCLUDE_STD_FEATURES = True

# Optional manual subset of held-out sites for LOSO.
# [] / None means: create one fold for every site in CUBE_IDS.
LOSO_VAL_SITES = []

# Optional substring filter used by GPP_plot.py when auto-selecting checkpoints
# from grid_results_*.csv. Example: "no_si" or "si".
PLOT_CHECKPOINT_HINT = "si"

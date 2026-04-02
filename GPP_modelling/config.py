#!/usr/bin/env python3

# Shared configuration for the GPP modelling pipeline.

# from grid_results_*.csv. Example: "no_si" or "si".
PLOT_CHECKPOINT_HINT = "si"

if PLOT_CHECKPOINT_HINT == "si":
    IN_DIR = "/net/data/deepfeatures/feature"
else:    IN_DIR = "/net/data/deepfeatures/feature_no_si"
OUT_DIR = IN_DIR

CUBE_IDS = ["003", "004", "005"]

# False -> use interpolated mean features only (6 features)
# True  -> use interpolated mean + std features (12 features)
INCLUDE_STD_FEATURES = True

# Optional manual subset of held-out sites for LOSO.
# [] / None means: create one fold for every site in CUBE_IDS.
LOSO_VAL_SITES = []



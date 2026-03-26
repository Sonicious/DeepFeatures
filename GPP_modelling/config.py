#!/usr/bin/env python3

# Shared configuration for the GPP modelling pipeline.

IN_DIR = "/net/data/deepfeatures/feature"
OUT_DIR = "/net/data/deepfeatures/feature"

CUBE_IDS = ["003", "004"]

# False -> use interpolated mean features only (6 features)
# True  -> use interpolated mean + std features (12 features)
INCLUDE_STD_FEATURES = False

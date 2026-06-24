import os
SITENUMBER = int(os.getenv("SITENUMBER", "48")) # default to 47 if not set
# make directory if not exists
output_dir = str(SITENUMBER) + "_" + "outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
PLOTTING = True
DEBUGGING = True

# create dictionary per site for timestep
TIMESTEP_DICT = {
    47: 200,
    48: 100,
    49: 100,
    50: 328,
    51: 200,
    52: 212,
    53: 200
}

TIMESTEP = TIMESTEP_DICT.get(SITENUMBER, 100) # default to 100 if site number not in dict

###############################################################################
# Load packages

import logging
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")

if DEBUGGING:
    FORMAT = 'INFO: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
else:
    logging.basicConfig(level=logging.WARNING)

# basic packages
from importlib_metadata import version
import platform
logging.info('The Python version is {}.'.format(platform.python_version()))
import random
import time

# data handling
import numpy as np
logging.info('The numpy version is {}.'.format(version('numpy')))
import dask
from dask.diagnostics import ProgressBar
logging.info('The dask version is {}.'.format(version('dask')))
import pandas as pd
logging.info('The pandas version is {}.'.format(version('pandas')))
import geopandas as gpd
logging.info('The geopandas version is {}.'.format(version('geopandas')))
import xarray as xr
logging.info('The xarray version is {}.'.format(version('xarray')))
import rasterio
from rasterio.features import rasterize
logging.info('The rasterio version is {}.'.format(version('rasterio')))
import rioxarray
logging.info('The rioxarray version is {}.'.format(version('rioxarray')))
import pyproj
logging.info('The pyproj version is {}.'.format(version('pyproj')))
import datetime
import spyndex
logging.info('The spyndex version is {}.'.format(version('spyndex')))
## xcube stuff
from xcube.core.store import new_data_store
from xcube.core.maskset import MaskSet
logging.info('The xcube version is {}.'.format(version('xcube')))

# AI and data processing
## scipy
import scipy
from scipy.spatial import cKDTree
logging.info('The scipy version is {}.'.format(version('scipy')))
## torch
import torch
logging.info('The torch version is {}.'.format(version('torch')))
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
## torchvision
import torchvision
logging.info('The torchvision version is {}.'.format(version('torchvision')))
from torchvision.transforms import v2
## lightning
import lightning as L
logging.info('The Lightning version is {}.'.format(version('lightning')))
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.accelerators import find_usable_cuda_devices

# plotting
# jupyter
import IPython.display
from IPython.display import GeoJSON
logging.info('The IPython version is {}.'.format(version('IPython')))
import shapely.geometry
from shapely.geometry import shape
logging.info('The shapely version is {}.'.format(version('shapely')))
# plot libs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
logging.info('The matplotlib version is {}.'.format(version('matplotlib')))
import seaborn as sns
logging.info('The Seaborn version is {}.'.format(version('seaborn')))

# Check for CUDA support
if torch.cuda.is_available():
    logging.info("CUDA is available. CUDA acceleration is supported.")
    logging.info(f"CUDA version: {torch.version.cuda}")
    available_devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    logging.info(f"Found {len(available_devices)} CUDA Devices:")
    for idx, gpu_id in enumerate(available_devices):
        logging.info(f"{idx:02}: {torch.cuda.get_device_name(gpu_id)}")
else:
    logging.info("CUDA is not available. CUDA acceleration is not supported.")

# Check for Mac GPU support
if torch.backends.mps.is_available():
    logging.info("Metal Performance Shaders (MPS) is available for GPU acceleration on Mac.")
else:
    logging.info("Metal Performance Shaders (MPS) is not available on this system.")

###############################################################################
# spectral indices computation

BANDID_TRANSLATOR = {}
for band in spyndex.bands:
    if hasattr(spyndex.bands.get(band), "sentinel2a"):
        s2_band = spyndex.bands.get(band).sentinel2a.band
        s2_band = s2_band[0] + s2_band[1:].zfill(2)
        BANDID_TRANSLATOR[band] = s2_band

def compute_spectral_indices(cube: xr.Dataset) -> xr.Dataset:
    ds = cube["s2l2a"].to_dataset(dim="band")
    indices = list(spyndex.indices.keys())

    # remove indices which are not provided by Sentinel-2
    for index in indices.copy():
        if "Sentinel-2" not in spyndex.indices.get(index).platforms:
            indices.remove(index)

    # remove index NIRvP, which needs Photosynthetically Available Radiation (PAR)
    # note that PAR is given by Sentinel-3 Level 2
    indices.remove("NIRvP") # Julia excludes this one, too
    
    # prepare the parameters for the mapping from expressions to data
    params = {}
    for band in BANDID_TRANSLATOR.keys():
        params[band] = ds[BANDID_TRANSLATOR[band]]

    extra = dict(
        # Kernel parameters
        kNN=1.0,
        kGG=1.0,
        kNR=spyndex.computeKernel(
            kernel="RBF",
            a=ds[BANDID_TRANSLATOR["N"]],
            b=ds[BANDID_TRANSLATOR["R"]],
            sigma=(
                ((ds[BANDID_TRANSLATOR["N"]] + ds[BANDID_TRANSLATOR["R"]]) / 2).median(
                    dim=["y", "x"]
                )
            ),
        ),
        kNB=spyndex.computeKernel(
            kernel="RBF",
            a=ds[BANDID_TRANSLATOR["N"]],
            b=ds[BANDID_TRANSLATOR["B"]],
            sigma=(
                ((ds[BANDID_TRANSLATOR["N"]] + ds[BANDID_TRANSLATOR["B"]]) / 2).median(
                    dim=["y", "x"]
                )
            ),
        ),
        kNL=spyndex.computeKernel(
            kernel="RBF",
            a=ds[BANDID_TRANSLATOR["N"]],
            b=spyndex.constants.L.default,
            sigma=(
                ((ds[BANDID_TRANSLATOR["N"]] + spyndex.constants.L.default) / 2).median(
                    dim=["y", "x"]
                )
            ),
        ),
        kGR=spyndex.computeKernel(
            kernel="RBF",
            a=ds[BANDID_TRANSLATOR["G"]],
            b=ds[BANDID_TRANSLATOR["R"]],
            sigma=(
                ((ds[BANDID_TRANSLATOR["G"]] + ds[BANDID_TRANSLATOR["R"]]) / 2).median(
                    dim=["y", "x"]
                )
            ),
        ),
        kGB=spyndex.computeKernel(
            kernel="RBF",
            a=ds[BANDID_TRANSLATOR["G"]],
            b=ds[BANDID_TRANSLATOR["B"]],
            sigma=(
                ((ds[BANDID_TRANSLATOR["G"]] + ds[BANDID_TRANSLATOR["B"]]) / 2).median(
                    dim=["y", "x"]
                )
            ),
        ),
        # Additional parameters
        L=spyndex.constants.L.default,
        C1=spyndex.constants.C1.default,
        C2=spyndex.constants.C2.default,
        g=spyndex.constants.g.default,
        gamma=spyndex.constants.gamma.default,
        alpha=spyndex.constants.alpha.default,
        sla=spyndex.constants.sla.default,
        slb=spyndex.constants.slb.default,
        nexp=spyndex.constants.nexp.default,
        cexp=spyndex.constants.cexp.default,
        k=spyndex.constants.k.default,
        fdelta=spyndex.constants.fdelta.default,
        epsilon=spyndex.constants.epsilon.default,
        omega=spyndex.constants.omega.default,
        beta=spyndex.constants.beta.default,
        # Wavelength parameters
        lambdaN=spyndex.bands.N.modis.wavelength,
        lambdaG=spyndex.bands.G.modis.wavelength,
        lambdaR=spyndex.bands.R.modis.wavelength,
        lambdaS1=spyndex.bands.S1.modis.wavelength,
    )
    params.update(extra)

    # calculate indices
    return spyndex.computeIndex(index=indices, params=params)

###############################################################################
# load data

# store_team=new_data_store(
#             "s3",
#             root=os.environ["S3_USER_STORAGE_BUCKET"],
#             max_depth=4,
#             storage_options=dict(
#                 anon=False,
#                 key=os.environ["S3_USER_STORAGE_KEY"],
#                 secret=os.environ["S3_USER_STORAGE_SECRET"],
#             )
# )
# ds = store_team.open_data("cubes/science/0.1.0/047.zarr")
sciencecube = xr.open_zarr("/Users/bp23keri/workspace/sciencecubes/0" + str(SITENUMBER) + ".zarr")
featurecube = xr.open_zarr("/Users/bp23keri/workspace/featurecubes/0" + str(SITENUMBER) + ".zarr")
data = sciencecube[["cloud_mask", "lccs_class", "esa_wc", "s2l2a"]]

print("Finished: Dataset loaded.")

# load shapefile
file = "data/" + str(SITENUMBER) + ".geojson"
shp = gpd.read_file(file)

# check if the CRS of the shapefile and the dataset match
logger.info("CRS: " + str(shp.crs))
logger.info("Dataset CRS: " + str(data.rio.crs))

if shp.crs != data.rio.crs:
    logger.info("CRS of the shapefile and the dataset do not match. Reprojecting the shapefile to match the dataset CRS.")
    shp = shp.to_crs(data.rio.crs)

logger.info(f"New CRS of the shapefile:")
logger.info(shp.crs)

print("Finished: Shapefile loaded.")

###############################################################################
# create mask and distance layer

# create boolean mask from shapefile geometry
mask_full = xr.DataArray(
    rasterize(
        [(geom, 1) for geom in shp.geometry],
        out_shape=(len(data.y), len(data.x)),
        transform=data.rio.transform(),
        fill=0,
        dtype="uint8"
    ),
    dims=("y", "x"),
    coords={"y": data.y, "x": data.x},
    name="LigniteMask"
)
# add mask to dataset
data["LigniteMask"] = mask_full

if PLOTTING:
    rgb = data.s2l2a.sel(band = ["B04", "B03", "B02"]).isel(time=TIMESTEP)
    fig = rgb.where(data['LigniteMask'] == 0).plot.imshow(robust=True, size = 10).get_figure()
    fig.savefig(str(SITENUMBER) + "_" + "outputs" + "/" + "LigniteMask" + ".png")

print("Finished: Lignite mask created.")

yy, xx = np.where(data['LigniteMask'].values == 1)
coord_df = pd.DataFrame({
    "x": data.x.values[xx],
    "y": data.y.values[yy]
})
distance_layer = xr.full_like(
    data["LigniteMask"],
    fill_value=np.Inf,
    dtype=float
)
distance_layer = distance_layer.where(data["LigniteMask"] == 0, 0)
data["DistanceToLignite"] = distance_layer

# create a KDTree from the coordinates of the lignite mines and query the tree for the distance to the nearest mine for each point in the grid
mine_points = np.column_stack([
    coord_df.x.values,
    coord_df.y.values
])
tree = cKDTree(mine_points)

# --- prepare coordinate (important for correct broadcasting) ---
mesh_X, mesh_Y = np.meshgrid(
    data.x.values,
    data.y.values
)
grid_points = np.column_stack([
    mesh_X.ravel(),
    mesh_Y.ravel()
])

grid_distances, _ = tree.query(grid_points, k=1)

temp_layer = grid_distances.reshape(mesh_X.shape)
data["DistanceToLignite"] = np.minimum(
    data["DistanceToLignite"],
    temp_layer
)

# visualize the distance to lignite mines
if PLOTTING:
    fig = data["DistanceToLignite"].plot.imshow(robust=True, size=10).get_figure()
    fig.savefig(str(SITENUMBER) + "_" + "outputs" + "/" + "DistanceToLignite" + ".png")

print("Finished: Distance layer created.")

#############################################################################
# masking and data preparation for aggregation

# combine masks
esa_wc_mask = MaskSet(data.esa_wc)
# processing_mask = esa_wc_mask.tree_cover | esa_wc_mask.grassland | esa_wc_mask.cropland
processing_mask = esa_wc_mask.tree_cover

# visualize the RGB composite of the Sentinel-2 data, masked by the processing mask, the lignite mask, and the cloud mask
if PLOTTING:
    rgb = data.s2l2a.sel(band = ["B04", "B03", "B02"])
    fig = rgb.where(processing_mask).where(data["LigniteMask"] == 0).where(data["cloud_mask"] == 0).isel(time=TIMESTEP).plot.imshow(robust=True, aspect = 1, size = 10).get_figure()
    fig.savefig(str(SITENUMBER) + "_" + "outputs" + "/" + "RGB_Composite" + ".png")

# compute spectral indices and add them to the dataset
data["SpectralIndices"] = compute_spectral_indices(data)
print("Finished: Spectral indices computed.")

# select the kNDVI index and the distance to lignite mines, masked by the processing mask, the lignite mask, and the cloud mask
kndvi_data = data["SpectralIndices"].sel(index = "kNDVI").where(processing_mask).where(data["LigniteMask"] == 0).where(data["cloud_mask"] == 0)
features_data = featurecube.where(processing_mask).where(data["LigniteMask"] == 0).where(data["cloud_mask"] == 0)
features_data = features_data["features"].to_dataset(dim="feature")
features_data = features_data.rename_vars({
    name: f"feature_{name}"
    for name in features_data.data_vars
})
kndvi_data = kndvi_data.sel(
    x=features_data.x,
    y=features_data.y
)
distance_data = data["DistanceToLignite"].where(kndvi_data.notnull())
lignite_ds = xr.merge([
    xr.Dataset({
        "kNDVI": kndvi_data,
        "DistanceToLignite": distance_data
    }),
    features_data
])
# additional chunking won't help because the data is already chunked in a way that allows for efficient processing but stacking might help
lignite_stack = lignite_ds.stack(pixel=("y", "x"))

print("Finished: Data prepared for aggregation.")
print("Start Processing...")

# select data to process
if DEBUGGING: # Debugging with random slices
    n = 30
    t_dim = lignite_stack.sizes["time"]
    idx = np.random.choice(t_dim, size=n, replace=False)
    tt = lignite_stack.isel(time=idx)
else: # Full Dataset
    tt = lignite_stack

# convert to dataframe and drop NaN values
ddf = tt.to_dataframe().dropna()
print("Finished: Data converted to DataFrame.")

# Save to CSV
ddf.to_csv(str(SITENUMBER) + "_" + "outputs" + "/" + "LigniteData" + ".csv")
print("Finished: Data saved to CSV.")

#################################################################################
# Post processing and visualization of the data

bin_size = 100  # meters
ddf["dist_bin"] = (ddf["DistanceToLignite"] // bin_size) * bin_size

agg = ddf.groupby("dist_bin")["kNDVI"].agg(
    mean="mean",
    std="std",
    count="count"
).reset_index()

# remove small bins with less then 1000 samples and bins with distance less than 1500m
agg = agg.where(agg["count"] > 1000).dropna()
agg = agg.where(agg["dist_bin"] > 1500).dropna()

agg["sem"] = agg["std"] / np.sqrt(agg["count"])
agg["ci_low"] = agg["mean"] - 1.96 * agg["sem"]
agg["ci_high"] = agg["mean"] + 1.96 * agg["sem"]

plt.figure(figsize=(9, 6))

# mean curve
plt.plot(
    agg["dist_bin"],
    agg["mean"],
    label="Mean kNDVI",
    linewidth=2
)

# confidence band
plt.fill_between(
    agg["dist_bin"],
    agg["ci_low"],
    agg["ci_high"],
    alpha=0.3,
    label="95% CI"
)

plt.xlabel("Distance to Lignite (m)")
plt.ylabel("kNDVI")
plt.title("Distance–Response Curve (kNDVI vs. Mining Distance)")
plt.legend()

plt.savefig(str(SITENUMBER) + "_" + "outputs" + "/" + "Distance_Response_Curve" + ".png")
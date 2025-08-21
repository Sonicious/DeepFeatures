import json

import pandas as pd
import segmentation_models_pytorch as smp
import torch
from xcube.core.store import new_data_store
from xcube.core.chunk import chunk_dataset
import zarr

import get_datasets
import constants
import utils
from version import version


def setup_cloudmask_model():
    checkpoint = torch.utils.model_zoo.load_url(constants.CLOUDMASK_MODEL_URL)
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        classes=4,
        in_channels=len(constants.CLOUDMASK_BANDS),
    )
    model.load_state_dict(checkpoint)
    model.eval()
    return model


if __name__ == "__main__":
    with open("s3-credentials.json") as f:
        s3_credentials = json.load(f)

    # initiate all data stores
    super_store = dict(
        store_team=new_data_store(
            "s3",
            root=s3_credentials["bucket"],
            max_depth=10,
            storage_options=dict(
                anon=False,
                key=s3_credentials["key"],
                secret=s3_credentials["secret"],
            ),
        ),
        store_dem=new_data_store(
            "s3",
            root="copernicus-dem-30m",
            storage_options=dict(
                anon=True, client_kwargs=dict(region_name="eu-central-1")
            ),
        ),
        store_lccs=new_data_store("s3", root="deep-esdl-public"),
        store_esa_wc=new_data_store(
            "s3",
            root="esa-worldcover",
            max_depth=10,
            storage_options=dict(
                anon=True, client_kwargs=dict(region_name="eu-central-1")
            ),
        ),
        cloudmask_model=setup_cloudmask_model(),
    )

    # loop over sites
    sites_params = pd.read_csv(constants.PATH_SITES_PARAMETERS_SCIENCE)
    for loc_idx in range(0, 1):
        for time_idx in range(2):
            constants.LOG.info(f"Generation of cube {loc_idx:04}_{time_idx} started.")

            path = (
                f"cubes/{constants.TRAINING_FOLDER_NAME}/{version}/"
                "{loc_idx:04}_{time_idx}.zarr"
            )
            if super_store["store_team"].has_data(path):
                constants.LOG.info(f"Cube {path} already generated.")
                continue

            # get Sentinel-2 data
            cube = get_datasets.get_s2l2a_single_training_year(
                super_store, loc_idx, time_idx, check_nan=True
            )
            if cube is None:
                continue
            constants.LOG.info(f"Open Sentinel-2 L2A.")

            # get attributes of cube
            time_range_start = cube.time.values[0]
            time_range_end = cube.time.values[-1]
            attrs = utils.readin_sites_parameters(
                sites_params,
                loc_idx,
                constants.TRAINING_FOLDER_NAME,
                size_bbox=0.9,
                time_range_start=time_range_start,
                time_range_end=time_range_end,
            )

            # apply BRDF correction
            cube = utils.apply_nbar(cube)
            constants.LOG.info(f"BRDF correction applied.")

            # reorgnaize cube
            cube = get_datasets.reorganize_cube(cube)
            constants.LOG.info(f"Cube reorgnaized.")

            # add cloud mask
            cube = get_datasets.get_cloudmask(super_store, cube)
            if cube is None:
                continue
            constants.LOG.info(f"Cloud mask added.")

            # add DEM
            cube = get_datasets.add_reprojected_dem(super_store, cube)
            constants.LOG.info(f"DEM added.")

            # add CCI land cover classification
            cube = get_datasets.add_reprojected_lccs(super_store, cube)
            constants.LOG.info(f"Land cover classification added.")

            # add ESA World Cover
            cube = get_datasets.add_reprojected_esa_wc(super_store, cube)
            constants.LOG.info(f"ESA World Cover added.")

            # add ERA5
            cube = get_datasets.add_era5(super_store, cube)
            constants.LOG.info(f"ERA5 data added.")

            # add grid_mapping to encoding
            for var in cube.data_vars:
                if "grid_mapping" in cube[var].attrs:
                    del cube[var].attrs["grid_mapping"]
                if "grid_mapping" in cube[var].encoding:
                    del cube[var].encoding["grid_mapping"]
                if cube[var].dims[-2:] == ("y", "x"):
                    cube[var].attrs["grid_mapping"] = "spatial_ref"
            constants.LOG.info(f"Grid mapping added to attrs.")

            # write final cube
            cube["band"] = cube.band.astype("str")
            cube.coords["angle"] = ["Zenith", "Azimuth"]
            cube = chunk_dataset(
                cube,
                chunk_sizes=dict(
                    angle=-1,
                    angle_x=-1,
                    angle_y=-1,
                    band=-1,
                    time=1,
                    time_era5=-1,
                    time_lccs=-1,
                    x=500,
                    y=500,
                ),
                format_name="zarr",
            )
            compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=1)
            encoding = {"s2l2a": {"compressor": compressor}}
            super_store["store_team"].write_data(
                cube, path, replace=True, encoding=encoding
            )
            constants.LOG.info(f"Final cube written to {path}.")

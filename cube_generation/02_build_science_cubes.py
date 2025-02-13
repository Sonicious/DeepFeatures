import json
import os

import pandas as pd
import segmentation_models_pytorch as smp
import torch
from xcube.core.store import new_data_store
import zarr

import get_datasets
import constants
import utils


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
    # get credentials for Sentinel Hub
    with open("sh-cdse-credentials.json") as f:
        credentials = json.load(f)

    # initiate all data stores
    super_store = dict(
        store_team=new_data_store(
            "s3",
            root=os.environ["S3_USER_STORAGE_BUCKET"],
            max_depth=4,
            storage_options=dict(
                anon=False,
                key=os.environ["S3_USER_STORAGE_KEY"],
                secret=os.environ["S3_USER_STORAGE_SECRET"],
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
    for idx in range(0, 1):
        constants.LOG.info(f"Generation of cube {idx} started.")
        # get attributes of cube
        attrs = utils.readin_sites_parameters(
            sites_params, idx, constants.SCIENCE_FOLDER_NAME
        )

        # get Sentinel-2 data
        cube = get_datasets.get_s2l2a(super_store, attrs)
        # cube = get_datasets.get_s2l2a_creodias_vm(super_store, attrs)
        constants.LOG.info(f"Sentinel-2 L2A retrieved.")

        # # apply BRDF correction
        # cube = utils.apply_nbar(cube)
        # constants.LOG.info(f"BRDF correction applied.")

        # allpy cloud mask
        cube = get_datasets.add_cloudmask(super_store, cube)
        constants.LOG.info(f"Cloud mask added.")

        # add DEM
        cube = get_datasets.add_reprojected_dem(super_store, cube)
        constants.LOG.info(f"DEM added.")

        # add CCI land cover classification
        cube = get_datasets.add_reprojected_lccs(super_store, cube)
        constants.LOG.info(f"Land cover classification added.")

        # add ESA World Cover
        cube = get_datasets.add_reprojected_esa_wc(super_store, cube)
        constants.LOG.info(f"Land cover classification added.")

        # add ERA5
        cube = get_datasets.add_era5(super_store, cube)
        constants.LOG.info(f"ERA5 data added.")

        # write final cube
        cube["band"] = cube.band.astype("str")
        compressor = zarr.Blosc(cname="zstd", clevel=5, shuffle=0)
        encoding = {"s2l2a": {"compressor": compressor}}
        super_store["store_team"].write_data(
            cube, cube.attrs["path"], replace=True, encoding=encoding
        )
        constants.LOG.info(f"Final cube written to {cube.attrs["path"]}.")

        # # delete temp directory
        # utils.delete_temp_files(super_store, attrs)
        # constants.LOG.info(f"Temporary files deleted.")

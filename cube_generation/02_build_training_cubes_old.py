import os

import pandas as pd
import segmentation_models_pytorch as smp
import torch
from xcube.core.store import new_data_store

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
        cloudmask_model=setup_cloudmask_model(),
    )

    # loop over sites
    sites_params = pd.read_csv(constants.PATH_SITES_PARAMETERS_TRAINING)
    for idx in range(0, 500):
        constants.LOG.info(f"Generation of cube {idx} started.")
        # get attributes of cube
        attrs = utils.readin_sites_parameters(
            sites_params, idx, constants.TRAINING_FOLDER_NAME
        )

        # get Sentinel-2 data
        cube = get_datasets.get_s2l2a_creodias_vm(super_store, attrs)
        constants.LOG.info(f"Sentinel-2 data loaded.")

        # allpy cloud mask
        cube = get_datasets.add_cloudmask(super_store, cube)
        constants.LOG.info(f"Cloud mask added.")

        # write final cube
        cube["band"] = cube.band.astype("str")
        path = (
            f"cubes/{constants.TRAINING_NB_CUBES}/{version}/{cube.attrs['id']:06}.zarr",
        )
        super_store["store_team"].write_data(cube, path, replace=True)
        constants.LOG.info(f"Training cube written to {cube.attrs['path']}.")

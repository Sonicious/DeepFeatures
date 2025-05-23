import json

import pandas as pd
import segmentation_models_pytorch as smp
import torch
from xcube.core.store import new_data_store
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
        cloudmask_model=setup_cloudmask_model(),
    )

    # loop over sites
    sites_params = pd.read_csv(constants.PATH_SITES_PARAMETERS_SCIENCE_SENTINEL2)
    for idx in range(0, 71):
        constants.LOG.info(f"Cloud mask calculation of cube {idx} started.")
        # get attributes of cube
        attrs = utils.readin_sites_parameters(
            sites_params, idx, constants.SCIENCE_FOLDER_NAME
        )
        path = f"cubes/temp/{constants.SCIENCE_FOLDER_NAME}/{version}/{attrs['id']:03}_cloudmask.zarr"
        if super_store["store_team"].has_data(path):
            constants.LOG.info(f"Cloud mask already exists at {path}.")
            continue

        # get Sentinel-2 data
        cube = get_datasets.get_s2l2a(super_store, attrs)
        if cube is None:
            continue
        constants.LOG.info(f"Open Sentinel-2 L2A.")

        # apply BRDF correction
        cube = utils.apply_nbar(cube)
        constants.LOG.info(f"BRDF correction applied.")

        # reorgnaize cube
        cube = get_datasets.reorganize_cube(cube)
        constants.LOG.info(f"Cube reorgnaized.")

        # add cloud mask
        cube = get_datasets.add_cloudmask(super_store, cube)
        constants.LOG.info(f"Cloud mask added.")

        # write final cube
        cube = cube[["cloud_mask"]]

        super_store["store_team"].write_data(cube, path, replace=True)
        constants.LOG.info(f"Cloud mask written to {path}.")

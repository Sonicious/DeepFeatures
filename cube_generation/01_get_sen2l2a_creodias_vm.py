import json
import time

import pandas as pd
import xarray as xr
from xcube.core.store import new_data_store

import constants
import utils


def _retry_get_sen2l2(super_store: dict, attrs: dict) -> xr.Dataset:
    retries = 10
    for attempt in range(1, retries + 1):
        try:
            data_id_components = attrs["path"].split("/")
            fname = f"{attrs['site_id']:06}_s2l2a.zarr"
            data_id = f"{'/'.join(data_id_components[:-1])}/temp/{fname}"

            if not super_store["store_team"].has_data(data_id):
                ds = super_store["store_stac"].open_data(
                    data_id="SENTINEL-2",
                    bbox=attrs["bbox_utm"],
                    crs=f"EPSG:326{attrs["utm_zone"][:2]}",
                    spatial_res=constants.SPATIAL_RES,
                    time_range=[attrs["time_range_start"], attrs["time_range_end"]],
                    apply_scaling=True,
                    asset_names=[
                        "B01",
                        "B02",
                        "B03",
                        "B04",
                        "B05",
                        "B06",
                        "B07",
                        "B08",
                        "B8A",
                        "B09",
                        "B11",
                        "B12",
                        "SCL",
                    ],
                )
                constants.LOG.info(f"Writing of cube {idx} started.")
                super_store["store_team"].write_data(ds, data_id, replace=True)
            return "Done"
        except Exception as e:
            if attempt == retries:
                raise
            constants.LOG.info(f"Attempt {attempt} failed: {e}. Retrying in 1 seconds.")
            time.sleep(1)


def get_s2l2a_creodias_vm(super_store: dict, attrs: dict):
    response = _retry_get_sen2l2(super_store, attrs)


if __name__ == "__main__":
    with open("s3-credentials.json") as f:
        s3_credentials = json.load(f)

    super_store = dict(
        store_stac=new_data_store("stac-cdse", stack_mode="odc-stac", creodias_vm=True),
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
    )

    sites_params = pd.read_csv(constants.PATH_SITES_PARAMETERS_TRAINING)
    for idx in range(0, 500):
        constants.LOG.info(f"Generation of cube {idx} started.")
        attrs = utils.readin_sites_parameters(
            sites_params,
            idx,
            constants.TRAINING_FOLDER_NAME,
        )
        get_s2l2a_creodias_vm(super_store, attrs)

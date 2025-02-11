import json
import pandas as pd
from xcube.core.store import new_data_store

import constants
import utils


def get_s2l2a(super_store: dict, attrs: dict):
    data_id = utils.get_temp_file(attrs)
    # if not super_store["store_team"].has_data(data_id):
    if True:
        ds = super_store["store_stac"].open_data(
            data_id="sentinel-2-l2a",
            bbox=attrs["bbox_utm"],
            crs=f"EPSG:326{attrs["utm_zone"][:2]}",
            spatial_res=constants.SPATIAL_RES,
            time_range=["2019-01-01", "2019-12-31"],
            apply_scaling=True,
            angles_sentinel2=True,
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
        constants.LOG.info(f"Writing of cube {idx} to {data_id} started.")
        super_store["store_team"].write_data(ds, data_id, replace=True)
        constants.LOG.info(f"Writing of cube {idx} to {data_id} finished.")


if __name__ == "__main__":
    with open("s3-credentials.json") as f:
        s3_credentials = json.load(f)

    super_store = dict(
        store_stac=new_data_store("stac-cdse", stack_mode=True, creodias_vm=True),
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

    sites_params = pd.read_csv(constants.PATH_SITES_PARAMETERS_SCIENCE)
    for idx in range(0, 1):
        constants.LOG.info(f"Generation of cube {idx} started.")
        attrs = utils.readin_sites_parameters(
            sites_params,
            idx,
            constants.SCIENCE_FOLDER_NAME,
        )
        get_s2l2a(super_store, attrs)

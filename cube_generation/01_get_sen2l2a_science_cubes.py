import json
import pandas as pd
from xcube.core.store import new_data_store

from version import version
import constants
import utils


def get_s2l2a(super_store: dict, site_params: pd.Series):
    bbox = utils.create_utm_bounding_box(
        site_params["lat"], site_params["lon"], box_size_km=10
    )
    data_id = f"cubes/temp/{constants.SCIENCE_FOLDER_NAME}/{version}/{idx:03}.zarr"

    def _get_s2l2a_year(time_range: list[str]):
        data_id_mod = data_id.replace(".zarr", f"_{time_range[1][:4]}.zarr")
        if not super_store["store_team"].has_data(data_id):
            constants.LOG.info(f"Open cube {idx} for year {time_range[1][:4]}.")
            ds = super_store["store_stac"].open_data(
                data_id="sentinel-2-l2a",
                bbox=bbox["bbox_utm"],
                crs=f"EPSG:326{bbox["utm_zone"][:2]}",
                spatial_res=constants.SPATIAL_RES,
                time_range=time_range,
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

            constants.LOG.info(f"Writing of cube to {data_id_mod} started.")
            super_store["store_team"].write_data(ds, data_id_mod, replace=True)
            constants.LOG.info(f"Writing of cube to {data_id_mod} finished.")
        else:
            constants.LOG.info(f"Cube {data_id_mod} already retrieved.")

    time_ranges = [
        ["2016-11-01", "2017-12-31"],
        ["2018-01-01", "2018-12-31"],
        ["2019-01-01", "2019-12-31"],
        ["2020-01-01", "2020-12-31"],
        ["2021-01-01", "2021-12-31"],
        ["2022-01-01", "2022-12-31"],
        ["2023-01-01", "2023-12-31"],
        ["2024-01-01", "2024-12-31"],
    ]
    for time_range in time_ranges:
        _get_s2l2a_year(time_range)


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

    sites_params = pd.read_csv(constants.PATH_SITES_PARAMETERS_SCIENCE_SENTINEL2)
    for idx in range(0, 10):
        constants.LOG.info(f"Generation of cube {idx} started.")
        site_params = sites_params.loc[idx]
        get_s2l2a(super_store, site_params)

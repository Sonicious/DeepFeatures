import json
import datetime

import pandas as pd
import numpy as np
from xcube.core.store import new_data_store

from version import version
import constants
import utils


def generate_time_interval() -> tuple[tuple[str, str], tuple[str, str]]:
    dt_start = datetime.date.fromisoformat(constants.DT_START)
    dt_end = datetime.date.fromisoformat(constants.DT_END)
    nb_days = (dt_end - dt_start).days
    half_year = constants.TRAINING_TEMPORAL_RANGE_DAYS // 2
    delta_days0 = np.random.randint(half_year, nb_days - half_year)
    if delta_days0 < nb_days / 2:
        delta_days1 = np.random.randint(
            delta_days0 + 3 * half_year, nb_days - half_year
        )
    else:
        delta_days1 = np.random.randint(half_year, delta_days0 - 3 * half_year)

    dt0 = dt_start + datetime.timedelta(days=delta_days0)
    dt1 = dt_start + datetime.timedelta(days=delta_days1)
    td = datetime.timedelta(days=half_year)
    return (
        ((dt0 - td).isoformat(), (dt0 + td).isoformat()),
        ((dt1 - td).isoformat(), (dt1 + td).isoformat()),
    )


def get_s2l2a(super_store: dict, site_params: pd.Series):
    bbox = utils.create_utm_bounding_box(
        site_params["lat"], site_params["lon"], box_size_km=0.9
    )
    data_id = f"cubes/temp/{constants.TRAINING_FOLDER_NAME}/{version}/{idx:04}.zarr"

    def _get_s2l2a_year(time_range: list[str], data_id_mod: str):
        if not super_store["store_team"].has_data(data_id_mod):
            constants.LOG.info(f"Open cube {idx} for year {time_range[1][:4]}.")
            ds = super_store["store_stac"].open_data(
                data_id="sentinel-2-l2a",
                bbox=bbox["bbox_utm"],
                crs=f"EPSG:326{bbox["utm_zone"][:2]}",
                spatial_res=constants.SPATIAL_RES,
                time_range=time_range,
                apply_scaling=True,
                add_angles=True,
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
            print(ds)
            constants.LOG.info(f"Writing of cube to {data_id_mod} started.")
            super_store["store_team"].write_data(ds, data_id_mod, replace=True)
            constants.LOG.info(f"Writing of cube to {data_id_mod} finished.")
        else:
            constants.LOG.info(f"Cube {data_id_mod} already retrieved.")

    time_ranges = generate_time_interval()
    for time_idx, time_range in enumerate(time_ranges):
        data_id_mod = data_id.replace(".zarr", f"_{time_idx}.zarr")
        for attempt in range(1, 4):
            try:
                _get_s2l2a_year(time_range, data_id_mod)
                break
            except Exception as e:
                if super_store["store_team"].has_data(data_id_mod):
                    super_store["store_team"].delete_data(data_id_mod)
                constants.LOG.error(f"Attempt {attempt} failed: {e}")
                if attempt == 3:
                    constants.LOG.info(
                        f"Cube {data_id_mod} tried to retrieve {attempt} times. "
                        f"We go on..."
                    )


if __name__ == "__main__":
    with open("s3-credentials.json") as f:
        s3_credentials = json.load(f)
    with open("cdse-credentials.json") as f:
        cdse_credentails = json.load(f)

    super_store = dict(
        store_stac=new_data_store("stac-cdse-ardc", **cdse_credentails),
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
    for idx in range(1):
        constants.LOG.info(f"Generation of cube {idx} started.")
        site_params = sites_params.loc[idx]
        get_s2l2a(super_store, site_params)

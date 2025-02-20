import datetime

import numpy as np
import pandas as pd
from sen2nbar import c_factor
import spyndex
import utm
import xarray as xr

from constants import BANDID_TRANSLATOR
from constants import DT_START
from constants import DT_END
from constants import SITES_LON_LABEL
from constants import SITES_LAT_LABEL
from constants import SPATIAL_RES
from version import version


def readin_sites_parameters(
    sites_params: pd.DataFrame,
    index: int,
    folder_name: str,
) -> dict:
    site_params = sites_params.loc[index]
    lat = float(site_params[SITES_LAT_LABEL])
    lon = float(site_params[SITES_LON_LABEL])
    if "size_bbox" in site_params:
        bbox = create_utm_bounding_box(lat, lon, box_size_km=site_params["size_bbox"])
    else:
        bbox = create_utm_bounding_box(lat, lon)
    cube_attrs = dict(
        site_id=index,
        path=f"cubes/{folder_name}/{version}/{index:06}.zarr",
        center_wgs84=bbox["center_wgs84"],
        center_utm=bbox["center_utm"],
        bbox_wgs84=bbox["bbox_wgs84"],
        bbox_utm=bbox["bbox_utm"],
        utm_zone=bbox["utm_zone"],
        version=version,
        creation_datetime=datetime.datetime.now().isoformat(),
        last_modified_datetime=datetime.datetime.now().isoformat(),
        landcover_first=None,
        landcover_first_percentage=None,
        landcover_second=None,
        landcover_second_percentage=None,
        protection_mask=None,
        acknowledgment="DeepFeatures project",
        contributor_name="Brockmann Consult GmbH",
        contributor_url="www.brockmann-consult.de",
        creator_email="info@brockmann-consult.de",
        creator_name="Brockmann Consult GmbH",
        creator_url="www.brockmann-consult.de",
        institution="Brockmann Consult GmbH",
        project="DeepExtreme",
        publisher_email="info@brockmann-consult.de",
        publisher_name="Brockmann Consult GmbH",
    )
    keys_in = [
        "Ground measurement [Y/N]",
        "Protection status [Y/N]",
        "elevation above mean sea level [m]\nmainly for flux towers",
    ]
    keys_out = [
        "ground_measurement",
        "protection_status",
        "flux_tower_elevation",
    ]
    for key_in, key_out in zip(keys_in, keys_out):
        if key_in in site_params:
            cube_attrs[key_out] = site_params[key_in]
        else:
            cube_attrs[key_out] = None

    if "time_range_start" in site_params:
        cube_attrs["time_range_start"] = site_params["time_range_start"]
    else:
        cube_attrs["time_range_start"] = DT_START
    if "time_range_end" in site_params:
        cube_attrs["time_range_end"] = site_params["time_range_end"]
    else:
        cube_attrs["time_range_end"] = DT_END
    return cube_attrs


def create_utm_bounding_box(
    latitude: float, longitude: float, box_size_km: float = 10
) -> dict:
    # Convert WGS84 coordinates to UTM
    easting, northing, zone_number, zone_letter = utm.from_latlon(latitude, longitude)

    # Calculate half the size of the box in meters (5 km in each direction)
    # reduce the half size by half a pixel, because xcube evaluates the values at
    # the center of a pixel. Otherwise we get 1001x1001pixels instead of 1000x1000 pixel. 
    box_size_m = box_size_km * 1000 - SPATIAL_RES
    half_size_m = int(box_size_m / 2)

    # Calculate the coordinates of the bounding box corners, rounded to full meters
    easting_min = int(easting - half_size_m)
    northing_min = int(northing - half_size_m)
    easting_max = easting_min + box_size_m
    northing_max = northing_min + box_size_m

    # transform the bounds to lat lon
    point_west_south = utm.to_latlon(
        easting_min, northing_min, zone_number, zone_letter
    )
    point_east_north = utm.to_latlon(
        easting_max, northing_max, zone_number, zone_letter
    )
    bounding_box = {
        "center_utm": [float(northing), float(easting)],
        "center_wgs84": [latitude, longitude],
        "bbox_utm": [easting_min, northing_min, easting_max, northing_max],
        "bbox_wgs84": [
            float(point_west_south[1]),
            float(point_west_south[0]),
            float(point_east_north[1]),
            float(point_east_north[0]),
        ],
        "utm_zone": f"{zone_number}{zone_letter}",
    }
    return bounding_box


def apply_nbar(cube: xr.Dataset) -> xr.Dataset:
    solar_azimuth = cube.solar_angle.sel(angle="Azimuth").drop_vars("angle")
    viewing_azimuth = cube.viewing_angle.sel(angle="Azimuth").drop_vars("angle")
    rel_azimuth = solar_azimuth - viewing_azimuth
    c_fac = c_factor.c_factor(
        cube.solar_angle.sel(angle="Zenith").drop_vars("angle"),
        cube.viewing_angle.sel(angle="Zenith").drop_vars("angle"),
        rel_azimuth,
    )
    c_fac = c_fac.rename(dict(angle_x="x", angle_y="y"))
    c_fac = c_fac.interp(
        y=cube.y.values,
        x=cube.x.values,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    c_fac = c_fac.to_dataset(dim="band")
    for var in c_fac:
        cube[var] *= c_fac[var]
    return cube


def compute_spectral_indices(cube: xr.Dataset) -> xr.Dataset:
    ds = cube["s2l2a"].to_dataset(dim="band")
    indices = list(spyndex.indices.keys())

    # remove indices which are not provided by Sentinel-2
    for index in indices.copy():
        if "Sentinel-2" not in spyndex.indices.get(index).platforms:
            indices.remove(index)

    # remove index NIRvP, which needs Photosynthetically Available Radiation (PAR)
    # note that PAR is given by Sentinel-3 Level 2
    indices.remove("NIRvP")

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
    cube["spec_indices"] = spyndex.computeIndex(index=indices, params=params)

    return cube


def get_temp_file(attrs: dict) -> str:
    data_id_components = attrs["path"].split("/")
    fname = f"{attrs['site_id']:06}_s2l2a.zarr"
    return f"{data_id_components[0]}/temp/{'/'.join(data_id_components[1:-1])}/{fname}"


def delete_temp_files(super_store: dict, attrs: dict) -> None:
    assert super_store["store_team"].has_data(
        attrs["path"]
    ), f"final cube not written to {attrs["path"]}"
    data_id = get_temp_file(attrs)
    super_store["store_team"].delete_data(data_id)


def update_dict(dic: dict, dic_update: dict, inplace: bool = True) -> dict:
    if not inplace:
        dic = copy.deepcopy(dic)
    for key, val in dic_update.items():
        if isinstance(val, dict):
            dic[key] = update_dict(dic.get(key, {}), val)
        else:
            dic[key] = val
    return dic

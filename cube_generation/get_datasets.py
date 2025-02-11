import datetime

import affine
import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
from xcube.core.geom import clip_dataset_by_geometry

import constants
import utils


# def get_s2l2a(super_store: dict, attrs: dict) -> xr.Dataset:
#     data_id_components = attrs["path"].split("/")
#     fname = f"{attrs['site_id']:06}_s2l2a.zarr"
#     data_id = f"{'/'.join(data_id_components[:-1])}/temp/{fname}"
#
#     if not super_store["store_team"].has_data(data_id):
#         variable_names = [v for v in constants.BANDID_TRANSLATOR.values()]
#         variable_names = variable_names + ["SCL"]
#         ds = super_store["store_sh"].open_data(
#             "S2L2A",
#             variable_names=variable_names,
#             bbox=attrs["bbox_utm"],
#             crs=f"EPSG:326{attrs["utm_zone"][:2]}",
#             spatial_res=constants.SPATIAL_RES,
#             time_range=[attrs["time_range_start"], attrs["time_range_end"]],
#             tile_size=[500, 500],
#             mosaicking_order="leastCC",
#         )
#         ds = ds.drop_vars("time_bnds")
#         scl = ds.SCL
#         crs = ds.crs
#         ds = ds.drop_vars(["SCL", "crs"])
#         xcube_sh_attrs = ds.attrs
#         s2l2a = ds.to_dataarray(dim="band")
#         s2l2a = s2l2a.sel(
#             band=[
#                 "B01",
#                 "B02",
#                 "B03",
#                 "B04",
#                 "B05",
#                 "B06",
#                 "B07",
#                 "B08",
#                 "B8A",
#                 "B09",
#                 "B11",
#                 "B12",
#             ]
#         )
#         s2l2a = s2l2a.chunk(
#             chunks=dict(
#                 band=s2l2a.sizes["band"],
#                 time=constants.CHUNKSIZE_TIME,
#                 x=ds.sizes["x"],
#                 y=ds.sizes["y"],
#             )
#         )
#         cube = xr.Dataset()
#         cube["s2l2a"] = s2l2a
#         cube["crs"] = crs
#         cube["scl"] = scl.chunk(
#             chunks=dict(time=constants.CHUNKSIZE_TIME, x=ds.sizes["x"], y=ds.sizes["y"])
#         )
#         cube["scl"].attrs = dict(
#             flag_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
#             flag_meanings=(
#                 "no_data saturated_or_defective_pixel topographic_casted_shadows "
#                 "cloud_shadows vegetation not_vegetation water "
#                 "unclassified cloud_medium_probability "
#                 "cloud_high_probability thin_cirrus snow_or_ice"
#             ),
#             flag_colors=(
#                 "#000000 #ff0000 #2f2f2f #643200 #00a000 #ffe65a #0000ff "
#                 "#808080 #c0c0c0 #ffffff #64c8ff #ff96ff"
#             ),
#         )
#         cube.attrs = attrs
#         xcube_sh_attrs["home_url"] = "https://www.sentinel-hub.com/"
#         xcube_sh_attrs["data_url"] = "https://www.sentinel-hub.com/explore/data/"
#         xcube_sh_attrs["license_url"] = (
#             "https://open.esa.int/copernicus-sentinel-"
#             "satellite-imagery-under-open-licence/"
#         )
#         cube.attrs["xcube_sh_attrs"] = xcube_sh_attrs
#         cube.attrs["affine_transform"] = cube.rio.transform()
#         super_store["store_team"].write_data(cube, data_id, replace=True)
#
#     cube = super_store["store_team"].open_data(data_id)
#     if cube.attrs["center_wgs84"] != attrs["center_wgs84"]:
#         constants.LOG.warning(
#             f"Location in the desired attributes {attrs['center_wgs84']} "
#             f"does not fit to the location stored in S2L1A "
#             f"cube{cube.attrs['center_wgs84']} "
#         )
#     cube.attrs = utils.update_dict(cube.attrs, attrs)
#     return cube


def get_s2l2a_creodias_vm(super_store: dict, attrs: dict) -> xr.Dataset:
    data_id = utils.get_temp_file(attrs)
    ds = super_store["store_team"].open_data(data_id)
    scl = ds.SCL.astype(np.int8)
    ds = ds.drop_vars(["SCL"])
    xcube_stac_attrs = ds.attrs
    s2l2a = ds.to_dataarray(dim="band")
    s2l2a = s2l2a.sel(
        band=[
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
        ]
    )
    s2l2a = s2l2a.chunk(
        chunks=dict(
            band=s2l2a.sizes["band"],
            time=constants.CHUNKSIZE_TIME,
            x=s2l2a.sizes["x"],
            y=s2l2a.sizes["y"],
        )
    )
    scl = scl.chunk(
        chunks=dict(
            time=constants.CHUNKSIZE_TIME,
            x=scl.sizes["x"],
            y=scl.sizes["y"],
        )
    )
    cube = xr.Dataset()
    cube["s2l2a"] = s2l2a
    cube["scl"] = scl
    cube["scl"].attrs = dict(
        flag_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        flag_meanings=(
            "no_data saturated_or_defective_pixel topographic_casted_shadows "
            "cloud_shadows vegetation not_vegetation water "
            "unclassified cloud_medium_probability "
            "cloud_high_probability thin_cirrus snow_or_ice"
        ),
        flag_colors=(
            "#000000 #ff0000 #2f2f2f #643200 #00a000 #ffe65a #0000ff "
            "#808080 #c0c0c0 #ffffff #64c8ff #ff96ff"
        ),
    )
    cube.attrs = attrs
    xcube_stac_attrs["data_url"] = (
        "https://documentation.dataspace.copernicus.eu/APIs/S3.html"
    )
    cube.attrs["xcube_stac_attrs"] = xcube_stac_attrs
    cube.attrs["affine_transform"] = cube.rio.transform()
    cube.attrs = utils.update_dict(cube.attrs, attrs)
    return cube


# def get_s2l2a_angles(super_store: dict, attrs: dict) -> xr.Dataset:
#     data_id_components = attrs["path"].split("/")
#     fname = f"{attrs['site_id']:06}_s2l2a_angles.zarr"
#     data_id = f"{'/'.join(data_id_components[:-1])}/temp/{fname}"
#
#     if not super_store["store_team"].has_data(data_id):
#         bbox = attrs["bbox_utm"]
#         bbox = [bbox[0] - 500, bbox[1] - 500, bbox[2] + 500, bbox[3] + 500]
#         variable_names = [
#             "sunAzimuthAngles",
#             "sunZenithAngles",
#             "viewAzimuthMean",
#             "viewZenithMean",
#         ]
#         if "training" in data_id_components:
#             spatial_res = 50
#         else:
#             spatial_res = 1000
#         ds = super_store["store_sh"].open_data(
#             "S2L2A",
#             variable_names=variable_names,
#             bbox=bbox,
#             crs=f"EPSG:326{attrs["utm_zone"][:2]}",
#             spatial_res=spatial_res,
#             time_range=[attrs["time_range_start"], attrs["time_range_end"]],
#             upsampling="BILINEAR",
#             downsampling="BILINEAR",
#         )
#         ds = ds.drop_vars("time_bnds")
#         ds = ds.chunk(
#             chunks=dict(time=ds.sizes["time"], x=ds.sizes["x"], y=ds.sizes["y"])
#         )
#         super_store["store_team"].write_data(ds, data_id, replace=True)
#
#     ds = super_store["store_team"].open_data(data_id)
#     return ds


def add_cloudmask(super_store: dict, cube: xr.Dataset) -> xr.Dataset:
    s2l2a = cube.s2l2a
    s2l2a = s2l2a.sel(band=constants.CLOUDMASK_BANDS)
    s2l2a = s2l2a.transpose(*constants.CLOUDMASK_COORDS)

    dataarrays_to_merge = []
    for time_step in range(0, s2l2a.sizes["time"], constants.CLOUDMASK_BATCHSIZE_TIME):
        s2l2a_sub = s2l2a.isel(
            time=slice(time_step, time_step + constants.CLOUDMASK_BATCHSIZE_TIME)
        )
        res = _compute_earthnet_cloudmask(super_store, s2l2a_sub)
        dataarrays_to_merge.append(res)
    cloud_mask = xr.concat(dataarrays_to_merge, dim="time")
    cloud_mask = cloud_mask.chunk(
        chunks=dict(
            time=constants.CHUNKSIZE_TIME,
            x=cloud_mask.sizes["x"],
            y=cloud_mask.sizes["y"],
        )
    )
    cube["cloud_mask"] = cloud_mask
    cube["cloud_mask"].attrs = dict(
        flag_values=[0, 1, 2, 3],
        flag_meanings="clear thick_cloud thin_cloud cloud_shadow",
        flag_colors="#000000 #FFFFFF #D3D3D3 #636363",
    )
    attrs = {}
    attrs["description"] = (
        "Cloudmask generated using an AI approach following the implementation "
        "of EarthNet Minicuber, based on CloudSEN12."
    )
    attrs["home_url"] = "https://github.com/earthnet2021/earthnet-minicuber"
    attrs["cloudsen12_url"] = "https://cloudsen12.github.io/"
    cube.attrs["cloud_mask_attrs"] = attrs
    return cube


def add_reprojected_lccs(super_store: dict, cube: xr.Dataset) -> xr.Dataset:
    mlds_lc = super_store["store_lccs"].open_data(constants.DATA_ID_LAND_COVER_CLASS)
    lc = mlds_lc.base_dataset

    # clip LC dataset by data cube geometry
    buffer = 0.005
    bbox = cube.attrs["bbox_wgs84"]
    bbox = [bbox[0] - buffer, bbox[1] - buffer, bbox[2] + buffer, bbox[3] + buffer]
    lc = clip_dataset_by_geometry(lc, bbox)

    # clip LC dataset by time
    start = datetime.datetime(2015, 12, 31)
    end = datetime.datetime(2022, 1, 2)
    lc = lc.sel(time=slice(start, end))

    lc.rio.write_crs(lc.crs.attrs["wkt"], inplace=True)
    lc = lc.rename(dict(lon="x", lat="y", time="time_lccs"))
    lc = lc.drop_vars(["lat_bounds", "lon_bounds", "time_bounds", "crs"])
    lc_reproject = lc.rio.reproject(
        f"EPSG:326{cube.attrs['utm_zone'][:2]}",
        shape=(cube.sizes["y"], cube.sizes["x"]),
        transform=affine.Affine(*cube.attrs["affine_transform"]),
        resampling=rasterio.enums.Resampling.nearest,
    )
    lc_reproject = lc_reproject.drop_vars("spatial_ref")
    name_dict = {
        "change_count": "lccs_change_count",
        "current_pixel_state": "lccs_current_pixel_state",
        "lccs_class": "lccs_class",
        "observation_count": "lccs_observation_count",
        "processed_flag": "lccs_processed_flag",
    }
    for key, val in name_dict.items():
        cube[val] = lc_reproject[key].chunk(
            dict(
                time_lccs=lc_reproject.sizes["time_lccs"],
                x=lc_reproject.sizes["x"],
                y=lc_reproject.sizes["y"],
            )
        )
    attrs = lc_reproject.attrs
    attrs["home_url"] = (
        "https://cds-beta.climate.copernicus.eu/datasets/"
        "satellite-land-cover?tab=overview"
    )
    attrs["data_url"] = (
        "https://cds-beta.climate.copernicus.eu/datasets/"
        "satellite-land-cover?tab=download"
    )
    attrs["license_url"] = (
        "https://object-store.os-api.cci2.ecmwf.int/cci2-prod-catalogue/licences"
        "/satellite-land-cover/satellite-land-cover_8423d13d3dfd95bbeca92d935551"
        "6f21de90d9b40083a915ead15a189d6120fa.pdf"
    )
    cube.attrs["lccs_attrs"] = attrs

    # fill in attributes regarding land cover classification
    lc_first = []
    lc_first_percentage = []
    lc_second = []
    lc_second_percentage = []
    for i in range(lc_reproject.sizes["time_lccs"]):
        arr = lc_reproject.lccs_class.values[i, :, :]
        vals, counts = np.unique(arr, return_counts=True)
        idx = np.argmax(counts)
        lc_first.append(int(vals[idx]))
        lc_first_percentage.append(float(counts[idx] / arr.size))
        vals = np.delete(vals, idx)
        counts = np.delete(counts, idx)
        idx = np.argmax(counts)
        lc_second.append(int(vals[idx]))
        lc_second_percentage.append(float(counts[idx] / arr.size))
    cube.attrs["landcover_first"] = lc_first
    cube.attrs["landcover_first_percentage"] = lc_first_percentage
    cube.attrs["landcover_second"] = lc_second
    cube.attrs["landcover_second_percentage"] = lc_second_percentage
    return cube


def add_reprojected_esa_wc(super_store: dict, cube: xr.Dataset) -> xr.Dataset:
    ds_esa_wc, files = _load_esa_wc_data(super_store, cube.attrs["bbox_wgs84"])
    esa_wc_reproject = ds_esa_wc.rio.reproject(
        f"EPSG:326{cube.attrs['utm_zone'][:2]}",
        shape=(cube.sizes["y"], cube.sizes["x"]),
        transform=affine.Affine(*cube.attrs["affine_transform"]),
        resampling=rasterio.enums.Resampling.bilinear,
    )
    esa_wc_reproject = esa_wc_reproject.rename(dict(time="time_lccs"))
    cube["esa_wc"] = esa_wc_reproject["band_1"].chunk(
        dict(
            time_lccs=esa_wc_reproject.sizes["time_lccs"],
            x=esa_wc_reproject.sizes["x"],
            y=esa_wc_reproject.sizes["y"],
        )
    )
    attrs = {}
    attrs["home_url"] = "https://esa-worldcover.org"
    attrs["data_url"] = "https://esa-worldcover.org/en/data-access"
    attrs["license_url"] = "https://creativecommons.org/licenses/by/4.0/"
    attrs["files"] = files
    cube.attrs["esa_wc_attrs"] = attrs
    return cube


ERA5_AGGREGATION_ALL = [
    "2dm",
    "skt",
    "sp",
    "msl",
    "stl1",
    "stl2",
    "stl3",
    "stl4",
    "swvl1",
    "swvl2",
    "swvl3",
    "swvl4",
    "t2m",
    "u10, v10",
]
ERA5_AGGREGATION_MEAN_MEDIAN_STD = ["lai_hv", "lai_lv"]
ERA5_AGGREGATION_SUM = ["e", "pev", "slhf", "sshf", "ssr", "ssrd", "str", "strd", "tp"]


def add_era5(super_store: dict, cube: xr.Dataset) -> xr.Dataset:
    data_ids = super_store["store_team"].list_data_ids()
    data_ids = [data_id for data_id in data_ids if "cubes/aux/era5/" in data_id]
    era5_steps = []
    for data_id in data_ids:
        era5 = super_store["store_team"].open_data(data_id)
        era5_cube = era5.interp(
            lat=cube.attrs["center_wgs84"][0],
            lon=cube.attrs["center_wgs84"][1],
            method="linear",
        )
        era5_cube = era5_cube.drop(["expver", "number", "lat", "lon"])

        # aggregate from hourly to daily
        era5_step = _aggregate_era5(era5_cube[ERA5_AGGREGATION_SUM], "sum")
        era5_step = era5_step.update(
            _aggregate_era5(era5_cube[ERA5_AGGREGATION_ALL], "mean")
        )
        era5_step = era5_step.update(
            _aggregate_era5(era5_cube[ERA5_AGGREGATION_ALL], "min")
        )
        era5_step = era5_step.update(
            _aggregate_era5(era5_cube[ERA5_AGGREGATION_ALL], "max")
        )
        era5_step = era5_step.update(
            _aggregate_era5(era5_cube[ERA5_AGGREGATION_ALL], "median")
        )
        era5_step = era5_step.update(
            _aggregate_era5(era5_cube[ERA5_AGGREGATION_ALL], "std")
        )
        era5_step = era5_step.update(
            _aggregate_era5(era5_cube[ERA5_AGGREGATION_MEAN_MEDIAN_STD], "mean")
        )
        era5_step = era5_step.update(
            _aggregate_era5(era5_cube[ERA5_AGGREGATION_MEAN_MEDIAN_STD], "median")
        )
        era5_step = era5_step.update(
            _aggregate_era5(era5_cube[ERA5_AGGREGATION_MEAN_MEDIAN_STD], "std")
        )
        era5_steps.append(era5_step)
    era5_final = xr.concat(era5_steps, "time_era5")
    era5_final = era5_final.chunk(chunks=dict(time_era5=era5_final.sizes["time_era5"]))
    cube = cube.update(era5_final)

    attrs = {}
    attrs["home_url"] = (
        "https://cds-beta.climate.copernicus.eu/datasets/"
        "reanalysis-era5-single-levels?tab=overview"
    )
    attrs["data_url"] = (
        "https://cds-beta.climate.copernicus.eu/datasets/"
        "reanalysis-era5-single-levels?tab=download"
    )
    attrs["license_url"] = (
        "https://object-store.os-api.cci2.ecmwf.int/cci2-prod-catalogue/"
        "licences/licence-to-use-copernicus-products/licence-to-use-"
        "copernicus-products_b4b9451f54cffa16ecef5c912c9cebd6979925a95"
        "6e3fa677976e0cf198c2c18.pdf"
    )
    cube.attrs["era5_attrs"] = attrs

    return cube


def add_reprojected_dem(super_store: dict, cube: xr.Dataset) -> xr.Dataset:
    dem_files = _get_dem_file_paths(cube.attrs["bbox_wgs84"])
    ds_dem = _load_dem_data(super_store, dem_files)
    dem_reproject = ds_dem.rio.reproject(
        f"EPSG:326{cube.attrs['utm_zone'][:2]}",
        shape=(cube.sizes["y"], cube.sizes["x"]),
        transform=affine.Affine(*cube.attrs["affine_transform"]),
        resampling=rasterio.enums.Resampling.bilinear,
    )
    dem_reproject = dem_reproject.drop_vars("spatial_ref")
    cube["dem"] = dem_reproject["band_1"].chunk(
        dict(
            x=dem_reproject.sizes["x"],
            y=dem_reproject.sizes["y"],
        )
    )

    attrs = dem_reproject.attrs
    attrs["home_url"] = "https://registry.opendata.aws/copernicus-dem/"
    attrs["data_url"] = "https://registry.opendata.aws/copernicus-dem/"
    attrs["license_url"] = (
        "https://spacedata.copernicus.eu/documents/20126/0/"
        "CSCDA_ESA_Mission-specific+Annex.pdf"
    )
    cube.attrs["dem_attrs"] = attrs
    return cube


def _compute_earthnet_cloudmask(super_store: dict, da: xr.DataArray):
    x = torch.from_numpy(da.fillna(1.0).values.astype("float32"))
    b, c, h, w = x.shape

    h_big = (h // 32 + 1) * 32
    h_pad_left = (h_big - h) // 2
    h_pad_right = ((h_big - h) + 1) // 2

    w_big = (w // 32 + 1) * 32
    w_pad_left = (w_big - w) // 2
    w_pad_right = ((w_big - w) + 1) // 2

    x = torch.nn.functional.pad(
        x, (w_pad_left, w_pad_right, h_pad_left, h_pad_right), mode="reflect"
    )
    x = torch.nn.functional.interpolate(
        x, scale_factor=constants.CLOUDMASK_SCALE_FACTOR, mode="bilinear"
    )
    with torch.no_grad():
        y_hat = super_store["cloudmask_model"](x)
    y_hat = torch.argmax(y_hat, dim=1).float()
    y_hat = torch.nn.functional.max_pool2d(y_hat[:, None, ...], kernel_size=2)[
        :, 0, ...
    ]
    y_hat = y_hat[:, h_pad_left:-h_pad_right, w_pad_left:-w_pad_right]

    return xr.DataArray(
        y_hat.cpu().numpy().astype("uint8"),
        dims=("time", "y", "x"),
        coords=dict(time=da.coords["time"], y=da.coords["y"], x=da.coords["x"]),
    )


def _load_dem_data(super_store: dict, dem_files: list[str]) -> xr.Dataset:
    dss = []
    for dem_file in dem_files:
        dss.append(super_store["store_dem"].open_data(dem_file))

    ds = xr.combine_by_coords(dss, combine_attrs="override")
    ds.rio.write_crs(4326, inplace=True)
    return ds


def _get_dem_file_path(lon: float, lat: float) -> str:
    if lon < 0:
        lon_str = f"W{int(abs(lon - 1)):03}"
    else:
        lon_str = f"E{int(abs(lon)):03}"
    if lat < 0:
        lat_str = f"S{int(abs(lat - 1)):02}"
    else:
        lat_str = f"N{int(abs(lat)):02}"
    return (
        f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM/"
        f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM.tif"
    )


def _get_dem_file_paths(bbox: list[float]) -> list[str]:
    points = [
        (bbox[0], bbox[1]),
        (bbox[2], bbox[1]),
        (bbox[2], bbox[3]),
        (bbox[0], bbox[3]),
    ]
    dem_files = []
    for point in points:
        dem_file = _get_dem_file_path(point[0], point[1])
        if dem_file not in dem_files:
            dem_files.append(dem_file)
    return dem_files


def _load_esa_wc_data(super_store: dict, bbox: list[float]) -> (xr.Dataset, list[str]):
    files_2020, files_2021 = _get_esa_wc_file_paths(bbox)
    dss = []
    for files in [files_2020, files_2021]:
        dss_year = []
        for file in files:
            dss_year.append(super_store["store_esa_wc"].open_data(file))
        ds = xr.combine_by_coords(dss_year, combine_attrs="override")
        ds.rio.write_crs(ds.spatial_ref.attrs["wkt"], inplace=True)
        ds = ds.drop_vars("spatial_ref")
        dss.append(ds)
    ds = xr.concat(dss, dim="time", join="exact")
    custom_times = [pd.Timestamp("2020-01-01 00:00"), pd.Timestamp("2021-01-01 00:00")]
    ds = ds.assign_coords(coords=dict(time=custom_times))
    return ds, files_2020 + files_2021


def _get_esa_wc_lon_lat(lon: float, lat: float) -> (str, str):
    if lon < 0:
        lon_mod = int(abs(lon - 1))
        lon_mod += 3 - lon_mod % 3
        lon_str = f"W{lon_mod:03}"
    else:
        lon_mod = int(abs(lon))
        lon_mod -= lon_mod % 3
        lon_str = f"E{lon_mod:03}"
    if lat < 0:
        lat_mod = int(abs(lat - 1))
        lat_mod += 3 - lat_mod % 3
        lat_str = f"S{lat_mod:02}"
    else:
        lat_mod = int(abs(lat))
        lat_mod -= lat_mod % 3
        lat_str = f"N{lat_mod:02}"
    return lat_str, lon_str


def _get_esa_wc_file_paths(bbox: list[float]) -> (list[str], list[str]):
    points = [
        (bbox[0], bbox[1]),
        (bbox[2], bbox[1]),
        (bbox[2], bbox[3]),
        (bbox[0], bbox[3]),
    ]
    files_2020 = []
    files_2021 = []
    for point in points:
        lat_str, lon_str = _get_esa_wc_lon_lat(point[0], point[1])
        file_2020 = (
            f"v100/2020/map/ESA_WorldCover_10m_2020_v100_{lat_str}{lon_str}_Map.tif"
        )
        file_2021 = (
            f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"
        )
        if file_2020 not in files_2020:
            files_2020.append(file_2020)
        if file_2021 not in files_2021:
            files_2021.append(file_2021)
    return files_2020, files_2021


def _aggregate_era5(era5: xr.Dataset, mode: str) -> xr.Dataset:
    era5_mode = getattr(era5.resample(time="1D"), mode)("time")
    rename_dict = {key: f"era5_{key}_{mode}" for key in era5_mode}
    rename_dict["time"] = "time_era5"
    era5_mode = era5_mode.rename(rename_dict)
    return era5_mode

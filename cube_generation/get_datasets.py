import datetime

import affine
import numpy as np
import rasterio
import torch
import xarray as xr
from xcube.core.geom import clip_dataset_by_geometry
from xcube.core.chunk import chunk_dataset

import constants
from version import version


def get_s2l2a(
    super_store: dict, attrs: dict, check_nan: bool = True
) -> xr.Dataset | None:
    data_id = (
        f"cubes/temp/{constants.SCIENCE_FOLDER_NAME}/{version}/{attrs['id']:03}.zarr"
    )
    dss = []
    for year in range(2017, 2025):
        data_id_year = data_id.replace(".zarr", f"_{year}.zarr")
        if not super_store["store_team"].has_data(data_id_year):
            constants.LOG.info(
                f"Dataset with data ID {data_id_year} does not exists. We "
                f"discard the data cube generation for {data_id} for now."
            )
            return None
    for year in range(2017, 2025):
        data_id_year = data_id.replace(".zarr", f"_{year}.zarr")
        ds = super_store["store_team"].open_data(data_id_year)
        if check_nan:
            constants.LOG.info(
                f"Check dataset with data ID {data_id_year} for nan values"
            )
            threshold = 10
            _ = assert_dataset_nan(ds, threshold)
        ds = ds.isel(x=slice(-1000, None), y=slice(-1000, None))
        dss.append(ds)

    # add attributes
    xcube_stac_attrs = {}
    xcube_stac_attrs["source"] = (
        "https://documentation.dataspace.copernicus.eu/APIs/S3.html"
    )
    xcube_stac_attrs["institution"] = "Copernicus Data Space Ecosystem"
    xcube_stac_attrs["standard_name"] = "sentinel2_l2a"
    xcube_stac_attrs["long_name"] = "Sentinel-2 L2A prduct"
    xcube_stac_attrs["stac_catalog_url"] = dss[0].attrs["stac_catalog_url"]
    xcube_stac_attrs["stac_item_ids"] = dss[0].attrs["stac_item_ids"]
    for ds in dss[1:]:
        xcube_stac_attrs["stac_item_ids"].update(ds.attrs["stac_item_ids"])

    # concatenate datasets
    try:
        ds = xr.concat(dss, dim="time", join="exact", combine_attrs="drop")
    except:
        constants.LOG.info(f"Dims are wrong.")
        return None
    ds.attrs = attrs
    ds.attrs["xcube_stac_attrs"] = xcube_stac_attrs
    ds.attrs["affine_transform"] = ds.rio.transform()
    return ds


def get_s2l2a_single_training_year(
    super_store: dict, loc_idx: int, time_idx: int, check_nan: bool = True
) -> xr.Dataset | None:
    data_id = (
        f"cubes/temp/{constants.TRAINING_FOLDER_NAME}/{version}/"
        f"{loc_idx:04}_{time_idx}.zarr"
    )
    if not super_store["store_team"].has_data(data_id):
        constants.LOG.info(
            f"Dataset with data ID {data_id} does not exists. We "
            f"discard the data cube generation for {data_id} for now."
        )
        return None
    ds = super_store["store_team"].open_data(data_id)
    if check_nan:
        constants.LOG.info(f"Check dataset with data ID {data_id} for nan values")
        threshold = 50
        exceeded = assert_dataset_nan(ds, threshold, no_angles=True)
        if exceeded:
            return None
    ds = ds.isel(x=slice(0, 90), y=slice(0, 90))

    # add attributes
    xcube_stac_attrs = {}
    xcube_stac_attrs["source"] = (
        "https://documentation.dataspace.copernicus.eu/APIs/S3.html"
    )
    xcube_stac_attrs["institution"] = "Copernicus Data Space Ecosystem"
    xcube_stac_attrs["standard_name"] = "sentinel2_l2a"
    xcube_stac_attrs["long_name"] = "Sentinel-2 L2A prduct"
    xcube_stac_attrs["stac_catalog_url"] = ds.attrs["stac_catalog_url"]
    xcube_stac_attrs["stac_item_ids"] = ds.attrs["stac_item_ids"]

    ds.attrs = dict()
    ds.attrs["xcube_stac_attrs"] = xcube_stac_attrs
    ds.attrs["affine_transform"] = ds.rio.transform()
    return ds


def reorganize_cube(ds: xr.Dataset) -> xr.Dataset:
    scl = ds.SCL.astype(np.uint8)
    solar_angle = ds.solar_angle.astype(np.float32)
    viewing_angle = ds.viewing_angle.astype(np.float32)
    ds = ds.drop_vars(["SCL", "solar_angle", "viewing_angle"])
    s2l2a = ds.to_dataarray(dim="band").astype(np.float32)
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
            x=constants.CHUNKSIZE_X,
            y=constants.CHUNKSIZE_Y,
        )
    )
    scl = scl.chunk(
        chunks=dict(
            time=constants.CHUNKSIZE_TIME,
            x=constants.CHUNKSIZE_X,
            y=constants.CHUNKSIZE_Y,
        )
    )
    cube = xr.Dataset()
    cube_attrs = ds.attrs
    sen2_attrs = cube_attrs.pop("xcube_stac_attrs")
    cube["s2l2a"] = s2l2a
    cube["s2l2a"].attrs = sen2_attrs
    cube["solar_angle"] = solar_angle
    cube["solar_angle"].attrs = sen2_attrs
    cube["viewing_angle"] = viewing_angle
    cube["viewing_angle"].attrs = sen2_attrs
    cube["scl"] = scl
    sen2_attrs.update(
        dict(
            description="Scene classification layer of the Sentinel-2 L2A product.",
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
    )
    cube["scl"].attrs = sen2_attrs
    cube.attrs = cube_attrs

    return cube


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
            x=constants.CHUNKSIZE_X,
            y=constants.CHUNKSIZE_Y,
        )
    )
    cube["cloud_mask"] = cloud_mask
    cube["cloud_mask"].attrs = dict(
        standard_name="cloudmask",
        long_name=(
            "Cloudmask generated using an AI approach following the implementation "
            "of EarthNet Minicuber, based on CloudSEN12."
        ),
        source="https://github.com/earthnet2021/earthnet-minicuber",
        institution="https://cloudsen12.github.io/",
        flag_values=[0, 1, 2, 3],
        flag_meanings="clear thick_cloud thin_cloud cloud_shadow",
        flag_colors="#000000 #FFFFFF #D3D3D3 #636363",
    )
    return cube


def get_cloudmask(super_store: dict, cube: xr.Dataset) -> xr.Dataset:
    data_id = f"cubes/temp/{constants.SCIENCE_FOLDER_NAME}/{version}/{cube.attrs['id']:03}_cloudmask.zarr"
    if not super_store["store_team"].has_data(data_id):
        constants.LOG.info(
            f"Cloud mask with data ID {data_id} does not exists. We "
            f"discard the data cube generation for now."
        )
        return None
    cube["cloud_mask"] = super_store["store_team"].open_data(data_id)["cloud_mask"]
    return cube


def add_reprojected_lccs(super_store: dict, cube: xr.Dataset) -> xr.Dataset:
    mlds_lc = super_store["store_lccs"].open_data(constants.DATA_ID_LAND_COVER_CLASS)
    lc = mlds_lc.base_dataset

    # clip LC dataset by data cube geometry
    buffer = 0.02
    bbox = cube.attrs["bbox_wgs84"]
    bbox = [bbox[0] - buffer, bbox[1] - buffer, bbox[2] + buffer, bbox[3] + buffer]
    lc = clip_dataset_by_geometry(lc, bbox)

    # clip LC dataset by time
    start = cube.time[0] - np.timedelta64(365, "D")
    end = cube.time[-1] + np.timedelta64(365, "D")
    if start > np.datetime64('2021-01-01T00:00:00'):
        start = np.datetime64('2020-01-01T00:00:00')
        end = np.datetime64('2022-12-31T00:00:00')
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
    lc_reproject = chunk_dataset(
        lc_reproject,
        dict(
            time_lccs=lc_reproject.sizes["time_lccs"],
            x=constants.CHUNKSIZE_X,
            y=constants.CHUNKSIZE_Y,
        ),
        format_name="zarr",
        data_vars_only=True,
    )
    attrs = {}
    attrs["long_name"] = "Land Cover Map of ESA CCI brokered by CDS"
    attrs["source"] = (
        "https://cds.climate.copernicus.eu/datasets/satellite-land-cover?tab=overview"
    )
    attrs["institution"] = "Copernicus Climate Data Store"
    attrs["license"] = (
        "https://object-store.os-api.cci2.ecmwf.int/cci2-prod-catalogue/licences"
        "/satellite-land-cover/satellite-land-cover_8423d13d3dfd95bbeca92d935551"
        "6f21de90d9b40083a915ead15a189d6120fa.pdf"
    )
    attrs["product_version"] = "2.0.7cds"
    name_dict = {
        "change_count": "lccs_change_count",
        "current_pixel_state": "lccs_current_pixel_state",
        "lccs_class": "lccs_class",
        "observation_count": "lccs_observation_count",
        "processed_flag": "lccs_processed_flag",
    }
    for key, val in name_dict.items():
        cube[val] = lc_reproject[key].astype(lc[key].dtype)
        cube[val].attrs = attrs
        if "flag_colors" in lc_reproject[key].attrs:
            cube[val].attrs["flag_colors"] = lc_reproject[key].attrs["flag_colors"]
        if "flag_meanings" in lc_reproject[key].attrs:
            cube[val].attrs["flag_meanings"] = lc_reproject[key].attrs["flag_meanings"]
        if "flag_values" in lc_reproject[key].attrs:
            cube[val].attrs["flag_values"] = lc_reproject[key].attrs["flag_values"]
        if "valid_max" in lc_reproject[key].attrs:
            cube[val].attrs["valid_max"] = lc_reproject[key].attrs["valid_max"]
        if "valid_min" in lc_reproject[key].attrs:
            cube[val].attrs["valid_min"] = lc_reproject[key].attrs["valid_min"]

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
        if counts.size > 0:
            idx = np.argmax(counts)
            lc_second.append(int(vals[idx]))
            lc_second_percentage.append(float(counts[idx] / arr.size))
        else:
            lc_second.append("Only one class found.")
            lc_second_percentage.append("Only one class found.")
    cube.attrs["landcover_first"] = lc_first
    cube.attrs["landcover_first_percentage"] = lc_first_percentage
    cube.attrs["landcover_second"] = lc_second
    cube.attrs["landcover_second_percentage"] = lc_second_percentage
    return cube


def add_reprojected_esa_wc(super_store: dict, cube: xr.Dataset) -> xr.Dataset:
    ds_esa_wc, files = _load_esa_wc_data(super_store, cube.attrs["bbox_wgs84"])

    # clip ESA WC dataset by data cube geometry
    buffer = 0.02
    bbox = cube.attrs["bbox_wgs84"]
    bbox = [bbox[0] - buffer, bbox[1] - buffer, bbox[2] + buffer, bbox[3] + buffer]
    ds_esa_wc = clip_dataset_by_geometry(ds_esa_wc, bbox)

    esa_wc_reproject = ds_esa_wc.rio.reproject(
        f"EPSG:326{cube.attrs['utm_zone'][:2]}",
        shape=(cube.sizes["y"], cube.sizes["x"]),
        transform=affine.Affine(*cube.attrs["affine_transform"]),
        resampling=rasterio.enums.Resampling.nearest,
    )
    cube["esa_wc"] = (
        esa_wc_reproject["band_1"]
        .chunk(
            dict(
                x=constants.CHUNKSIZE_X,
                y=constants.CHUNKSIZE_Y,
            )
        )
        .astype(np.uint8)
    )
    cube["esa_wc"].attrs = dict(
        standard_name="esa_world_cover",
        long_name="ESA World Cover",
        source="https://esa-worldcover.org",
        license="https://creativecommons.org/licenses/by/4.0/",
        institution="VITO Remote Sensing",
        file_names=files,
        flag_values=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
        flag_meanings=(
            "tree_cover shrubland grassland cropland built_up "
            "bare_sparse_vegetation snow_and_ice "
            "permanent_water_bodies herbaceous_wetland "
            "mangroves moss_and_lichen"
        ),
        flag_colors=(
            "#006400 #ffbb22 #ffff4c #f096ff #fa0000 #b4b4b4 "
            "#f0f0f0 #0064c8 #0096a0 #00cf75 #fae6a0"
        ),
    )

    return cube


ERA5_AGGREGATION_ALL = [
    "d2m",
    "skt",
    "sp",
    "stl1",
    "stl2",
    "stl3",
    "stl4",
    "swvl1",
    "swvl2",
    "swvl3",
    "swvl4",
    "t2m",
    "u10",
    "v10",
    "vpd",
    "rh",
    "src",
    "fal",
]
ERA5_AGGREGATION_MEAN_MEDIAN_STD = ["lai_hv", "lai_lv"]
ERA5_AGGREGATION_SUM = [
    "e",
    "pev",
    "slhf",
    "sshf",
    "ssr",
    "ssrd",
    "str",
    "strd",
    "tp",
    "ro",
]
# ref Magnus Formula for vapor pressure (Gleichung 6)
# https://journals.ametsoc.org/view/journals/bams/86/2/bams-86-2-225.xml
MF_A = 17.625
MF_B = 243.04
MF_C = 610.94


def add_era5(super_store: dict, cube: xr.Dataset, sel_time=False) -> xr.Dataset:
    data_id = "cubes/aux/era5_land_time_optimized.zarr"
    era5 = super_store["store_team"].open_data(data_id)
    if sel_time:
        era5 = era5.sel(time=slice(
            np.datetime64(cube.time[0].values).astype('M8[D]'),
            np.datetime64(cube.time[-1].values).astype('M8[D]')  + 1
            )
        )
    es = MF_C * np.exp((MF_A * era5["t2m"]) / (era5["t2m"] + MF_B))
    e = MF_C * np.exp((MF_A * era5["d2m"]) / (era5["d2m"] + MF_B))
    era5["rh"] = (e / es) * 100
    era5["rh"].attrs = dict(
        standard_name="relative_humidity",
        long_name="Relative humidity (computed)",
        units="%",
    )
    era5["vpd"] = es - e
    era5["vpd"].attrs = dict(
        standard_name="vapour_pressure_deficit",
        long_name="Vapour pressure deficit (computed)",
        units="Pa",
    )

    era5_cube = era5.interp(
        lat=cube.attrs["center_wgs84"][0],
        lon=cube.attrs["center_wgs84"][1],
        method="linear",
    )
    if np.all(np.isnan(era5_cube.d2m.values)):
        # Find closest non-nan value
        df = era5.isel(time=0)[['lat', 'lon', 'd2m']].to_dataframe().reset_index()
        df = df.dropna(subset=['d2m'])
        df['distance'] = np.sqrt(
            (df['lat'] - cube.attrs["center_wgs84"][0])**2 +
            (df['lon'] - cube.attrs["center_wgs84"][1])**2
        )
        nearest_valid = df.loc[df['distance'].idxmin()]
        constants.LOG.info(
            f"ERA5-Land has nan values; closest non-NaN values is "
            f"taken. Distance to center: {nearest_valid["distance"]:.4}°"
        )
        era5_cube = era5.sel(
            lat=nearest_valid["lat"],
            lon=nearest_valid["lon"],
        )
    era5_cube = era5_cube.drop(["lat", "lon"])

    # aggregate from hourly to daily
    era5_final = _aggregate_era5(era5_cube[ERA5_AGGREGATION_SUM], "sum")
    era5_final.update(
        _aggregate_era5(era5_cube[ERA5_AGGREGATION_ALL], "mean")
    )
    era5_final.update(
        _aggregate_era5(era5_cube[ERA5_AGGREGATION_ALL], "min")
    )
    era5_final.update(
        _aggregate_era5(era5_cube[ERA5_AGGREGATION_ALL], "max")
    )
    era5_final.update(
        _aggregate_era5(era5_cube[ERA5_AGGREGATION_ALL], "median")
    )
    era5_final.update(
        _aggregate_era5(era5_cube[ERA5_AGGREGATION_ALL], "std")
    )
    era5_final.update(
        _aggregate_era5(era5_cube[ERA5_AGGREGATION_MEAN_MEDIAN_STD], "mean")
    )
    era5_final.update(
        _aggregate_era5(era5_cube[ERA5_AGGREGATION_MEAN_MEDIAN_STD], "median")
    )
    era5_final.update(
        _aggregate_era5(era5_cube[ERA5_AGGREGATION_MEAN_MEDIAN_STD], "std")
    )
    era5_final = era5_final.chunk(chunks=dict(time_era5=era5_final.sizes["time_era5"]))
    era5_final = era5_final.astype(np.float32)

    cube.update(era5_final)

    return cube


def add_reprojected_dem(super_store: dict, cube: xr.Dataset) -> xr.Dataset:
    bbox = [
        cube.attrs["bbox_wgs84"][0] - 0.05,
        cube.attrs["bbox_wgs84"][1] - 0.05,
        cube.attrs["bbox_wgs84"][2] + 0.05,
        cube.attrs["bbox_wgs84"][3] + 0.05,
    ]
    dem_files = _get_dem_file_paths(bbox)
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
            x=constants.CHUNKSIZE_X,
            y=constants.CHUNKSIZE_Y,
        )
    )
    cube["dem"].attrs = dict(
        standard_name="copernicus_dem_30m",
        long_name="Copernicus Digital Elevation Model (DEM) at 30m",
        units="m",
        institution="Synergise",
        source="https://registry.opendata.aws/copernicus-dem/",
        license=(
            "https://spacedata.copernicus.eu/documents/20126/0/"
            "CSCDA_ESA_Mission-specific+Annex.pdf"
        ),
    )
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
        dss.append(super_store["store_dem"].open_data(dem_file, opener_id="dataset:geotiff:s3"))

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
    files_2021 = _get_esa_wc_file_paths(bbox)
    dss = []
    for file in files_2021:
        dss.append(super_store["store_esa_wc"].open_data(file, opener_id="dataset:geotiff:s3"))
    ds = xr.combine_by_coords(dss, combine_attrs="override")
    ds.attrs["date"] = "2021-01-01"
    return ds, files_2021


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


def _get_esa_wc_file_paths(bbox: list[float]) -> list[str]:
    points = [
        (bbox[0], bbox[1]),
        (bbox[2], bbox[1]),
        (bbox[2], bbox[3]),
        (bbox[0], bbox[3]),
    ]
    files_2021 = []
    for point in points:
        lat_str, lon_str = _get_esa_wc_lon_lat(point[0], point[1])
        file_2021 = (
            f"v200/2021/map/ESA_WorldCover_10m_2021_v200_{lat_str}{lon_str}_Map.tif"
        )
        if file_2021 not in files_2021:
            files_2021.append(file_2021)
    return files_2021


def _aggregate_era5(era5: xr.Dataset, mode: str) -> xr.Dataset:
    era5_mode = getattr(era5.compute().resample(time="1D"), mode)()
    rename_dict = {key: f"era5_{key}_{mode}" for key in era5_mode}
    rename_dict["time"] = "time_era5"
    era5_mode = era5_mode.rename(rename_dict)
    for key in era5.data_vars:
        attrs = dict(
            standard_name=era5[key].attrs["standard_name"],
            long_name=era5[key].attrs["long_name"],
            units=era5[key].attrs["units"],
            institution="Copernicus Climate Data Store",
            source=(
                "https://cds.climate.copernicus.eu/datasets/"
                "reanalysis-era5-land?tab=overview"
            ),
            license=(
                "https://object-store.os-api.cci2.ecmwf.int/cci2-prod-catalogue/"
                "licences/licence-to-use-copernicus-products/licence-to-use-"
                "copernicus-products_b4b9451f54cffa16ecef5c912c9cebd6979925a95"
                "6e3fa677976e0cf198c2c18.pdf"
            ),
        )
        era5_mode[f"era5_{key}_{mode}"].attrs = attrs
    return era5_mode


def assert_dataset_nan(ds: xr.Dataset, threshold: float | int, no_angles: bool = False) -> bool:
    exceeded = False
    for key in list(ds.keys()):
        if no_angles and key in ["solar_angle", "viewing_angle"]:
            continue
        array = ds[key].values.ravel()
        null_size = array[np.isnan(array)].size
        perc = (null_size / array.size) * 100
        if perc > threshold:
            constants.LOG.info(f"Data variable {key} has {perc:.3f}% nan values.")
            exceeded = True
            break
    return exceeded

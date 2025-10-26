import pickle
import spyndex
import numpy as np
import xarray as xr
from typing import Dict, Union, List


def normalize(ds: Union[xr.Dataset, Dict[str, np.ndarray]], range_dict: Dict[str, List[float]], filter_var: str = None) -> Union[xr.Dataset, Dict[str, np.ndarray]]:
    """
    Normalize all variables in the dataset or dictionary using the provided range dictionary.

    Args:
        ds (Union[xr.Dataset, Dict[str, np.ndarray]]): The xarray dataset  or dictionary to normalize.
        range_dict (Dict[str, List[float]]): Dictionary with min and max values for each variable.
        filter_var (str): Variable name to exclude from normalization, such as a mask variable (e.g., 'land_mask').

    Returns:
        Union[xr.Dataset, Dict[str, np.ndarray]]: The normalized dataset or dictionary.
    """
    normalized_ds = ds.copy()
    data_vars = normalized_ds.data_vars if isinstance(normalized_ds, xr.Dataset) else normalized_ds.keys()
    for var in data_vars:
        if var == 'split' or var == filter_var: continue
        if var in range_dict:
            xmin, xmax = range_dict[var]
            if xmax != xmin:
                normalized_data = (ds[var] - xmin) / (xmax - xmin)
                if isinstance(normalized_ds, xr.Dataset):
                    normalized_ds[var] = normalized_data
                else:
                    normalized_ds[var] = normalized_data
            else:
                normalized_ds[var] = ds[var]  # If xmin == xmax, normalization isn't possible
    return normalized_ds


def normalize_dataarray(da: xr.DataArray, range_dict: Dict[str, List[float]]) -> xr.DataArray:
    """
    Normalize a DataArray with 'index' dimension using per-variable min-max ranges.

    Args:
        da (xr.DataArray): DataArray with shape (index, time, y, x)
        range_dict (Dict[str, List[float]]): Dictionary with min/max values per index label

    Returns:
        xr.DataArray: Normalized DataArray
    """
    index_names = da.coords["index"].values
    normalized_slices = []

    for i, var_name in enumerate(index_names):
        if var_name in range_dict:
            xmin, xmax = range_dict[var_name]
            slice_i = da.isel(index=i)
            if xmax != xmin:
                normalized = (slice_i - xmin) / (xmax - xmin)
            else:
                normalized = slice_i
        else:
            normalized = da.isel(index=i)  # untouched if no range

        normalized_slices.append(normalized.expand_dims(index=[var_name]))

    return xr.concat(normalized_slices, dim="index")




def prepare_spectral_data(da: xr.DataArray, min_max_dict=None, to_ds = True, compute_SI = False, load_b01b09=False) -> xr.DataArray:
    """
    Prepares an input DataArray with shape (band, time, y, x) by computing selected spectral indices
    and stacking the result into a single DataArray with shape (index, time, y, x).

    Args:
        da (xr.DataArray): Input cube with dimensions (band, time, y, x)
        min_max_dict (dict): Dictionary of min/max values for normalization

    Returns:
        xr.DataArray: Output cube with dimensions (index, time, y, x)
    """
    da = da.clip(min=0, max=1)
    if not load_b01b09:
        da = da.sel(band=[b for b in da.band.values if b not in ["B01", "B09"]])
    bands = da['band'].values

    if min_max_dict is None:
        try:
            with open("../all_ranges_no_clouds.pkl", "rb") as f:
                min_max_dict = pickle.load(f)
        except:
            with open("./all_ranges_no_clouds.pkl", "rb") as f:
                min_max_dict = pickle.load(f)

    # === Step 1: Create a Dataset with bands as variables ===
    all_bands = xr.Dataset(
        {band: da.sel(band=band).drop_vars('band') for band in bands},
        coords={dim: da.coords[dim] for dim in ['time', 'y', 'x']}
    )

    # === Step 2: Select indices to compute ===

    if compute_SI:

        all_indices = [idx for idx, attrs in spyndex.indices.items() if ("Sentinel-2" in attrs.platforms)]

        all_indices.remove("NIRvP")
        all_indices.remove("RI4XS")
        all_indices.remove("VIBI")
        all_indices.remove("MTCI")
        all_indices.remove("OCVI")
        all_indices.remove("NDDI")
        all_indices.remove("CVI")
        all_indices.remove("S2REP")
        all_indices.remove("TCARIOSAVI")
        all_indices.remove("TCARIOSAVI705")
        all_indices.remove("MCARIOSAVI705")
        all_indices.remove("MCARIOSAVI")
        all_indices.remove("SIPI")
        all_indices.remove("REDSI")
        all_indices.remove("kEVI")
        all_indices.remove("IBI")
        all_indices.remove("MCARI")
        all_indices.remove("SAVI2")
        all_indices.remove("SR")
        all_indices.remove("DSWI4")
        all_indices.remove("ARI2")
        all_indices.remove("NMDI")
        all_indices.remove("GRVI")
        all_indices.remove("CIG")
        all_indices.remove("SR2")
        all_indices.remove("RVI")
        all_indices.remove("SR555")
        all_indices.remove("GM1")
        all_indices.remove("EVI")
        all_indices.remove("TWI")
        all_indices.remove("TCARI")
        all_indices.remove("EVIv")
        all_indices.remove("GEMI")
        all_indices.remove("NBSIMS")
        all_indices.remove("DSWI2")
        all_indices.remove("VARI700")
        all_indices.remove("DSWI3")
        all_indices.remove("OSI")
        all_indices.remove("BAI")
        all_indices.remove("IAVI")
        all_indices.remove("RGRI")
        all_indices.remove("SEVI")
        all_indices.remove("VARI")
        all_indices.remove("ARI")
        all_indices.remove("SWM")
        all_indices.remove("GARI")
        all_indices.remove("mND705")
        all_indices.remove("DSI")
        all_indices.remove("MSI")
        all_indices.remove("WRI")
        all_indices.remove("SR3")
        all_indices.remove("DSWI5")
        all_indices.remove("PSRI")
        all_indices.remove("DSWI1")
        all_indices.remove("TRRVI")
        all_indices.remove("BRBA")
        all_indices.remove("NSDSI1")
        all_indices.remove("MCARI705")
        all_indices.remove("IRECI")
        all_indices.remove("CSI")
        all_indices.remove("EBI")
        all_indices.remove("SLAVI")
        all_indices.remove("CIRE")
        all_indices.remove("GM2")
        all_indices.remove("SR705")
        all_indices.remove("NSDSI2")
        all_indices.remove("MSR") #
        all_indices.remove("TCI")
        all_indices.remove("BAIS2") #
        all_indices.remove("BAIM")
        all_indices.remove("NDSInw")
        all_indices.remove("MSR705")
        all_indices.remove("TGI")

        print(all_indices)


        # === Step 3: Compute spectral indices ===
        index_result = spyndex.computeIndex(

            # Indices to compute
            all_indices,

            # Bands
            A=da.sel(band="B01"),
            B=da.sel(band="B02"),
            G=da.sel(band="B03"),
            R=da.sel(band="B04"),
            RE1=da.sel(band="B05"),
            RE2=da.sel(band="B06"),
            RE3=da.sel(band="B07"),
            N=da.sel(band="B08"),
            N2=da.sel(band="B8A"),
            S1=da.sel(band="B11"),
            S2=da.sel(band="B12"),

            # Kernel indices
            kNN=1.0,
            kNR=spyndex.computeKernel(
                kernel="RBF",
                a=da.sel(band="B08"),
                b=da.sel(band="B04"),
                sigma=(da.sel(band="B08") + da.sel(band="B04")) / 2
            ),
            kNB=spyndex.computeKernel(
                kernel="RBF",
                a=da.sel(band="B08"),
                b=da.sel(band="B02"),
                sigma=(da.sel(band="B08") + da.sel(band="B02")) / 2
            ),
            kNL=spyndex.computeKernel(
                kernel="RBF",
                a=da.sel(band="B08"),
                b=spyndex.constants.L.default,
                sigma=(da.sel(band="B08") + spyndex.constants.L.default) / 2
            ),
            kGG=1.0,
            kGR=spyndex.computeKernel(
                kernel="RBF",
                a=da.sel(band="B03"),
                b=da.sel(band="B04"),
                sigma=(da.sel(band="B03") + da.sel(band="B04")) / 2
            ),
            kGB=spyndex.computeKernel(
                kernel="RBF",
                a=da.sel(band="B03"),
                b=da.sel(band="B02"),
                sigma=(da.sel(band="B03") + da.sel(band="B02")) / 2
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

    # === Step 4: Merge bands and indices ===
    # index_result has dimensions (index, time, y, x), we now add bands
    bands_da = xr.concat([all_bands[var] for var in all_bands.data_vars], dim="index")
    bands_da = bands_da.assign_coords(index=("index", list(all_bands.data_vars.keys())))
    if compute_SI:
        index_result = normalize_dataarray(index_result, min_max_dict)
        index_result = index_result.clip(0, 1)

        # Normalize both bands and indices (together)
        full_stack = xr.concat([bands_da, index_result], dim="index")
    else:
        full_stack = bands_da
    if to_ds:
        index_values = full_stack.index.values
        data_vars = {str(idx): full_stack.sel(index=idx).drop_vars('index') for idx in index_values}
        full_stack = xr.Dataset(data_vars)  # [['EVI']]
        full_stack = normalize(full_stack, min_max_dict)

        full_stack = full_stack.map(lambda da: da.clip(0, 1))
    else:
        full_stack = normalize_dataarray(full_stack, min_max_dict)
        full_stack = full_stack.clip(0, 1)

    return full_stack  # shape: (index, time, y, x)

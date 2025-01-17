import pickle
import spyndex
import xarray as xr
from ml4xcube.preprocessing import replace_inf_with_nan


def prepare_cube(da):
    # Assuming `da` is your DataArray with dimensions (band, time, y, x)
    bands = da['band'].values  # Extract band names (e.g., ['B01', 'B02', ...])

    # Create a Dataset with each band as a separate variable
    all_bands = xr.Dataset(
        {band: da.sel(band=band).drop_vars('band') for band in bands},  # Add variables for each band
        coords={
            'time': da.coords['time'],  # Preserve time coordinates
            'x': da.coords['x'],        # Preserve x coordinates
            'y': da.coords['y']         # Preserve y coordinates
        }
    )


    all_indices = [idx for idx, attrs in spyndex.indices.items() if ("Sentinel-2" in attrs.platforms)]

    all_indices.remove("NIRvP")

    all_indices = spyndex.computeIndex(

        # Indices to compute
        all_indices,

        # Bands
        A   = da.sel(band="B01"),
        B   = da.sel(band="B02"),
        G   = da.sel(band="B03"),
        R   = da.sel(band="B04"),
        RE1 = da.sel(band="B05"),
        RE2 = da.sel(band="B06"),
        RE3 = da.sel(band="B07"),
        N   = da.sel(band="B08"),
        N2  = da.sel(band="B8A"),
        S1  = da.sel(band="B11"),
        S2  = da.sel(band="B12"),

        # Kernel indices
        kNN = 1.0,
        kNR = spyndex.computeKernel(
            kernel = "RBF",
            a      = da.sel(band="B08"),
            b      = da.sel(band="B04"),
            sigma  = (da.sel(band="B08") + da.sel(band="B04")) / 2
        ),
        kNB = spyndex.computeKernel(
            kernel = "RBF",
            a      = da.sel(band="B08"),
            b      = da.sel(band="B02"),
            sigma  = (da.sel(band="B08") + da.sel(band="B02")) / 2
        ),
        kNL = spyndex.computeKernel(
            kernel = "RBF",
            a      = da.sel(band="B08"),
            b      = spyndex.constants.L.default,
            sigma  = (da.sel(band="B08") + spyndex.constants.L.default) / 2
        ),
        kGG = 1.0,
        kGR = spyndex.computeKernel(
            kernel = "RBF",
            a      = da.sel(band="B03"),
            b      = da.sel(band="B04"),
            sigma  = (da.sel(band="B03") + da.sel(band="B04")) / 2
        ),
        kGB=  spyndex.computeKernel(
            kernel = "RBF",
            a      = da.sel(band="B03"),
            b      = da.sel(band="B02"),
            sigma  = (da.sel(band="B03") + da.sel(band="B02")) / 2
        ),

        # Additional parameters
        L       = spyndex.constants.L.default,
        C1      = spyndex.constants.C1.default,
        C2      = spyndex.constants.C2.default,
        g       = spyndex.constants.g.default,
        gamma   = spyndex.constants.gamma.default,
        alpha   = spyndex.constants.alpha.default,
        sla     = spyndex.constants.sla.default,
        slb     = spyndex.constants.slb.default,
        nexp    = spyndex.constants.nexp.default,
        cexp    = spyndex.constants.cexp.default,
        k       = spyndex.constants.k.default,
        fdelta  = spyndex.constants.fdelta.default,
        epsilon = spyndex.constants.epsilon.default,
        omega   = spyndex.constants.omega.default,
        beta    = spyndex.constants.beta.default,

            # Wavelength parameters
        lambdaN  = spyndex.bands.N.modis.wavelength,
        lambdaG  = spyndex.bands.G.modis.wavelength,
        lambdaR  = spyndex.bands.R.modis.wavelength,
        lambdaS1 = spyndex.bands.S1.modis.wavelength,
    )


    # when using ml4xcube
    index_values = all_indices.index.values
    data_vars = {str(idx): all_indices.sel(index=idx).drop_vars('index') for idx in index_values}
    ds = xr.Dataset(data_vars)#[['EVI']]

    #print(ds)
    # Append variables from all_bands to ds
    ds = xr.merge([all_bands, ds])

    #print(ds)
    #ds = replace_inf_with_nan(ds)

    return ds

import spyndex
import xarray as xr

da = xr.open_zarr('/net/scratch/mreinhardt/testcube.zarr')["s2l2a"]
# print(da['band']) # ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A']

all_indices = [idx for idx, attrs in spyndex.indices.items() if ("Sentinel-2" in attrs.platforms)]

#all_indices.remove("CCI")
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
selected_index = all_indices
#selected_index = all_indices.isel(time=slice(0,1), x=slice(0, 10), y=slice(0, 10))
#selected_index = all_indices.sel(time="2016-11-01").isel(x=slice(0, 10), y=slice(0, 10))
#print(selected_index.chunks)
#print(selected_index.compute().shape)

index_values = all_indices.index.values
data_vars = {str(idx): all_indices.sel(index=idx).drop_vars('index') for idx in index_values}
ds = xr.Dataset(data_vars)
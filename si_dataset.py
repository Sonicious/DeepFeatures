import pickle
import spyndex
import xarray as xr
from ml4xcube.preprocessing import replace_inf_with_nan, get_range, standardize, get_median, get_statistics

da = xr.open_zarr('/net/scratch/mreinhardt/testcube.zarr')["s2l2a"]
#print(da['band']) # ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12', 'B8A']


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

# Drop any coordinates not in ['x', 'y', 'time']
da_bands = da.drop([coord for coord in da.coords if coord not in ['x', 'y', 'time']])


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

#selected_index = all_indices.sel(index=["NDVI"], time="2016-11-01", x=(596726., 596746.), y=(5664648., 5664628.)).compute()

#print(selected_index.shape)

"""selected_times = '2016-11-01'
selected_x = [596726., 596746., 596766., 596786., 596806., 596826., 596846., 596866., 596886., 596906., 596926., 596946., 596966., 596986., 597006.]
selected_y = [5664648., 5664628., 5664608., 5664588., 5664568., 5664548., 5664528., 5664508., 5664488., 5664468., 5664448., 5664428., 5664408., 5664388., 5664368.,]

subcube = all_indices.sel(
    index = 'NDVI',
    time=selected_times,
    x=selected_x,
    y=selected_y
)
print(subcube.compute())

print('===============================')
print('===============================')
print('===============================')
print('===============================')
"""
#print(selected_index)
#selected_index = all_indices.isel(time=slice(0,1), x=slice(0, 10), y=slice(0, 10))
#selected_index = all_indices.sel(time="2016-11-01").isel(x=slice(0, 10), y=slice(0, 10))
#print(selected_index.chunks)
#print(selected_index.compute().shape)

# when using ml4xcube
index_values = all_indices.index.values
data_vars = {str(idx): all_indices.sel(index=idx).drop_vars('index') for idx in index_values}
ds = xr.Dataset(data_vars)#[['EVI']]
print(ds)
# Append variables from all_bands to ds
for var in all_bands.data_vars:
    ds[var] = all_bands[var]

print(ds)
ds = replace_inf_with_nan(ds)
"""ds = replace_inf_with_nan(ds)
#

stats = get_statistics(ds)
with open('stats.pkl', "wb") as f:
    pickle.dump(stats, f)
#
print('standardize')
std_ds = standardize(ds, stats)
#
print('get_ranges')
ranges = get_range(std_ds)
#
print('save ranges')
with open("ranges_standardized.pkl", "wb") as file:
    pickle.dump(ranges, file)"""

"""print('compute median')
med_dict = get_median(std_ds)
with open('median.pkl', "wb") as f:
    pickle.dump(med_dict, f)"""
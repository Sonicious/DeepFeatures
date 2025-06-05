# DeepFeatures
Repository of the ESA project DeepFeatures

## Installation

```bash
# add date to the environment for organisation issues
# conflicting zarr versions: https://github.com/xcube-dev/xcube/issues/1102
conda create -n deepfeatures_YYYYMMDD -c conda-forge --override-channels adlfs boto3 botocore conda-lock cubo dask h5netcdf importlib_metadata ipykernel lightning matplotlib ml4xcube netcdf4 pandas pip pyproj python pytorch seaborn scikit-image scikit-learn scipy sen2nbar shapely spyndex torchaudio torchvision xarray 'xcube>=1.8.1' zappend 'zarr<3' xcube-sh xcube-cds xcube-stac
conda activate deepfeatures_YYYYMMDD
pip install global-land-mask

conda env export --no-build > environment.yml

python -Xfrozen_modules=off -m ipykernel install --user --name "DeepFeatures" --display-name "DeepFeatures Kernel"
```

For conda-store in deepfeatures: change defaults to main, remove prefix, build kernel and save the final environment from there for reproduction

```bash
conda remove -n deepfeatures --all
```

## Cube generation for DeepFeatures

The directory `cube_generation` contains the pipline to generate data cubes for the
DeepFeatures project. The directory `docs` contains scientific literature for the
BRDF correction and cloud masking.

To run the cube generation, create a conda environment by running 

```bash
conda env create -f cube_generation/environment.yml
conda activate cubegen
```

### Preparation

The following datasets are required during the cube generation and were prepared.

#### Meta-data of the ScienceCubes
The locations of the data cubes are defined by the respective science cases. The location
and further meta-data are gathered in `cube_generation/sites_scinece.csv`.

#### Meta-data of the TrainCubes
The block-sampling of the TrainCubes is performed in `cube_generation/00_generate_training_sites_table.py`
and saved to `cube_generation/sites_training.csv`.

#### Land cover classification
The land cover classification is retrieved from the
[Copernicus Data Store](https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-land-cover?tab=overview)
using the notebook `cube_generation/preparation/get_landcover_classification.ipynb`. 
The data cube is stored to the public DeepESDL bucket `"deep-esdl-public"`.

#### Load ERA5 data
The ERA5 data is loaded for central Europe using the notebook
`cube_generation/preparation/get_era5_central_europe.ipynb`. The cube is saved to
the AWS S3 team bucket with the data ID `"cubes/aux/era5.zarr"`.

### Main Pipeline

The Sentinel-2 data is retrieved on a Creodias VM using the script 
`cube_generation/01_get_sen2l2a_creodias_vm.py`. This script retrieves Sentinel-2 data
using [xcube-stac](https://github.com/xcube-dev/xcube-stac), which uses the CDSE STAC API
to find the tiles and access the EO data from the CDSE S3 bucket. 

The `cube_generation/02_build_training_cubes.py` is used to build the final TrainCube,
where the cloud mask is added to the respective TrainCube. 

The `cube_generation/02_build_science_cubes.py` is used to build the final SienceCube,
where the cloud mask, DEM, CCI Land Cover, and ERA-5 data is added to the respective
ScienceCube. 

The script `main.py` generates the cubes in a loop. The indices are defined in
constants. 

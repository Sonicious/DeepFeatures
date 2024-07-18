# DeepFeatures
Repository of the ESA project DeepFeatures

## Installation

```bash
conda create -n deepfeatures python pip
conda activate deepfeatures
conda install -n deepfeatures -c conda-forge -c pytorch -c nvidia cubo xarray xcube xcube-sh spyndex importlib_metadata ipykernel matplotlib dask sen2nbar scipy scikit-learn netcdf4 h5netcdf scikit-image pandas zarr zappend pyproj shapely pytorch lightning torchvision torchaudio pytorch-cuda=11.8
pip install global-land-mask

conda env export --no-build > environment.yml

python -Xfrozen_modules=off -m ipykernel install --user --name "DeepFeatures" --display-name "DeepFeatures Kernel"
```

For conda-store in deepfeatures: change defaults to main, build kernel and save the final environment from there for reproduction

Optionally add a tag for description of the environment

```bash
conda remove -n deepfeatures --all
```
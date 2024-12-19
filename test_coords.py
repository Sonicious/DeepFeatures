import torch
import numpy as np
from ml4xcube.utils import get_chunk_by_index, split_chunk
from ml4xcube.preprocessing import drop_nan_values, fill_nan_values
from ml4xcube.splits import assign_block_split
from si_dataset import ds

class XrDataset:
    def __init__(self, data_cube, split_val=0.0, sample_size=None):
        self.data_cube = data_cube
        self.split_val = split_val
        self.sample_size = sample_size

    def load_chunk(self, chunk_idx):
        print(chunk_idx)

        chunk, coords = get_chunk_by_index(self.data_cube, chunk_idx)
        #mask = chunk['split'] == self.split_val

        #chunk = {var: np.ma.masked_where(~mask, chunk[var]).filled(np.nan) for var in chunk if var != 'split'}
        chunk, coords = split_chunk(chunk, coords, sample_size=self.sample_size, overlap=None)

        vars = list(chunk.keys())
        chunk, coords = drop_nan_values(chunk, coords, mode='if_all_nan', vars=vars)
        #chunk = fill_nan_values(chunk, method='sample_mean')
        return chunk, coords

def main():
    # Load data cube and assign splits
    xds = ds
    data_cube = assign_block_split(
        ds=xds,
        block_size=[("time", 11), ("y", 150), ("x", 150)],
        split=0.8
    )[['NDVI']]

    # Initialize dataset
    data_iterator = XrDataset(data_cube, sample_size=[("time", 11), ("y", 15), ("x", 15)])

    # Load a specific chunk and select the first sample
    chunk_idx = 0  # Specify the chunk index to load
    chunk, coords = data_iterator.load_chunk(chunk_idx)

    selected_times = coords['time'][0]
    selected_x = coords['x'][0]
    selected_y = coords['y'][0]
    #print(coords['time'][0])
    print(coords['x'][0])
    print(coords['y'][0])

    subcube = xds.sel(
        time=selected_times[0],
        x=selected_x,
        y=selected_y
    )['NDVI'].compute()

    print(subcube)


    print('==============================================')
    print('==============================================')
    print('==============================================')
    print('==============================================')

    #BNDVI
    first_sample = chunk['NDVI'][0][0]
    print(first_sample)

    # Print details of the first sample
    #print("First sample:")
    #for key, value in first_sample.items():
    #    print(f"{key}: {value}")

if __name__ == "__main__":
    main()

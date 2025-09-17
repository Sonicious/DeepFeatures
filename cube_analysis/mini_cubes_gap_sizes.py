import os
import xarray as xr
from utils.utils import compute_time_gaps


def process_cube(cube_num):
    cube_path = os.path.join(base_path, cube_num + '.zarr')
    print(cube_path)

    da = xr.open_zarr(cube_path)
    #ds = prepare_cube(da)
    #chunk, coords = get_chunk_by_index(ds, 0)
    #print(coords['time'])
    time_gaps = compute_time_gaps(da["time"].values)
    print(time_gaps.tolist())

# Call the method
numbers = list(range(500))

# Iterate through the selected numbers and create 6-digit strings
#six_digit_strings = [f"{num:06d}" for num in selected_numbers]
six_digit_strings = [f"{num:06d}" for num in numbers]

base_path = '/net/data_ssd/deepfeatures/trainingcubes'
base_path = '/net/data_ssd/deepfeatures/sciencecubes_processed'

for number in six_digit_strings:
    path = os.path.join(base_path, number)
    process_cube(path)











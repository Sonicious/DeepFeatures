import torch
import random
import numpy as np
from typing import Dict, List, Tuple


def drop_if_central_point_nan_or_inf(
        ds: Dict[str, np.ndarray],
        coords: Dict[str, np.ndarray],
        vars: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Drop samples entirely if the central point of any variable contains NaN or inf values.

    Args:
        ds (Dict[str, np.ndarray]): The dataset to filter. It should be a dictionary where keys are variable names and values are numpy arrays.
        coords (Dict[str, np.ndarray]): The coordinates associated with the dataset.
        vars (List[str]): The variables to check for NaN or inf values at their central point.

    Returns:
        Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]: The filtered dataset and coordinates.
    """
    valid_mask_lists = []

    for var in vars:
        if var not in ds:
            continue
        value = ds[var]

        # Determine the central point index
        central_index = tuple(dim_size // 2 for dim_size in value.shape[1:])

        if value.ndim == 1:  # For 1D arrays, the central point is the single value
            valid_mask = ~np.isnan(value) & ~np.isinf(value)
        elif value.ndim >= 2:  # For multi-dimensional arrays
            central_values = value[(slice(None),) + central_index]  # Extract central point along non-batch axes
            valid_mask = ~np.isnan(central_values) & ~np.isinf(central_values)
        else:
            raise ValueError("Unsupported number of dimensions for the variable.")

        valid_mask_lists.append(valid_mask)

    # Combine masks across all variables using logical AND
    valid_mask = np.all(valid_mask_lists, axis=0)

    # Filter the dataset and coordinates
    ds = {key: ds[key][valid_mask] for key in ds.keys()}
    coords = {key: coords[key][valid_mask] for key in coords.keys()}

    return ds, coords


def concatenate(chunks, coords, masks) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Concatenate the chunks along the time dimension.

    Returns:
        Tuple:
            - Stacked data array
            - Concatenated coordinates
            - Stacked masks array
    """
    # Preallocate memory-efficient structures
    chunk_keys = [key for key in chunks[0] if key != 'split']
    coord_keys = list(coords[0].keys())

    # Concatenate data chunks directly into arrays
    stacked_data = np.concatenate(
        [np.stack([chunk[key] for key in chunk_keys], axis=-1) for chunk in chunks],
        axis=0
    )

    print(stacked_data.shape)

    # Concatenate masks directly into arrays
    stacked_masks = np.concatenate(
        [np.stack([mask[key] for key in chunk_keys], axis=-1) for mask in masks],
        axis=0
    )

    # Normalize and concatenate coordinates
    concatenated_coords = {}
    for key in coord_keys:
        # Filter out 1D arrays
        filtered_coords = [coord[key] for coord in coords if len(coord[key].shape) > 1]

        # Check if there are valid arrays left after filtering
        if not filtered_coords:
            raise ValueError(f"No valid coordinate arrays for key '{key}' after filtering.")

        # Determine the target shape
        max_dim = max(len(arr.shape) for arr in filtered_coords)
        target_shape = (0,) + tuple(
            max(arr.shape[i] for arr in filtered_coords if arr.size > 0) for i in range(1, max_dim)
        )

        # Normalize arrays
        normalized_arrays = [
            np.empty(target_shape, dtype=arr.dtype) if arr.size == 0 else arr
            for arr in filtered_coords
        ]

        # Concatenate along the 0th axis
        concatenated_coords[key] = np.concatenate(normalized_arrays, axis=0)

    # Shuffle in-place to save memory
    idx = np.random.permutation(stacked_data.shape[0])
    stacked_data = stacked_data[idx]  # Shuffle data
    stacked_masks = stacked_masks[idx]  # Shuffle masks
    for key in concatenated_coords:
        concatenated_coords[key] = concatenated_coords[key][idx]  # Shuffle coordinates

    return stacked_data, concatenated_coords, stacked_masks


def compute_time_gaps(time_coords):
    """
    Helper method to compute time gaps for a given set of time coordinates.

    Args:
        time_coords (np.ndarray): Array of time coordinates.

    Returns:
        torch.Tensor: Tensor of time gaps between consecutive timestamps.
    """
    if len(time_coords) > 1:
        time_deltas = np.diff(time_coords.astype('datetime64[D]')).astype(int)
        time_gaps = torch.tensor(time_deltas, dtype=torch.int32)  # Ensure tensor format
    else:
        time_gaps = torch.empty((0,), dtype=torch.int32)  # Empty tensor if not enough time points
    return time_gaps


def select_random_timestamps(chunk, coords, num_timestamps=11):
    """
    Selects a random subset of timestamps for each sample in a data cube.

    Args:
        chunk (dict of np.ndarray): The data cube, where each variable is a NumPy array of shape
                                  (num_samples, num_time, height, width).
        coords (dict): Coordinates associated with the data cube. Assumes 'time' key exists.
        num_timestamps (int): The number of random timestamps to select for each sample.

    Returns:
        updated_chunk (dict of np.ndarray): The updated data cube with selected timestamps.
        updated_coords (dict): The updated coordinates with adjusted 'time'.
    """
    # Determine the shape from a representative variable
    example_var = next(iter(chunk.values()))  # Take the first variable as an example
    num_samples, total_timestamps, height, width = example_var.shape

    # Generate random indices for selecting timestamps for each sample
    selected_indices = np.array([
        sorted(random.sample(range(total_timestamps), num_timestamps)) for _ in range(num_samples)
    ])

    # Create a new dictionary to store the updated data
    updated_chunk = {}

    # Iterate over each variable in the chunk
    for var_name, data in chunk.items():
        # Prepare an array to store the selected timestamps for this variable
        selected_data = np.empty((num_samples, num_timestamps, height, width), dtype=data.dtype)

        # Apply random selection for each sample
        for i in range(num_samples):
            selected_data[i] = data[i, selected_indices[i], :, :]

        # Store the updated data in the new chunk
        updated_chunk[var_name] = selected_data

    # Update 'time' in coords
    updated_coords = coords.copy()
    updated_coords['time'] = np.array([
        np.array(coords['time'][i]).flatten()[selected_indices[i]] for i in range(num_samples)
    ])

    # Verify that all updated samples have exactly num_timestamps
    for var_name, data in updated_chunk.items():
        assert data.shape[1] == num_timestamps, f"Variable {var_name} does not have {num_timestamps} timestamps."

    return updated_chunk, updated_coords


import os
import sys
import time
import torch
import warnings
import numpy as np
import xarray as xr
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Dict, Tuple, List
from itertools import product
from model.model import TransformerAE
from utils.utils import compute_time_gaps
from multiprocessing import Pool
from dataset.si_dataset import ds
import bottleneck as bn

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def extract_patch_pool_safe(args):
    import warnings
    warnings.filterwarnings('ignore')
    chunk, step_sizes, relevant_coords, idx_vals = args
    slices = tuple(slice(idx, idx + step_sizes[dim]) for idx, dim in zip(idx_vals, relevant_coords))
    return chunk[:, slices[0], slices[1], slices[2]]


def split_dataarray_parallel(
    chunk: np.ndarray,
    coords: Dict[str, np.ndarray],
    sample_size: List[Tuple[str, int]],
    overlap: List[Tuple[str, float]] = None,
    n_jobs: int = 24,
    #samples_per_split: int = 20910,
    samples_per_split: int = 209023,
    split_id: int = 0,
    min_central_time: int = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Split a 4D xarray.DataArray into overlapping patches, extracting all variables at once.

    Args:
        chunk (xr.DataArray): Input array with shape (index, time, y, x)
        sample_size (List[Tuple[str, int]]): Dimensions and patch sizes (e.g. [('time', 11), ('y', 15), ('x', 15)])
        overlap (List[Tuple[str, float]]): Fractional overlap per dimension
        n_jobs (int): Number of parallel jobs
        samples_per_split (int): Number of samples to return in this split
        split_id (int): Which split to return (used for batching across multiple calls)

    Returns:
        Tuple:
            - patches: np.ndarray with shape (n_samples, index, t, y, x)
            - coords: dict of coordinate values per sample (e.g. {'time': [...], 'y': [...], 'x': [...]})
    """

    # === Step 1: Prepare dimension dictionaries ===
    sample_size_dict = dict(sample_size)
    overlap_dict = dict(overlap) if overlap else {dim: 0.0 for dim in sample_size_dict}
    relevant_coords = [coord for coord, _ in sample_size]

    shape_dict = dict(zip(relevant_coords, chunk.shape[1:]))

    step_sizes = {dim: sample_size_dict[dim] for dim in relevant_coords}
    overlap_steps = {
        dim: int(step_sizes[dim] * overlap_dict.get(dim, 0.0)) if step_sizes[dim] > 1 else 0
        for dim in relevant_coords
    }

    # === Step 2: Compute valid start indices ===
    start_indices = {
        dim: [
            i for i in range(0, shape_dict[dim], step_sizes[dim] - overlap_steps[dim])
            if i + step_sizes[dim] <= shape_dict[dim]
        ]
        for dim in relevant_coords
    }

    # === Step 3: Generate combinations of patch start positions ===
    index_grid = list(product(*[start_indices[dim] for dim in relevant_coords]))

    # === Step 4: Select the current batch (split) of samples ===
    start = split_id * samples_per_split
    end = min((split_id + 1) * samples_per_split, len(index_grid))
    batch_indices = index_grid[start:end]

    # === Efficient check: Are all central coordinates of this split NaN? ===

    # Define central offsets for each dimension based on patch size
    center_offsets = {
        "time": sample_size_dict["time"] // 2,
        "y": sample_size_dict["y"] // 2,
        "x": sample_size_dict["x"] // 2,
    }

    # Convert batch indices to numpy array
    if len(batch_indices) == 0:
        empty_patches = np.empty(
            (0, chunk.shape[0], sample_size_dict["time"], sample_size_dict["y"], sample_size_dict["x"]))
        empty_coords = {dim: np.empty((0, step_sizes[dim])) for dim in relevant_coords}
        return empty_patches, empty_coords

    batch_indices_arr = np.array(batch_indices)  # shape: (N, 3)
    center_offsets_arr = np.array([center_offsets[dim] for dim in relevant_coords])  # shape: (3,)

    # Calculate the coordinates of the central point of every patch
    central_coords_arr = batch_indices_arr + center_offsets_arr  # shape: (N, 3)

    # Prepare indices for advanced numpy indexing
    t_idx = central_coords_arr[:, 0]
    y_idx = central_coords_arr[:, 1]
    x_idx = central_coords_arr[:, 2]

    # Gather all central values across all channels and all patch centers
    #vals = np.stack([chunk[:, t, y, x] for t, y, x in zip(t_idx, y_idx, x_idx)], axis=1)
    vals_b0 = np.array([chunk[0, t, y, x] for t, y, x in zip(t_idx, y_idx, x_idx)])

    # If there are no valid points or all points are NaN, skip extraction for this split
    if vals_b0.size == 0 or np.isnan(vals_b0).all():
        print(f"⚠️ All central patch positions are NaN for split {split_id}. Skipping extraction.")
        empty_patches = np.empty(
            (0, chunk.shape[0], sample_size_dict["time"], sample_size_dict["y"], sample_size_dict["x"]))
        empty_coords = {dim: np.empty((0, step_sizes[dim])) for dim in relevant_coords}
        return empty_patches, empty_coords

    # Filter valid patches based on central coordinate
    vals = np.broadcast_to(vals_b0, (chunk.shape[0], len(vals_b0)))
    central_valid_mask = ~np.isnan(vals).all(axis=0)

    # Retain only valid patch positions
    batch_indices = [batch_indices[i] for i in range(len(batch_indices)) if central_valid_mask[i]]
    #batch_indices = np.array(batch_indices)[central_valid_mask].tolist()

    # === Step 5: Extract patches ===
    def extract_patch(idx_vals):
        warnings.filterwarnings('ignore')
        # Convert dictionary of slices into tuple of slices for numpy indexing
        slices = tuple(slice(idx, idx + step_sizes[dim]) for idx, dim in zip(idx_vals, relevant_coords))
        return chunk[:, slices[0], slices[1], slices[2]]

    patches = Parallel(n_jobs=n_jobs)(
        delayed(extract_patch)(idx_vals)
        for idx_vals in tqdm(batch_indices, desc=f"Extracting patches (Split {split_id})")
    )

    # === Step 6: Extract coordinate slices (optional) ===
    coord_result = {}
    for dim in relevant_coords:
        coord_array = coords[dim]
        start_indices_dim = [idx[relevant_coords.index(dim)] for idx in batch_indices]
        coord_slices = Parallel(n_jobs=n_jobs)(
            delayed(lambda idx: coord_array[idx:idx + step_sizes[dim]])(start_idx)
            for start_idx in tqdm(start_indices_dim, desc=f"Extracting {dim} slices")
        )
        coord_result[dim] = np.stack(coord_slices)

    return np.stack(patches), coord_result

class XrFeatureDataset:
    def __init__(self, data_cube, sample_size=None, overlap=None, block_size=None):
        self.data_cube = data_cube
        #self.block_size = block_size if block_size else get_chunk_sizes(data_cube)
        #print(self.block_size)
        self.time_block_size = 25
        self.time_len = 1000
        self.chunk_starts = self._compute_time_chunk_starts()
        self.sample_size = sample_size
        self.overlap = overlap

        self.chunk_idx = 3
        self.split_idx = 24
        self.chunk = None
        if self.chunk_idx == 0:
            self.min_central_time = 19
        else: self.min_central_time = None

    def __iter__(self):
        return self


    def _compute_time_chunk_starts(self):
        step = self.time_block_size
        return [
            i for i in range(0, self.time_len, step)
            if i + self.time_block_size <= self.time_len
        ]


    def reset(self):
        pass  # Nothing to reset in this version


    def __next__(self):
        if self.chunk_idx >= 4:
            raise StopIteration

        if self.split_idx == 15 and self.chunk_idx == 0 or self.split_idx == 10 and self.chunk_idx == 4 or self.split_idx >= 25 and self.chunk_idx > 0:
            self.chunk_idx += 1
            self.split_idx = 0
            self.chunk = None

        t_start = self.chunk_starts[self.chunk_idx]
        t_end = t_start + self.time_block_size

        if self.chunk is None:
            warnings.filterwarnings('ignore')
            if self.chunk_idx > 0: t_start -= 10
            print(f'Getting chunk at time={t_start} - {t_end}')
            chunk = self.data_cube.isel(time=slice(t_start, t_end))

            nan_count = np.isnan(chunk.values).sum()
            non_nan_count = np.count_nonzero(~np.isnan(chunk.values))
            print(f"Chunk NaNs: {nan_count:,}, Non-NaNs: {non_nan_count:,}")
            print(f'Chunk shape: {chunk.shape}')

            coords = {k: chunk.coords[k].values for k in chunk.coords}
            self.chunk = (chunk.values, coords)

        chunk_values, coords = self.chunk

        print(f'Splitting chunk {self.chunk_idx}.{self.split_idx}')
        patches, coords = split_dataarray_parallel(
            chunk=chunk_values,
            coords=coords,
            sample_size=self.sample_size,
            overlap=self.overlap,
            n_jobs=32,
            samples_per_split=34848 * 2,
            #samples_per_split=41805,
            split_id=self.split_idx,
            min_central_time=self.min_central_time
        )

        if patches.shape[0] == 0:
            print(f"No valid samples in chunk {self.chunk_idx}, split {self.split_idx}. Skipping.")
            self.split_idx += 1
            #return self.__next__()
            return None
        print(patches.shape)
        print(f'Fill remaining NaNs with per-patch means')
        # Fill remaining NaNs with per-patch means
        t0 = time.time()
        patches = np.where(
            np.isnan(patches),
            np.nanmean(patches, axis=(2, 3, 4), keepdims=True),
            patches
        )
        #mean_patch = bn.nanmean(patches, axis=(2, 3, 4), keepdims=True)
        #patches = np.where(np.isnan(patches), mean_patch, patches)

        print(f"NaN filling: {time.time() - t0:.2f} seconds")

        t0 = time.time()
        time_gaps = compute_time_gaps(coords["time"])
        print(f"time_gaps computation: {time.time() - t0:.2f} seconds")

        print(f"✔️ Processed chunk {self.chunk_idx}.{self.split_idx}")

        self.split_idx += 1

        return patches, coords, time_gaps


def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Step 1: Load your trained model
    model = TransformerAE(dbottleneck=7)  # Replace with actual init
    #checkpoint = torch.load("/scratch/jpeters/DeepFeatures/checkpoints/149_002_018_080_vc/ae-7-epoch=212-val_loss=3.963e-03.ckpt", map_location=device)
    checkpoint = torch.load("/scratch/jpeters/DeepFeatures/checkpoints/149_002_018_080_test/1/ae-7-epoch=106-val_loss=4.960e-03.ckpt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Step 2: Open Zarr target
    #zarr_path = "/net/data_ssd/deepfeatures/sciencecubes_processed/000000_feature_cube.zarr"
    zarr_path = "/net/data_ssd/deepfeatures/sciencecubes_processed/000000_feature_cube_small.zarr"
    z = xr.open_zarr(zarr_path)
    features_zarr = z["features"]
    time_index = z["time"].values
    y_index = z["y"].values
    print(len(y_index))
    x_index = z["x"].values
    print(len(x_index))

    # Step 3: Create dataset loader
    from dataset.si_dataset import ds

    ds = ds.isel(y=slice(9-7, 273+7), x=slice(260-7, 524+7))
    ds = ds.sel(time=slice("2019-10-16", "2021-03-10"))

    dataset = XrFeatureDataset(
        data_cube=ds,
        sample_size=[("time", 11), ("y", 15), ("x", 15)],
        overlap=[("time", 10. / 11.), ("y", 14. / 15.), ("x", 14. / 15.)],
        #block_size=[('time', 53), ('y', 1000), ('x', 1000)]
        block_size=[('time', 25), ('y', 264), ('x', 264)]

    )
    batch_size = 2048

    # === Prepare buffer for a whole time chunk ===
    from collections import OrderedDict
    buffers = OrderedDict()
    written_time_indices = set()

    # Step 4: Iterate and write features
    with torch.no_grad():
        for split_idx, split in enumerate(dataset):
            if split is None: continue
            data, coords, time_gaps = split
            coord_time, coord_x, coord_y = coords['time'], coords['x'], coords['y']

            num_samples = data.shape[0]
            print(f'num_samples={num_samples}')

            total_absolute_error = 0.0
            total_sample_count = 0

            progress_bar = tqdm(
                range(0, num_samples, batch_size),
                desc=f"Model on chunk {dataset.chunk_idx}.{dataset.split_idx - 1}",
                unit="batch",
                dynamic_ncols=True  # optional, for cleaner terminal output
            )

            for batch_start in progress_bar:
                batch_end = min(batch_start + batch_size, num_samples)
                data_batch = data[batch_start:batch_end]
                #mask_batch = mask[batch_start:batch_end]
                time_gaps_batch = time_gaps[batch_start:batch_end]
                time_batch = coord_time[batch_start:batch_end]
                x_batch = coord_x[batch_start:batch_end]
                y_batch = coord_y[batch_start:batch_end]

                # Convert to tensors (rename to avoid overwrite)
                data_tensor = torch.tensor(data_batch, dtype=torch.float32, device=device).permute(0, 2, 1, 3, 4)
                time_gaps_tensor = torch.tensor(time_gaps_batch, dtype=torch.int32, device=device)

                # Run model
                decoded_batch, features = model(data_tensor, time_gaps=time_gaps_tensor)

                # Compute MAE at center
                input_center = data_tensor[:, 5, :, 7, 7]
                output_center = decoded_batch[:, 5, :, 7, 7]
                #mae = torch.mean(torch.abs(output_center - input_center))
                batch_absolute_error = torch.sum(torch.abs(output_center - input_center)).item()
                batch_sample_count = output_center.numel()

                total_absolute_error += batch_absolute_error
                total_sample_count += batch_sample_count

                # Update progress bar with current mean MAE
                mean_mae_so_far = total_absolute_error / total_sample_count
                postfix_str = f"MAE: {mean_mae_so_far:.8f}"
                progress_bar.set_postfix_str(str(postfix_str))

                central_time = time_batch[:, 5]  # index 5 of time window (11)
                #print(f'central_time: {central_time}')
                central_x = x_batch[:, 7]  # index 7 of spatial (15)
                #print(f'central_time: {central_x}')

                central_y = y_batch[:, 7]
                #print(f'central_time: {central_y}')


                # Vectorized index lookup
                t_idx = np.array([np.where(time_index == t)[0][0] for t in central_time])
                x_idx = np.array([np.where(x_index == x)[0][0] for x in central_x])
                y_idx = np.array([np.where(y_index == y)[0][0] for y in central_y])

                features = features.cpu().numpy()


                for unique_t in np.unique(t_idx):
                    if unique_t not in buffers:
                        buffers[unique_t] = np.full((features.shape[1], len(y_index), len(x_index)), np.nan,
                                                    dtype=np.float32)

                    indices_for_t = np.where(t_idx == unique_t)[0]
                    buffers[unique_t][:, y_idx[indices_for_t], x_idx[indices_for_t]] = features[
                        indices_for_t].transpose(1, 0)

                finished_timestamps = list(buffers.keys())[:-1]

                if finished_timestamps:

                    # === Warn if any timestamps were already written ===
                    for t in finished_timestamps:
                        if t in written_time_indices:
                            print(f"⚠️ WARNING: Timestamp index {t} already written — overwriting.")
                        else:
                            written_time_indices.add(t)

                    #concatenated_array = np.stack([buffers[t] for t in finished_timestamps], axis=1)
                    concatenated_array = np.stack([buffers.pop(t) for t in finished_timestamps], axis=1)

                    start_time = min(finished_timestamps)
                    end_time = max(finished_timestamps) + 1


                    da = xr.DataArray(
                        concatenated_array,
                        dims=["feature", "time", "y", "x"],
                        coords={
                            "feature": features_zarr.feature.values,
                            "time": time_index[start_time:end_time],
                            "y": y_index,
                            "x": x_index
                        }
                    )
                    ds = xr.Dataset({"features": da}).drop_vars("feature")

                    ds.to_zarr(zarr_path, mode="r+", region={
                        "time": slice(start_time, end_time),
                        "y": slice(0, len(y_index)),
                        "x": slice(0, len(x_index))
                    })

                    print(f"✅ Written time frames {start_time} to {end_time - 1}")


        for t_remaining in buffers:
            feature_block = buffers[t_remaining]  # shape (feature, y, x)

            da = xr.DataArray(
                feature_block,
                dims=["feature", "y", "x"],
                coords={
                    "feature": features_zarr.feature.values,
                    "y": y_index,
                    "x": x_index
                }
            ).expand_dims({"time": [time_index[t_remaining]]})

            ds = xr.Dataset({"features": da}).drop_vars("feature")

            ds.to_zarr(zarr_path, mode="r+", region={
                "time": slice(t_remaining, t_remaining + 1),
                "y": slice(0, len(y_index)),
                "x": slice(0, len(x_index))
            })

            print(f"✅ Written final time frame {t_remaining}")



    print("✅ All features written to Zarr cube.")


if __name__ == "__main__":
    main()

# PYTHONPATH=.. nohup python train.py > train.out 2>&1 &
# tail -f fill_cube4.out


# PYTHONPATH=.. nohup python feature_datacube_dataarray.py > fill_cube4.out 2>&1 &
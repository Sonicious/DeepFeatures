import os
import sys
import torch
import numpy as np
import xarray as xr
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import Dict, Tuple, List
from itertools import product
from model.model import TransformerAE
from utils.utils import drop_if_central_point_nan_or_inf, concatenate, compute_time_gaps
from ml4xcube.utils import get_chunk_sizes, get_chunk_by_index, calculate_total_chunks
from dataset.si_dataset import ds
from ml4xcube.preprocessing import  drop_nan_values, fill_nan_values

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



def split_chunk_parallel(chunk: Dict[str, np.ndarray], coords: Dict[str, np.ndarray],
                         sample_size: List[Tuple[str, int]] = None,
                         overlap: List[Tuple[str, float]] = None,
                         n_jobs: int = 24,
                         #samples_per_split: int = 167220,
                         samples_per_split: int = 20910,
                         split_id:int = 0) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if sample_size is None:
        return {x: chunk[x].ravel() for x in chunk.keys()}, {x: coords[x].ravel() for x in coords.keys()}

    sample_size_dict = dict(sample_size)
    overlap_dict = dict(overlap) if overlap else {dim: 0.0 for dim in sample_size_dict}
    relevant_coords = [coord for coord, _ in sample_size]

    example_array = next(iter(chunk.values()))
    shape = example_array.shape

    chunk_dim_order = [
        dim for dim in coords
        if coords[dim].ndim == 1 and len(coords[dim]) in shape and dim in relevant_coords
    ]
    shape_dict = dict(zip(chunk_dim_order, shape))
    step_sizes = {dim: sample_size_dict[dim] for dim in chunk_dim_order}
    overlap_steps = {
        dim: int(step_sizes[dim] * overlap_dict.get(dim, 0.0)) if step_sizes[dim] > 1 else 0
        for dim in chunk_dim_order
    }

    start_indices = {
        dim: [
            i for i in range(0, shape_dict[dim], step_sizes[dim] - overlap_steps[dim])
            if i + step_sizes[dim] <= shape_dict[dim]
        ]
        for dim in chunk_dim_order
    }

    coord_cf = {x: coords[x] for x in relevant_coords if x in coords}
    index_grid = list(product(*[start_indices[dim] for dim in chunk_dim_order]))

    start = split_id * samples_per_split
    end = min((split_id + 1) * samples_per_split, len(index_grid))
    batch_indices = index_grid[start:end]
    n_samples = len(batch_indices)

    # === Extract data per variable ===
    def extract_sample_single_var(idx_vals, var):
        idx_dict = dict(zip(chunk_dim_order, idx_vals))
        slices = tuple(slice(idx_dict[dim], idx_dict[dim] + step_sizes[dim]) for dim in chunk_dim_order)
        return chunk[var][slices]

    result = {}


    for var in chunk:
        result[var] = np.stack(Parallel(n_jobs=n_jobs)(
            delayed(extract_sample_single_var)(idx_vals, var)
            for idx_vals in tqdm(batch_indices, desc=f"Extracting variable {var} (Split {split_id})")
        ))
#
    # === Extract coords per coordinate key ===
    def extract_single_coord(idx_vals, coord_key):
        idx_dict = dict(zip(chunk_dim_order, idx_vals))
        start = idx_dict[coord_key]
        stop = start + step_sizes[coord_key]
        return coord_cf[coord_key][start:stop]

    coord_result = {}
    for coord_key in coord_cf:
        coord_result[coord_key] = np.stack(Parallel(n_jobs=n_jobs)(
            delayed(extract_single_coord)(idx_vals, coord_key)
            for idx_vals in tqdm(batch_indices, desc=f"Extracting coord {coord_key} (Batch {split_id})")
        ))

    return result, coord_result

class XrFeatureDataset:
    def __init__(self, data_cube, sample_size=None, overlap=None, block_size=None):
        self.data_cube = data_cube
        self.block_size = block_size if block_size else get_chunk_sizes(data_cube)
        print(self.block_size)
        self.sample_size = sample_size
        self.overlap = overlap

        self.chunk_idx_list = list(range(calculate_total_chunks(self.data_cube)))
        self.chunk_idx = 0
        self.split_idx = 0
        self.chunk = None

    def __iter__(self):
        return self


    def reset(self):
        pass  # Nothing to reset in this version


    def __next__(self):
        if self.chunk_idx >= len(self.chunk_idx_list):
            raise StopIteration

        idx = self.chunk_idx_list[self.chunk_idx]

        if self.chunk is None:
            print(f'Getting chunk {idx}')
            chunk, coords = get_chunk_by_index(self.data_cube, idx, block_size=self.block_size)

        if self.split_idx == 2000:
            self.chunk_idx += 1
            self.split_idx = 0

            print(f'Getting chunk {idx}')
            chunk, coords = get_chunk_by_index(self.data_cube, idx, block_size=self.block_size)

        print(f'Splitting chunk {idx}.{self.split_idx}')
        """cf, coords = split_chunk_parallel(chunk, coords, sample_size=self.sample_size, overlap=self.overlap, n_jobs=24, split_id=self.split_idx)

        import pickle


        print(f'number of elements after splitting {cf["B01"].shape[0]}')

        vars_all = list(cf.keys())
        print(f'Dropping NaNs/INFs from chunk {idx}')
        cf, coords = drop_nan_values(cf, coords, mode='if_all_nan', vars=vars_all)
        print(f'number of elements after removing entire nan samples {cf["B01"].shape[0]}')

        vars_all = list(cf.keys())

        # Save to file
        with open("split_output.pkl", "wb") as f:
            pickle.dump((cf, coords), f)
            print('cf, coords saved')"""

        # Load from file
        with open("split_output.pkl", "rb") as f:
            import pickle
            cf, coords = pickle.load(f)

        vars_all = list(cf.keys())


        cf, coords = drop_if_central_point_nan_or_inf(cf, coords, vars=vars_all)
        print(f'number of elements after removing nans in center {cf["B01"].shape[0]}')

        valid_mask = {var: ~np.isnan(cf[var]) for var in vars_all}
        print(f'Filling NaNs in chunk {idx}')
        cf = fill_nan_values(cf, vars=vars_all, method='sample_mean')

        data, coords, valid_mask = concatenate([cf], [coords], [valid_mask])
        time_gaps = compute_time_gaps(coords["time"])
        print(f"✔️ Processed chunk {idx} batch {self.split_idx}")

        self.split_idx += 1

        return data, coords, time_gaps, valid_mask


def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Step 1: Load your trained model
    model = TransformerAE(dbottleneck=7)  # Replace with actual init
    checkpoint = torch.load("/scratch/jpeters/DeepFeatures/checkpoints/149_002_018_080_vc/ae-7-epoch=212-val_loss=3.963e-03.ckpt", map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Step 2: Open Zarr target
    zarr_path = "/net/data_ssd/deepfeatures/sciencecubes_processed/000000_features.zarr"
    z = xr.open_zarr(zarr_path)
    features_zarr = z["features"]
    time_index = z["time"][:]
    x_index = z["x"][:]
    y_index = z["y"][:]

    # Step 3: Create dataset loader
    dataset = XrFeatureDataset(
        data_cube=ds,
        sample_size=[("time", 11), ("y", 15), ("x", 15)],
        overlap=[("time", 10. / 11.), ("y", 14. / 15.), ("x", 14. / 15.)],
        block_size=[('time', 53), ('y', 1000), ('x', 1000)]
    )

    batch_size = 2048

    # Step 4: Iterate and write features
    with torch.no_grad():
        for chunk_idx, chunk in enumerate(dataset):
            data, coords, time_gaps, _ = chunk
            coord_time, coord_x, coord_y = coords['time'], coords['x'], coords['y']

            num_samples = data.shape[0]
            print(f'num_samples={num_samples}')
            batch_size = 24

            total_absolute_error = 0.0
            total_sample_count = 0

            progress_bar = tqdm(
                range(0, num_samples, batch_size),
                desc=f"Applying model on chunk {chunk_idx}",
                unit="batch"
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
                data_tensor = torch.tensor(data_batch, dtype=torch.float32, device=device).permute(0, 1, 4, 2, 3)
                #mask_tensor = torch.tensor(mask_batch, dtype=torch.bool, device=device).permute(0, 1, 4, 2, 3)
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
                progress_bar.set_postfix(mean_MAE=f"{mean_mae_so_far:.6f}")

                central_time = time_batch[:, 5]  # index 5 of time window (11)
                central_x = x_batch[:, 7]  # index 7 of spatial (15)
                central_y = y_batch[:, 7]

                # === Schreiben in Zarr ===
                for i in range(features.shape[0]):
                    try:
                        t_idx = int(np.where(time_index == central_time[i])[0][0])
                        x_idx = int(np.where(x_index == central_x[i])[0][0])
                        y_idx = int(np.where(y_index == central_y[i])[0][0])
                        features_zarr[:, t_idx, y_idx, x_idx] = features[i].cpu().numpy()
                    except IndexError:
                        print(f"⚠️ Sample {i} with time={central_time[i]}, x={central_x[i]}, y={central_y[i]} not in Zarr grid - skipping")
                        continue



    print("✅ All features written to Zarr cube.")


if __name__ == "__main__":
    main()

# PYTHONPATH=.. nohup python fill_cube.py > fill_cube.out 2>&1 &
# tail -f fill_cube.out
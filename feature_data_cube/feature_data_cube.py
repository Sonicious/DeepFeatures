import os
import sys
import time
import torch
import pathlib
import warnings
import numpy as np
import xarray as xr
from bokeh.core.property.struct import Optional
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from utils.utils import compute_time_gaps, extract_center_coordinates
from model.model import TransformerAE
from dataset.prepare_dataarray import prepare_spectral_data
from dataset.preprocess_sentinel import extract_sentinel2_patches


CUBE_ID = '004'
BATCH_SIZE = 550
BASE_PATH = '/net/data/deepfeatures/science/0.1.0'
OUTPUT_PATH = '/net/data/deepfeatures/feature'

from multiprocessing import Pool


# --- top-level worker (must be top-level for multiprocessing pickling) ---
def _extract_worker(args):
    (data_sub, time_sub, y_sub, x_sub,
     y_keep_min, y_keep_max,
     extractor_kwargs) = args

    try:
        interior = data_sub[:, 5:-5, 7:-7, 7:-7]
    except IndexError:
        # too small to ever form a valid patch
        return None

    if not np.any(~np.isnan(interior)):
        # no valid data → no possible patches
        return None

    patches, coords, valid_mask, not_val = extract_sentinel2_patches(
        data_sub,
        time_sub,
        y_sub,
        x_sub,
        **extractor_kwargs
    )

    if patches is None or getattr(patches, "shape", (0,))[0] == 0:
        return None

    coords_f = dict(coords)

    return patches, coords_f, valid_mask, not_val


def extract_sentinel2_patches_pool(
    data,
    time_coords,
    y_coords,
    x_coords,
    *,
    processes: int = 4,
    halo: int = 14,
    **kwargs
):
    """
    Drop-in replacement wrapper for extract_sentinel2_patches that runs preprocessing
    in parallel using multiprocessing.Pool (default 4 processes) by splitting along Y.

    Parameters
    ----------
    data : np.ndarray
        Array shaped (C, T, Y, X) as you pass into extract_sentinel2_patches today.
        (If you pass other layouts, adapt the Y axis below.)
    time_coords, y_coords, x_coords : np.ndarray
        1D coordinate arrays for the chunk.
    processes : int
        Number of worker processes (default 4).
    pool : multiprocessing.Pool or None
        If None, a temporary pool is created and closed inside this function.
        If provided, it will be used (recommended: create it once globally).
    halo : int
        Extra pixels on each stripe boundary to keep patch windows valid.
        For 15x15 patches, 14 is safe (avoids boundary misses with sliding).
    **kwargs :
        Forwarded to extract_sentinel2_patches (time_win, strides, max_total_gap, inference, ...)

    Returns
    -------
    patches_all : torch.Tensor
    coords_all : dict
    valid_mask_all : torch.Tensor
    not_val : bool
    """

    # Expecting (C, T, Y, X)
    Yfull = data.shape[2]
    if Yfull == 0:
        return torch.empty((0,)), {}, torch.empty((0,)), True

    # build 4 stripes (or `processes`) along Y
    edges = np.linspace(0, Yfull, processes + 1, dtype=int)

    extractor_kwargs = dict(kwargs)

    jobs = []
    for i in range(processes):
        y0 = int(edges[i])
        y1 = min(Yfull, int(edges[i + 1]) + halo)

        data_sub = data[:, :, y0:y1, :]
        y_sub = y_coords[y0:y1]

        # keep-only interior by coordinate value to prevent duplicates
        y_keep_min = float(y_coords[y0])
        # make upper bound exclusive; if y1 exists use it, otherwise one step above last
        y_keep_max = float(y_coords[y1-1])

        jobs.append((data_sub, time_coords, y_sub, x_coords, y_keep_min, y_keep_max, extractor_kwargs))

    if len(jobs) == 0:
        return torch.empty((0,)), {}, torch.empty((0,)), True


    with Pool(processes=processes) as pool:
        results = pool.map(_extract_worker, jobs)

    results = [r for r in results if r is not None]
    if len(results) == 0:
        return torch.empty((0,)), {}, torch.empty((0,)), True

    patches_list, coords_list, masks_list, not_vals = zip(*results)


    patches_all = torch.cat(patches_list, dim=0)
    valid_mask_all = torch.cat(masks_list, dim=0)
    not_val = all(not_vals)

    # merge coords: concatenate 1D per-patch arrays, keep non per-patch arrays as-is
    coords_all = {
        "time": np.concatenate([c["time"] for c in coords_list], axis=0),
        "y": np.concatenate([c["y"] for c in coords_list], axis=0),
        "x": np.concatenate([c["x"] for c in coords_list], axis=0),
    }

    return patches_all, coords_all, valid_mask_all, not_val


def create_empty_dataset(feature_names, xs, ys, out_path, times=None, dtype=np.float32):
    """
    Create an empty xarray.Dataset with dims (feature, time, y, x).
    If times is None -> time dim length 0 (good for appending).
    """


    if os.path.exists(out_path):
        # Just open existing store
        return xr.open_zarr(out_path)

    times = np.asarray(times).astype("datetime64[ns]")

    shape = (len(feature_names), len(times), len(ys), len(xs))
    data = np.full(shape, np.nan, dtype=dtype)

    da = xr.DataArray(
        data,
        dims=("feature", "time", "y", "x"),
        coords={
            "feature": np.asarray(feature_names, dtype=str),
            "time": times,
            "y": np.asarray(ys),
            "x": np.asarray(xs),
        },
        name="features",
    )

    # chunk along time=1 to match region writes
    encoding = {"features": {"chunks": (len(feature_names), 1, len(ys), len(xs))}}

    ds_out = xr.Dataset({"features": da}).drop_vars("feature")
    ds_out.to_zarr(out_path, mode="w", encoding=encoding)
    return ds_out


def init_output_from_source(da, feature_names, out_path):
    """
    Determine valid timestamps (>= 5% complete pixels) on the cropped grid
    and create the target Zarr with exactly these times.
    - time is sliced as time[5:-5]
    - y, x are sliced as [7:-7]
    """

    if os.path.exists(output_path):
        ds0 = xr.open_zarr(output_path, consolidated=True)

        feats = ds0["features"]  # (feature, time, y, x)
        reduce_dims = tuple(d for d in feats.dims if d != "time")

        # Build mask and COMPUTE it to a NumPy bool array (1D over time)
        empty_mask_da = feats.isnull().all(dim=reduce_dims)  # (time,) dask-backed
        empty_mask_np = empty_mask_da.compute().values.astype(bool)  # -> (T,)

        # Index times via NumPy (avoid .where(drop=True) with dask)
        times_np = ds0["time"].values  # (T,)
        empty_times = times_np[empty_mask_np]  # timestamps entirely NaN

        xs = ds0["x"].values
        ys = ds0["y"].values
        return ds0, empty_times, xs, ys

    # --- compute strict keep times on the same array you iterate over ---
    # Optional crop to avoid borders (keep if you want it strict)
    da_c = da.isel(time=slice(5, -5), y=slice(7, -7), x=slice(7, -7))
    #da_c = da.isel(time=slice(5, -5), y=slice(7, -7), x=slice(7, -7), band=slice(0, 12))


    # Decide which bands define "completeness" (strict)
    # If your da now has 149 channels after prepare_spectral_data,
    # and you want strictness on the original 10 S2 reflectances, slice them:
    bands_for_check = da_c.isel(index=slice(0, 10))

    # Complete pixel = ALL chosen bands valid (strict)
    complete_px = bands_for_check.notnull().all(dim="index")  # (time,y,x)
    frac_complete = complete_px.mean(dim=("y", "x"))  # (time,)
    keep_mask = (frac_complete >= 0.05).compute().values  # strict 5%

    ok_idx = np.flatnonzero(keep_mask)
    times_ok_ns = np.asarray(da_c.time.values)[ok_idx]

    global_xs = da.x.values[7:-7]
    global_ys = da.y.values[7:-7]

    ds0 = create_empty_dataset(
        feature_names=feature_names,
        xs=global_xs,
        ys=global_ys,
        out_path=out_path,
        times=times_ok_ns,
        dtype=np.float32,
    )
    return ds0, times_ok_ns, global_xs, global_ys


def reset_frame():
    global current_canvas, filled_once, current_time
    current_canvas[:] = np.nan
    filled_once[:] = False
    current_time = None


def flush_frame(canvas, f_ds, out_path, time):
    """Add current frame to Zarr."""

    if time is None:
        return False

    t = np.datetime64(time).astype("datetime64[ns]")
    t_arr = f_ds.time.values
    idx = int(np.argmin(np.abs(t_arr - t)))

    print(canvas[:, np.newaxis, :, :].shape)
    # shape (C, Y, X) -> expand to (C, 1, Y, X)
    da = xr.DataArray(
        canvas[:, np.newaxis, :, :],   # add time axis in 2nd position
        dims=("feature", "time", "y", "x"),
        coords={
            "feature": f_ds.feature.values,
            "time": f_ds.time.values[idx:idx+1],
            "y": f_ds.y.values,
            "x": f_ds.x.values,
        },
        name="features",
    )

    ds = xr.Dataset({"features": da}).drop_vars("feature")

    ds.to_zarr(out_path, mode="r+", region={
        "time": slice(idx, idx + 1),
        "y": slice(0, len(f_ds.y)),
        "x": slice(0, len(f_ds.x))
    })


    return True


def coord_to_idx(vals, mapping, axis_vals):
    """
    Map 1D array of coords -> indices.
    Uses dict fast path for exact float matches.
    Falls back to nearest index if not found.
    """
    vals = np.asarray(vals)
    idxs = np.empty(vals.shape, dtype=np.int64)

    for j, v in enumerate(vals):
        fv = float(v)
        if fv in mapping:
            idxs[j] = mapping[fv]
        else:
            # fallback: nearest
            idxs[j] = int(np.argmin(np.abs(axis_vals - v)))
    return idxs


class XrFeatureDataset:
    def __init__(
            self,
            data_cube: xr.DataArray,
            times_ok_ns,
            time_block_size: int = 11,
            space_block_size: int = 250,
            time_overlap: int = 10,
            space_overlap: int = 14
    ):
        self.data_cube = data_cube
        self.times_ok_ns = times_ok_ns
        self.time_block_size = time_block_size
        self.space_block_size = space_block_size
        self.time_overlap = time_overlap
        self.space_overlap = space_overlap

        # infer sizes (dims: band, time, y, x)
        self.time_len = int(data_cube.sizes["time"])
        self.y_len = int(data_cube.sizes["y"])
        self.x_len = int(data_cube.sizes["x"])

        print(self.y_len, self.x_len, self.time_len)

        self.save_frame = True

        self.chunks_bounds = self.compute_bounds(time_slide = True, time_block=self.time_block_size, space_block=self.space_block_size)  # list of (t0,t1,y0,y1,x0,x1)
        print(f'Chunks to process: {len(self.chunks_bounds)}')
        self.chunk_idx = 0

        # fast membership test on ns-int
        self.times_ok_ns = np.asarray(times_ok_ns).astype("datetime64[ns]")
        self._times_ok_set = set(self.times_ok_ns.astype("int64").tolist())

    def __iter__(self):
        return self

    def compute_bounds(self, time_slide, time_block, space_block, split_chunk=False):
        """Return list of (t0, t1, y0, y1, x0, x1) with overlaps.
        Ends are computed from the nominal (non-overlapped) starts, then
        starts are shifted backward by the overlap for i>0."""

        def nominal_ranges(n, block, *, sliding=False):
            """
            Compute ranges as (start, end).
            - If sliding=True: use stride=1 (e.g., 0-11, 1-12, 2-13, ...)
            - Else: use stride=block (non-overlapping blocks).
            """
            if sliding:
                return [(i, i + block) for i in range(0, n - block + 1, 1)]
            else:
                return [(i, i + block) for i in range(0, n, block) if i + block <= n]


        if split_chunk and self.chunk_idx == 0:
            t_len = self.time_block_size
        elif split_chunk and self.chunk_idx > 0:
            t_len = self.time_block_size + 10
        else: t_len = self.time_len

        t_nom = nominal_ranges(t_len, time_block, sliding=time_slide)
        y_nom = nominal_ranges(self.y_len, space_block)
        x_nom = nominal_ranges(self.x_len, space_block)

        chunks = []
        # iterate Y, then X, then T  ⟶ fills same spatial frame first
        for (t0_nom, t1_nom) in t_nom:
            if time_slide:
                t0 = t0_nom
            else:
                t0 = t0_nom - self.time_overlap if t0_nom > 0 else t0_nom

            t1 = t1_nom #- self.time_overlap if t0_nom > 0 else t1_nom
            t0 = max(0, t0);
            t1 = min(t_len, t1)

            for (y0_nom, y1_nom) in y_nom:
                y0 = y0_nom - self.space_overlap if y0_nom > 0 else y0_nom
                y1 = y1_nom
                y0 = max(0, y0); y1 = min(self.y_len, y1)

                for (x0_nom, x1_nom) in x_nom:
                    x0 = x0_nom - self.space_overlap if x0_nom > 0 else x0_nom
                    x1 = x1_nom
                    x0 = max(0, x0); x1 = min(self.x_len, x1)

                    chunks.append((t0, t1, y0, y1, x0, x1))

        return chunks

    def subchunk_for_split(self):
        """
        Build a subchunk (bands, time, y, x) and its coords from the given split_idx.
        Expects `self.split_bounds` to be relative to the current chunk.
        Returns: (sub_values, sub_coords, (st0, st1, sy0, sy1, sx0, sx1))
        """

        chunk_values, coords = self.chunk

        st0, st1, sy0, sy1, sx0, sx1 = self.split_bounds[self.split_idx]

        # slice subchunk
        sub_values = chunk_values[:, st0:st1, sy0:sy1, sx0:sx1]
        sub_coords = {
            "time": coords["time"][st0:st1],
            "y": coords["y"][sy0:sy1],
            "x": coords["x"][sx0:sy1] if False else coords["x"][sx0:sx1],  # keep x slice
        }


        return sub_values, sub_coords

    def nan_stats(self, da: xr.DataArray) -> tuple[int, int]:
        """
        Returns (nan_count, non_nan_count) for an xarray.DataArray.
        Works for both NumPy- and Dask-backed arrays.
        """
        # count NaNs across all dims
        nan_da = da.isnull().sum()  # reduces over all dims
        nan = nan_da.compute().item() if getattr(da.data, "chunks", None) else nan_da.item()

        # total elements = product of dimension sizes
        total = 1
        for d in da.dims:
            total *= da.sizes[d]

        return int(nan), int(total - nan)

    def reset(self):
        pass  # Nothing to reset in this version


    def __next__(self):
        #if self.chunk is None:
        warnings.filterwarnings('ignore')

        if self.chunk_idx >= len(self.chunks_bounds):
            raise StopIteration

        t0, t1, y0, y1, x0, x1 = self.chunks_bounds[self.chunk_idx]
        print(f"Getting chunk time={t0}-{t1} y={y0}-{y1} x={x0}-{x1}")

        chunk = self.data_cube.isel(
            time=slice(t0, t1),
            y=slice(y0, y1),
            x=slice(x0, x1),
        )

        coords = {k: chunk.coords[k].values for k in chunk.coords}

        ct_idx = coords["time"].size // 2
        ct = np.datetime64(coords["time"][ct_idx]).astype("datetime64[ns]")
        print(ct)

        if int(ct.astype("int64")) not in self._times_ok_set:
            print(f"⏭️ Skipping chunk {self.chunk_idx}: center time {ct} not in times_ok_ns.")
            self.chunk_idx = ((self.chunk_idx // 16) + 1) * 16 - 1
            self.save_frame = False
            print(f'Setting flag save frame to {self.save_frame}')
            return None, None, None, None

        start_chunk_values = time.time()
        data = chunk.values
        print(f"Chunk values computed in {time.time() - start_chunk_values:.3f} seconds")
        valid_pixel_mask  = np.isnan(data[:12, 5:-5, 7:-7, 7:-7])#.any(axis=0)
        non_nan_count = (~valid_pixel_mask).sum()
        if non_nan_count == 0:

            return None, None, None, None

        nan_count = valid_pixel_mask.sum()


        print(f"Chunk NaNs: {nan_count:,}, Non-NaNs: {non_nan_count:,}")
        print(f'Chunk shape: {data.shape}')




        print(f'Splitting chunk {self.chunk_idx}')
        start_chunk_split = time.time()

        patches_all, coords_all, valid_mask_all, not_val = extract_sentinel2_patches_pool(
            data,
            coords["time"],
            coords["y"],
            coords["x"],
            processes=6,
            time_win=11,
            time_stride=1,
            h_stride=1,
            w_stride=1,
            max_total_gap=195,
            inference=True,
        )

        print(f"Patches preprocessed in {time.time() - start_chunk_split:.3f} seconds")

        #if patches_all.shape[0] == 0 and not_val: self.save_frame = False

        if patches_all.shape[0] == 0: return None, None, None, None


        time_gaps_s2 = compute_time_gaps(coords_all['time'])
        return patches_all, coords_all, valid_mask_all, time_gaps_s2


checkpoint_path = "../checkpoints/ae-epoch=141-val_loss=4.383e-03.ckpt"
device = torch.device("cuda:2")

model = TransformerAE(dbottleneck=6).eval()


checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()


batch_size = BATCH_SIZE
cube_num = CUBE_ID
eps = 1e-6  # avoid div by zero

print(f'processing cube {cube_num}')
ds = xr.open_zarr(f'{BASE_PATH}/{cube_num}.zarr')

da = ds.s2l2a.where((ds.cloud_mask == 0))
n_total = da.sizes["band"] * da.sizes["y"] * da.sizes["x"]
threshold = int(n_total * 0.0)
# Count non-NaN points per time step
valid_data_count = da.notnull().sum(dim=["band", "y", "x"])
# Keep only time steps with at least 3.5% valid data
da = da.sel(time=valid_data_count > threshold)

#if chunks is None: chunks = {"time": 1, "y": 1000, "x": 1000}
da = da.chunk({"time": 1, "y": 1000, "x": 1000})
da = prepare_spectral_data(da, to_ds=False, compute_SI=True, load_b01b09=True)

feature_names = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06']


init_path = f"{BASE_PATH}/{cube_num}.zarr"
output_path = f"{OUTPUT_PATH}/{cube_num}.zarr"
ds0, times_ok_ns, global_xs, global_ys = init_output_from_source(da, feature_names, output_path)
print(times_ok_ns)

print(f'creating feature_{cube_num}.zarr')


x_to_idx = {float(v): i for i, v in enumerate(global_xs)} #['2016-12-27T10:54:42.026000000'
y_to_idx = {float(v): i for i, v in enumerate(global_ys)}


C, Y, X = len(feature_names), len(global_ys), len(global_xs)
current_canvas = np.full((C, Y, X), np.nan, dtype=np.float32)
filled_once = np.zeros((Y, X), dtype=bool)  # first-write-wins for overlaps
current_time = None


dataset = XrFeatureDataset(
    data_cube=da,
    times_ok_ns = times_ok_ns,
)

chunk_processed_time = time.time()

for chunk_idx, chunk in enumerate(dataset):
    mae_sum = 0.0
    mape_sum = 0.0
    count = 0
    start_time = time.time()
    print(f'Chunk {chunk_idx} received in {start_time-chunk_processed_time:.2f} seconds')
    processed_data, coords, valid_mask, time_gaps_s2, = chunk

    if processed_data is None: N = 0
    else:
        N = processed_data.shape[0]

        center_time, center_xs, center_ys = extract_center_coordinates(coords)
        if current_time is None:
            current_time = center_time


    for start in tqdm(range(0, N, batch_size), desc="Reconstructing", unit="batch"):
        end = min(start + batch_size, N)

        # slice + move to device
        batch_processed = processed_data[start:end].to(device, dtype=torch.float32)
        batch_mask = valid_mask[start:end].to(device, dtype=torch.bool)
        batch_s2 = time_gaps_s2[start:end].to(device, dtype=torch.int32)
        #y_all, zf = model(batch_processed, batch_s2)
        try:
            y_all, zf = model(batch_processed, batch_s2)
        except Exception as e:
            print(str(e))
            print(batch_s2.shape)
            print(batch_processed.shape)

        # --- central coordinate ---
        B, T, C, H, W = batch_processed.shape
        ct, cx, cy = T // 2, H // 2, W // 2

        central_in = batch_processed[:, ct, :, cx, cy]  # [B, C]
        central_out = y_all[:, ct, :, cx, cy]  # [B, C]
        central_mask = batch_mask[:, ct, :, cx, cy]  # [B, C] (bool)

        # ---- write predictions to ds_pred at correct (band,time,y,x) ----
        # center_xs/center_ys are aligned to patches globally; take the batch slice
        bx = center_xs[start:end]  # length B
        by = center_ys[start:end]  # length B

        x_idx = coord_to_idx(bx, x_to_idx, global_xs)
        y_idx = coord_to_idx(by, y_to_idx, global_ys)

        # then vectorized write
        current_canvas[:, y_idx, x_idx] = zf.detach().cpu().numpy().astype(np.float32).T
        filled_once[y_idx, x_idx] = True

        # move predicted central values to CPU/np
        central_out_np = central_out.detach().cpu().numpy().astype(np.float32)
        central_in_np = central_in.detach().cpu().numpy().astype(np.float32)

        # filter only valid entries
        valid_in = central_in[central_mask]
        valid_out = central_out[central_mask]

        diff = (valid_out - valid_in).abs()
        mae_sum += diff.sum().item()
        mape_sum += (diff / valid_in.abs().clamp_min(eps)).sum().item()
        count += valid_mask[:, ct, :, cx, cy].sum().item()
    chunk_processed_time = time.time()
    print(f'Chunk {dataset.chunk_idx} ({cube_num}) processed in {chunk_processed_time - start_time:.3f} s')

    start_global_metrics = time.time()
    # global center metrics
    chunk_mae = mae_sum / max(count, 1)
    chunk_mape = 100.0 * mape_sum / max(count, 1)

    # Your rule: after every 16 chunks, flush the frame
    if (dataset.chunk_idx + 1) % 16 == 0:
        start_flash = time.time()

        if dataset.save_frame:
            saved = flush_frame(canvas=current_canvas, f_ds=ds0, out_path=output_path, time=current_time)
            print(f"🗂️ Frame {np.datetime_as_string(current_time, unit='D')} "
                  f"{'saved' if saved else 'skipped (<5% coverage)'}.")
            reset_frame()
        else: dataset.save_frame = True
        print(f'Flash frame in {time.time() - start_flash:.3f} seconds')

    dataset.chunk_idx += 1
    print(f"✅ Central-pixel Chunk MAE:  {chunk_mae:.6f}")
    print(f"✅ Central-pixel Chunk MAPE: {chunk_mape:.4f}%")
    print(f"✅ Global metrics computed in {time.time() - start_global_metrics:.3f} seconds")

    print(f"✅ Iteration ended in {time.time() - start_time:.3f} seconds\n")






# README  
## Script Usage

### **Parameters to set**
Inside the script, configure:

```
CUDA_DEVICE     = 'cuda:2'                                               # GPU Device used for inference
CUBE_IDS        = '017'                                                  # ID of the ScienceCube you want to process
BATCH_SIZE      = 500                                                    # Model inference batch size
BASE_PATH       = '/net/data/deepfeatures/sciencecubes'                  # location of stored ScienceCube
OUTPUT_PATH     = '/net/data/deepfeatures/feature'                       # FeatureCubes output path
CHECKPOINT_PATH = "../checkpoints/ae-epoch=141-val_loss=4.383e-03.ckpt"  # model weights location
PROCESSES       = 6                                                      # number of processes for patch creation & processing

```

`BASE_PATH` must point to a directory that contains the input ScienceCubes:

```
<BASE_PATH>/<cube_id>.zarr
```

---

## **Input**
- Sentinel-2 ScienceCubes stored as Zarr datasets  
  Example:  
  ```
  /net/data/deepfeatures/sciencecubes/017.zarr
  /net/data/deepfeatures/sciencecubes/039.zarr
  ```

Each cube should include `s2l2a`, `cloud_mask`, and standard `time/y/x` coordinates.

---

## **Output**
For `CUBE_ID`, the script generates:

```
<OUTPUT_PATH>/feature_<cube_id>.zarr
```

This output contains:

- Variable: `features`
- Shape: `(7, time, y, x)`
- The 7-dimensional latent features reconstructed for each valid timestamp.

Only timestamps with sufficient valid pixels are included; all other positions contain `NaN`.

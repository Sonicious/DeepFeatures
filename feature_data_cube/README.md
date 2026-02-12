# README  
## Script Usage

### **Parameters to set**
Defaults are defined in the script and can be overridden by CLI args.

```
CUDA_DEVICE           = 'cuda:3'                                                                           # GPU Device used for inference
CUBE_ID                    = '061'                                                                                # ID of the ScienceCube you want to process
BATCH_SIZE              = 550                                                                                 # Model inference batch size
BASE_PATH               = '/net/data/deepfeatures/science/0.1.0'                             # location of stored ScienceCube
OUTPUT_PATH          = '/net/data/deepfeatures/feature'                                      # FeatureCubes output path
CHECKPOINT_PATH  = "../checkpoints/ae-epoch=141-val_loss=4.383e-03.ckpt"  # model weights location
PROCESSES              = 6                                                                                     # number of processes for patch creation & processing
SPLIT_COUNT           = 1                                                                                      # total number of parallel runs
SPLIT_INDEX             = 0                                                                                     # which split this run should process 
                                                                                                                           # (SPLIT_INDEX=0: run first 50% of timestamps,
                                                                                                                           #  SPLIT_INDEX=1 run second 50% of timestamps)
SPACE_BLOCK_SIZE = 125                                                                                   # divides chunk of 11 frames  into Y/125 × X/125 spatial subchunks for patch generation
LOG_LEVEL        = 'INFO'                                                                                     # logging level (DEBUG, INFO, WARNING, ERROR)
```

`BASE_PATH` must point to a directory that contains the input ScienceCubes:

```
<BASE_PATH>/<cube_id>.zarr
```

---

### **CLI overrides**
You can override defaults via CLI:

```
python feature_data_cube.py \
  --cuda-device cuda:3 \
  --cube-id 061 \
  --batch-size 550 \
  --base-path /net/data/deepfeatures/science/0.1.0 \
  --output-path /net/data/deepfeatures/feature \
  --checkpoint-path ../checkpoints/ae-epoch=141-val_loss=4.383e-03.ckpt \
  --processes 6 \
  --split-count 1 \
  --split-index 0 \
  --space_block_size 125 \
  --log-level INFO
```

## **Input**
- Sentinel-2 ScienceCubes stored as Zarr datasets  
  Example:  
  ```
  /net/data/deepfeatures/science/0.1.0/017.zarr
  /net/data/deepfeatures/science/0.1.0/039.zarr
  ```

Each cube should include `s2l2a`, `cloud_mask`, and standard `time/y/x` coordinates.

---

## **Output**
For `CUBE_ID`, the script generates:

```
<OUTPUT_PATH>/<cube_id>.zarr
```

This output contains:

- Variable: `features`
- Shape: `(6, time, y, x)`
- The 6-dimensional latent features reconstructed for each valid timestamp.

Only timestamps with sufficient valid pixels are included; all others are discarded.

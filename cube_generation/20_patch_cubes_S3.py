import fsspec
from xcube.core.store import new_data_store
import zarr


AWS_ACCESS_KEY_ID = "xxx"
AWS_SECRET_ACCESS_KEY = "xxx"
AWS_BUCKET = "hub-deepesdl37"

store = new_data_store(
    "s3",
    root=AWS_BUCKET,
    max_depth=10,
    storage_options=dict(
        anon=False,
        key=AWS_ACCESS_KEY_ID,
        secret=AWS_SECRET_ACCESS_KEY,
    ),
)

data_ids = store.list_data_ids()
data_ids_sel = [data_id for data_id in data_ids if "cubes/science/0.1.0" in data_id]
print(f"Found {len(data_ids_sel)} datasets")

for data_id in data_ids_sel[1:]:
    path = f"s3://{AWS_BUCKET}/{data_id}"
    print(f"Updating: {path}")

    mapper = fsspec.get_mapper(
        path,
        key=AWS_ACCESS_KEY_ID,
        secret=AWS_SECRET_ACCESS_KEY,
        anon=False,
    )

    zarr_store = zarr.open_group(mapper, mode="r+")
    var = zarr_store["lccs_class"]
    var.attrs["valid_max"] = 220
    var.attrs["valid_min"] = 1
    
    # Optional: consolidate metadata if store uses it
    try:
        zarr.convenience.consolidate_metadata(mapper)
        print("  🔄 Updated attributes of lccs_class")
    except Exception as e:
        print(f"  ⚠️ Error: {e}")

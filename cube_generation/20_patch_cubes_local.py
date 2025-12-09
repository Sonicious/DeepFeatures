import os
import zarr

LOCAL_ROOT = "PATH to your local zarr stores"

# find all zarr stores in LOCAL_ROOT
zarr_stores = sorted(
    os.path.join(LOCAL_ROOT, d)
    for d in os.listdir(LOCAL_ROOT)
    if d.endswith(".zarr") and os.path.isdir(os.path.join(LOCAL_ROOT, d))
)


print(f"Gefundene Zarr-Stores: {len(zarr_stores)}")

for store_path in zarr_stores:
    print(f"🔧 Updating: {store_path}")

    # Opening the Zarr store
    try:
        z = zarr.open_group(store_path, mode="r+")
    except Exception as e:
        print(f"  ❌ Error opening: {e}")
        continue

    # Check if the variable exists
    if "lccs_class" not in z:
        print("  ⚠️ Variable 'lccs_class' not found. continue with next.")
        continue

    # Setting attributes
    var = z["lccs_class"]
    var.attrs["valid_min"] = 1
    var.attrs["valid_max"] = 220
    print("  ✔️ updated metadata")

    # Optional: consolidate metadata
    try:
        zarr.convenience.consolidate_metadata(store_path)
        print("  🔄 Metadata konsolidiert")
    except UnicodeDecodeError as e:
        print(f"  ⚠️ Skipping consolidate: {e}")
    except Exception as e:
        print(f"  ⚠️ Other consolidate error: {e}")
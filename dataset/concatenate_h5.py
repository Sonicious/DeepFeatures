import h5py
import numpy as np
import random

# Define file paths
#train_subset_file = "train_norm_11_15_15.h5"
train_subset_file = "hainich_149_train.h5"
#val_subset_file = "val_norm_11_15_15.h5"
val_subset_file = "./hainich_149_val.h5"

# Define output files
#output_train_file = "train_combined.h5"
output_train_file = "./train_149.h5"
#output_val_file = "val_combined2.h5"
output_val_file = "./val_149.h5"

# Set maximum chunk size
CHUNK_SIZE = 5000  # Process data in batches of 5000 to reduce memory usage


def append_hdf5_subset(output_file, subset_file, subset_size, chunk_size=CHUNK_SIZE):
    with h5py.File(output_file, "a") as f_out, h5py.File(subset_file, "r") as f_subset:

        # Get total samples in the subset file
        total_samples = f_subset[list(f_subset.keys())[0]].shape[0]

        # Randomly select subset_size samples
        if subset_size and subset_size < total_samples:
            selected_indices = sorted(
                random.sample(range(total_samples), subset_size))  # Sort to improve read performance
        else:
            selected_indices = np.arange(total_samples)  # If subset_size is None or larger, use all

        # Process in chunks
        for key in f_subset.keys():
            dataset_shape = f_subset[key].shape[1:]  # Shape excluding batch dimension
            dtype = f_subset[key].dtype

            # If dataset does not exist in output file, create it with an expandable shape
            if key not in f_out:
                f_out.create_dataset(
                    key, shape=(0, *dataset_shape), maxshape=(None, *dataset_shape), dtype=dtype, compression="gzip"
                )

            # Append data in chunks of at most CHUNK_SIZE
            for i in range(0, len(selected_indices), chunk_size):
                batch_indices = selected_indices[i: i + chunk_size]
                batch_data = f_subset[key][batch_indices, ...]

                # Expand dataset size
                f_out[key].resize((f_out[key].shape[0] + batch_data.shape[0]), axis=0)
                f_out[key][-batch_data.shape[0]:] = batch_data



# Append all data from train_norm_11_15_15.h5 to train_combined.h5

# Append 4000 randomly selected samples from val_norm_11_15_15.h5 to val_combined.h5
#append_hdf5_subset(output_val_file, val_subset_file, subset_size=1400)
#append_hdf5_subset(output_val_file, val_subset_file, subset_size=4000)
#print('combined validation sets')

#append_hdf5_subset(output_train_file, train_subset_file, subset_size=10500)
append_hdf5_subset(output_train_file, train_subset_file, subset_size=14000)
print('combined training sets')


print("Subset data successfully appended to the existing combined datasets!")



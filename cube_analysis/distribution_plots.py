import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns  # Optional for better-looking plots
from dataset.si_dataset import ds
import numpy as np

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the full path to the pickle file
pickle_file_path = os.path.join(script_dir, "98_percentile.pkl")

# Verify if the pickle file exists
if not os.path.exists(pickle_file_path):
    raise FileNotFoundError(f"Pickle file not found: {pickle_file_path}")

# Load the percentile bounds from the pickle file
with open(pickle_file_path, "rb") as f:
    percentile_bounds = pickle.load(f)

# Directory to save the plots
output_dir = "../distribution_plots"
#output_dir = "dist_plots"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Create and save distribution plots for each variable
for var_name in ds.data_vars:
    output_path = os.path.join(output_dir, f"{var_name}_distribution.png")
    if os.path.exists(output_path):
        print(f"{var_name}_distribution.png exists !!")
        continue
    else:
        print( f"creating {var_name}_distribution.png")

    # Extract the data and flatten it
    data = ds[var_name].values
    lower_bound, upper_bound = percentile_bounds.get(var_name, [None, None])

    # Apply the bounds: set values outside [lower_bound, upper_bound] to NaN
    if lower_bound is not None and upper_bound is not None:
        filtered_data = np.where((data >= lower_bound) & (data <= upper_bound), data, np.nan)
    else:
        print(f"Bounds not found for {var_name}, skipping.")
        continue

    # Flatten the data for plotting
    flat_data = filtered_data.flatten()
    flat_data = flat_data[~np.isnan(flat_data)]  # Remove NaN values

    # Plot the distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(flat_data, bins=150, color="blue", label=f"Filtered Distribution of {var_name}", stat="count")
    plt.title(f"Filtered Distribution Plot for {var_name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    # Save the plot
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()  # Close the figure to free up memory

print(f"Plots saved in '{output_dir}' directory.")


import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns  # Optional for better-looking plots
import os
from si_dataset import ds
import numpy as np



# Directory to save the plots
output_dir = "distribution_plots"
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
    data = ds[var_name].values.flatten()  # Flatten the data to 1D
    data = data[~np.isnan(data)]  # Remove NaN values
    plt.figure(figsize=(8, 6))
    #sns.histplot(data, kde=True, bins=30, color="blue", label=f"Distribution of {var_name}")
    sns.histplot(data, bins=150, color="blue", label=f"Distribution of {var_name}", stat="count")
    plt.title(f"Distribution Plot for {var_name}")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    # Save the plot
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()  # Close the figure to free up memory

print(f"Plots saved in '{output_dir}' directory.")

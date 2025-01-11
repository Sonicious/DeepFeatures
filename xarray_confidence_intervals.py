from si_dataset import ds
import numpy as np
import pickle
from scipy.stats import norm  # Import for confidence level calculation

# Load the xarray dataset (replace with actual dataset path or data)
# Example synthetic dataset for demonstration


# Function to compute confidence interval
def compute_confidence_interval(data, confidence=0.95):
    data = data[~np.isnan(data)]  # Remove NaN values
    n = len(data)
    if n == 0:  # Handle case where all values are NaN
        return [np.nan, np.nan]

    mean = np.mean(data)
    stderr = np.std(data, ddof=1) / np.sqrt(n)  # Standard error
    z_score = norm.ppf(1 - (1 - confidence) / 2)  # Compute Z-score for confidence level
    margin = stderr * z_score  # Margin of error (1.96)
    return [mean - margin, mean + margin]

# Compute confidence intervals for all variables
confidence_intervals = {}
for var in ds.data_vars:
    print(f'compute interval for {var}')
    flat_data = ds[var].values.flatten()  # Flatten the data for computation
    confidence_intervals[var] = compute_confidence_interval(flat_data)

# Pickle the result
output_path = "./confidence_intervals.pkl"
with open(output_path, "wb") as f:
    pickle.dump(confidence_intervals, f)

print(f"Confidence intervals saved to: {output_path}")

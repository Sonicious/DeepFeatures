from si_dataset import ds
import numpy as np
import pickle
from scipy.stats import norm  # Import for confidence level calculation

# Load the xarray dataset (replace with actual dataset path or data)
# Example synthetic dataset for demonstration


# Function to compute confidence interval
def compute_confidence_interval(data, confidence=0.95):
    print(type(data))
    data = data[~np.isnan(data)]  # Remove NaN values
    n = len(data)
    if n == 0:  # Handle case where all values are NaN
        return [np.nan, np.nan]

    mean = np.mean(data)
    std =  np.std(data)
    stderr = std / np.sqrt(n)  # Standard error    print(mean)
    print(f'mean: {mean}, std: {std}, stderr: {stderr}')
    z_score = norm.ppf(1 - (1 - confidence) / 2)  # Compute Z-score for confidence level
    margin = stderr * z_score  # Margin of error (1.96)

    return [mean - margin, mean + margin]

def compute_iqr_margins(data):
    """
    Compute margins for identifying outliers based on IQR.

    Args:
        data (array-like): Input data.

    Returns:
        tuple: Lower bound, upper bound.
    """
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values

    # Calculate Q1, Q3, and IQR
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # Compute the margins
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return [lower_bound, upper_bound]


def compute_bounds(data):
    """
    Compute bounds for identifying outliers based on percentiles.

    Args:
        data (array-like): Input data.

    Returns:
        list: Lower bound (2.5th percentile), upper bound (97.5th percentile).
    """
    data = np.array(data)
    data = data[~np.isnan(data)]  # Remove NaN values from the input data

    # Calculate the 2.5th percentile (lower bound) and 97.5th percentile (upper bound)
    lower_bound = np.percentile(data, 1.5)  # Value below which 1.5% of data falls
    upper_bound = np.percentile(data, 98.5)  # Value below which 98.5% of data falls

    return [lower_bound, upper_bound]


# Compute confidence intervals for all variables
confidence_intervals = {}
for var in ds.data_vars:
    print(f'compute interval for {var}')
    flat_data = ds[var].values.flatten()  # Flatten the data for computation
    confidence_intervals[var] = compute_bounds(flat_data)
    print(confidence_intervals[var])

# Pickle the result
output_path = "./98_percentile.pkl"
with open(output_path, "wb") as f:
    pickle.dump(confidence_intervals, f)

print(f"Confidence intervals saved to: {output_path}")

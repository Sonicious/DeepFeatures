from typing import Dict, List
import matplotlib.pyplot as plt
import pickle


def create_boxplots_from_min_max(min_max_dict: Dict[str, List[float]], output_file_min: str = "boxplot_mins.png", output_file_max: str = "boxplot_maxs.png"):

    """
    Create and save separate boxplots for the min and max values with filtering applied:
    - Remove all min values smaller than -3.
    - Remove all max values larger than 2.

    Args:
        min_max_dict (Dict[str, List[float]]): Dictionary with variable names and their min/max values.
        output_file_min (str): Path to save the boxplot for filtered min values.
        output_file_max (str): Path to save the boxplot for filtered max values.
    """
    # Extract and filter min and max values
    mins = [values[0] for values in min_max_dict.values() if values[0] <= 10]
    print('num mins: ', len(mins))
    maxs = [values[1] for values in min_max_dict.values() if values[1] <= 25]
    print('num maxs: ', len(maxs))

    # Create filtered boxplot for mins
    plt.figure(figsize=(6, 10))
    plt.boxplot(mins, vert=True, patch_artist=True, labels=["Min Values"])
    plt.title("Boxplot of Minimum Values")
    plt.ylabel("Values")
    plt.savefig(output_file_min)
    plt.close()

    # Create filtered boxplot for maxs
    plt.figure(figsize=(6, 10))
    plt.boxplot(maxs, vert=True, patch_artist=True, labels=["Max Values"])
    plt.title("Boxplot of Maximum Values")
    plt.ylabel("Values")
    plt.savefig(output_file_max)
    plt.close()


def create_median_boxplot(median_dict: Dict[str, float],
                          output_file: str = "median_boxplot.png"):
    """
    Create and save a boxplot from the median values without removing outliers.

    Args:
        median_dict (Dict[str, float]): Dictionary with variable names and their median values.
        output_file (str): Path to save the median boxplot.
    """
    # Extract median values
    #medians = list(median_dict.values())
    medians = [value for value in median_dict.values() if value ]

    # Create the boxplot
    plt.figure(figsize=(6, 10))
    plt.boxplot(medians, vert=True, patch_artist=True, labels=["Median Values"])
    plt.title("Boxplot of Median Values")
    plt.ylabel("Values")
    plt.savefig(output_file)
    plt.close()



# Example min/max dictionary
with open('stats.pkl', "rb") as f:
    stats = pickle.load(f)
    print(stats)

# Generate the boxplots
create_boxplots_from_min_max(
    stats,
    output_file_min="filtered_boxplot_mins.png",
    output_file_max="filtered_boxplot_maxs.png"
)
#
#with open('median.pkl', "rb") as f:
#    stats = pickle.load(f)
#    print(stats)
#
#
#create_median_boxplot(stats, output_file="median_boxplot.png")#
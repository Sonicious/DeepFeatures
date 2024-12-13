import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from pathlib import Path
import os

import constants

_DIR = Path(__file__).parent.resolve()
sites_params = pd.read_csv(constants.PATH_SITES_PARAMETERS_TRAINING)

# Define latitude and longitude points
y = sites_params["center_lat"].values
x = sites_params["center_lon"].values

points = [{"latitude": y[i], "longitude": x[i]} for i in range(len(x))]

# Create a map
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_extent([-2, 32, 40, 65], crs=ccrs.PlateCarree())

# Add features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.COASTLINE)

# Plot points
for point in points:
    ax.plot(
        point["longitude"],
        point["latitude"],
        "ro",
        markersize=5,
        transform=ccrs.PlateCarree(),
    )
# Add gridlines and labels
gridlines = ax.gridlines(
    draw_labels=True,
    crs=ccrs.PlateCarree(),
    color="gray",
    linestyle="--",
    linewidth=0.5,
)
gridlines.top_labels = False  # Turn off the labels on the top axis
gridlines.right_labels = False  # Turn off the labels on the right axis

# Add axis labels
ax.set_xlabel("Longitude [°]")
ax.set_ylabel("Latitude [°]")

plt.tight_layout()
plt.savefig(os.path.join(_DIR, "plots", "traincubes_location"))
plt.close()

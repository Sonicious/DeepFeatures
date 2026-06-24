import geopandas as gpd
from shapely.ops import unary_union

# Read the input files
gdf1 = gpd.read_file("data/51N.geojson")
gdf2 = gpd.read_file("data/51S.geojson")

# Merge geometries
merged_geom = unary_union(
    list(gdf1.geometry) + list(gdf2.geometry)
)

# Create output GeoDataFrame with desired attributes
result = gpd.GeoDataFrame(
    {
        "landuse": ["quarry"],
        "name": ["Profen-Nord"],
        "operator": ["MIBRAG"],
        "resource": ["coal;lignite;braunkohle"],
    },
    geometry=[merged_geom],
    crs=gdf1.crs,
)

# Write GeoJSON
result.to_file("data/51.geojson", driver="GeoJSON")
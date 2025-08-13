import numpy as np
import rasterio
from rasterio.transform import from_bounds
import os
import requests
from scipy.interpolate import griddata
from shapely.geometry import box
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

center = [41.6086, 21.7453]
south, north, west, east = center[0] - 1.5, center[0] + 1.5, center[1] - 1.5, center[1] + 1.5

# Define grid resolution (e.g., 0.01 degrees, roughly 1.1 km resolution)
resolution_deg = 0.01

lon_grid = np.arange(west, east + resolution_deg, resolution_deg)
lat_grid = np.arange(south, north + resolution_deg, resolution_deg)

lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

points_for_api = []
for lat_val, lon_val in zip(lat_mesh.flatten(), lon_mesh.flatten()):
    points_for_api.append({"latitude": lat_val, "longitude": lon_val})

API_URL = "https://api.open-elevation.com/api/v1/lookup"
MAX_BATCH_SIZE = 10000

elevation_values = []
total_points = len(points_for_api)
print(f"Querying {total_points} points for elevation data from Open-Elevation API...")

nodata_value = -9999  # Define nodata value for consistency

# Calculate number of batches
num_batches = (total_points + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE

for i in tqdm(range(0, total_points, MAX_BATCH_SIZE), total=num_batches, desc="Fetching elevations"):
    batch = points_for_api[i:i + MAX_BATCH_SIZE]
    payload = {"locations": batch}
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        for loc in data['results']:
            elevation_values.append(loc['elevation'])
        # Removed print inside loop; tqdm will handle progress display
    except requests.exceptions.RequestException as e:
        print(f"Error fetching elevation batch (points {i} to {i + len(batch)}): {e}")
        elevation_values.extend([nodata_value] * len(batch))

if not elevation_values or all(e == nodata_value for e in elevation_values):
    print("No valid elevation data retrieved from API. Cannot generate raster.")
else:
    points_coords = np.array([(p['longitude'], p['latitude']) for p in points_for_api[:len(elevation_values)]])
    elevations_array = np.array(elevation_values)

    filename = 'raster_south=%.6f_north=%.6f_west=%.6f_east=%.6f_res=%.6f' % (
    south, north, west, east, resolution_deg)
    np.savez(filename,
             elevations=np.flipud(elevations_array.reshape(lon_mesh.shape)),
             south=south,
             north=north,
             west=west,
             east=east,
             resolution=resolution_deg)

    elev = np.flipud(elevations_array.reshape(lon_mesh.shape))
    min_val, max_val = elev.min(), elev.max()
    scaled = ((elev - min_val) / (max_val - min_val) * 255).astype(np.uint8) if max_val > min_val else np.zeros_like(
        elev, dtype=np.uint8)
    Image.fromarray(scaled).save(filename.rsplit('.', 1)[0] + '.jpg', quality=95)

    print("Interpolating elevation data to raster grid...")

    # target_grid_points = np.column_stack((lon_mesh.ravel(), lat_mesh.ravel()))
    #
    # try:
    #     interpolated_elevations = griddata(
    #         points_coords,
    #         elevations_array,
    #         target_grid_points,
    #         method='linear'
    #     )
    #
    #     elevation_data_2d = interpolated_elevations.reshape(lat_mesh.shape)
    #     elevation_data_2d[np.isnan(elevation_data_2d)] = nodata_value
    #
    #     output_raster_file = 'elevation_raster_interpolated.tif'
    #
    #     transform = from_bounds(west, south, east, north, elevation_data_2d.shape[1], elevation_data_2d.shape[0])
    #
    #     profile = {
    #         'driver': 'GTiff', 'height': elevation_data_2d.shape[0], 'width': elevation_data_2d.shape[1],
    #         'count': 1, 'dtype': elevation_data_2d.dtype, 'crs': 'EPSG:4326', 'transform': transform,
    #         'nodata': nodata_value
    #     }
    #
    #     print(f"Writing final interpolated elevation data to {output_raster_file}...")
    #     with rasterio.open(output_raster_file, "w", **profile) as dest:
    #         dest.write(elevation_data_2d, 1)
    #
    #     print(f"Elevation raster successfully generated and saved to {output_raster_file}")
    # except Exception as e:
    #     print(f"Error during interpolation or writing raster: {e}")

print("Done!")

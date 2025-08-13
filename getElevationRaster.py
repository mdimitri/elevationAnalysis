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

# center = [41.6086, 21.7453]
# south, north, west, east = center[0] - 1.5, center[0] + 1.5, center[1] - 1.5, center[1] + 1.5

west = 20.4529023-.05; south = 40.852478-.05; east = 23.034051+.05; north = 42.3739044+.05

# Define grid resolution (e.g., 0.01 degrees, roughly 1.1 km resolution)
resolution_deg = 0.002

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


print("Done!")

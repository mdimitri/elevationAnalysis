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
import cv2

cv2.namedWindow("Elevation Progress", cv2.WINDOW_NORMAL)

# center = [41.6086, 21.7453]
# south, north, west, east = center[0] - 1.5, center[0] + 1.5, center[1] - 1.5, center[1] + 1.5

west = 20.4529023-.05; south = 40.852478-.05; east = 23.034051+.05; north = 42.3739044+.05

# Define grid resolution (e.g., 0.01 degrees, roughly 1.1 km resolution)
resolution_deg = 0.0005

# Full grid size
lon_grid = np.arange(west, east + resolution_deg, resolution_deg)
lat_grid = np.arange(south, north + resolution_deg, resolution_deg)

width = len(lon_grid)
height = len(lat_grid)

# Define tile grid size (e.g., 16x16)
tile_cols = 32
tile_rows = 32

tile_width = width // tile_cols
tile_height = height // tile_rows

nodata_value = -9999

API_URL = "https://api.open-elevation.com/api/v1/lookup"
MAX_BATCH_SIZE = 10000

# Initialize full elevation array with nodata
elevations_full = np.full((height, width), nodata_value, dtype=np.float32)

def query_elevation(points):
    try:
        response = requests.post(API_URL, json={"locations": points}, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [loc['elevation'] for loc in data['results']]
    except Exception as e:
        print(f"Error fetching elevation data: {e}")
        return [nodata_value] * len(points)

for row in tqdm(range(tile_rows), desc="Tiles rows"):
    for col in tqdm(range(tile_cols), desc="Tiles cols", leave=False):
        # Compute tile boundaries in indexes
        x_start = col * tile_width
        x_end = x_start + tile_width
        if col == tile_cols - 1:  # last tile takes rest if uneven
            x_end = width

        y_start = row * tile_height
        y_end = y_start + tile_height
        if row == tile_rows - 1:
            y_end = height

        # Build points for tile
        tile_lons = lon_grid[x_start:x_end]
        tile_lats = lat_grid[y_start:y_end]

        tile_lon_mesh, tile_lat_mesh = np.meshgrid(tile_lons, tile_lats)
        tile_points = [{"latitude": lat, "longitude": lon} for lat, lon in zip(tile_lat_mesh.flatten(), tile_lon_mesh.flatten())]

        # Because API max batch size is 10k, split if needed
        tile_elevations = []
        for i in range(0, len(tile_points), MAX_BATCH_SIZE):
            batch = tile_points[i:i + MAX_BATCH_SIZE]
            batch_elev = query_elevation(batch)
            tile_elevations.extend(batch_elev)

        if len(tile_elevations) != (y_end - y_start) * (x_end - x_start):
            print(f"Warning: tile {row},{col} elevation count mismatch")

        # Fill the tile part in the full array
        tile_elev_array = np.array(tile_elevations).reshape((y_end - y_start, x_end - x_start))
        elevations_full[y_start:y_end, x_start:x_end] = tile_elev_array

        # Visualize current progress
        valid_mask = elevations_full != nodata_value
        elev_min = np.min(elevations_full[valid_mask]) if np.any(valid_mask) else 0
        elev_max = np.max(elevations_full[valid_mask]) if np.any(valid_mask) else 1

        # Scale for visualization
        scaled = np.zeros_like(elevations_full, dtype=np.uint8)
        if elev_max > elev_min:
            scaled[valid_mask] = ((elevations_full[valid_mask] - elev_min) / (elev_max - elev_min) * 255).astype(np.uint8)

        # Flip vertically for north-up
        img = np.flipud(scaled)

        # Get current window size
        x, y, win_w, win_h = cv2.getWindowImageRect("Elevation Progress")

        # Resize image to current window size while keeping aspect ratio
        img_h, img_w = img.shape
        scale_w = win_w / img_w
        scale_h = win_h / img_h
        scale = min(scale_w, scale_h, 1.0)  # avoid scaling up beyond window size

        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        img_to_show = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Show the resized image
        cv2.imshow("Elevation Progress", img_to_show)
        cv2.waitKey(1)

        # Save final results
        filename = 'raster_south=%.6f_north=%.6f_west=%.6f_east=%.6f_res=%.6f' % (
        south, north, west, east, resolution_deg)
        np.savez(filename,
                 elevations=elevations_full,
                 south=south,
                 north=north,
                 west=west,
                 east=east,
                 resolution=resolution_deg)

        # Save visualization as JPG
        scaled_img = ((elevations_full - elev_min) / (elev_max - elev_min) * 255).astype(np.uint8)
        Image.fromarray(np.flipud(scaled_img)).save(filename + '.jpg', quality=95)

cv2.destroyAllWindows()

# Save final results
filename = 'raster_south=%.6f_north=%.6f_west=%.6f_east=%.6f_res=%.6f' % (south, north, west, east, resolution_deg)
np.savez(filename,
         elevations=elevations_full,
         south=south,
         north=north,
         west=west,
         east=east,
         resolution=resolution_deg)

# Save visualization as JPG
scaled_img = ((elevations_full - elev_min) / (elev_max - elev_min) * 255).astype(np.uint8)
Image.fromarray(np.flipud(scaled_img)).save(filename + '.jpg', quality=95)

print("Done!")

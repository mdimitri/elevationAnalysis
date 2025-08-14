import requests
import math
from PIL import Image
import numpy as np
import os
from tqdm.auto import tqdm  # Import tqdm for progress bars


def latlon_to_tile(lat, lon, zoom):
    """Converts latitude and longitude to ZXY tile coordinates."""
    n = 2 ** zoom
    tile_x = int(math.floor((lon + 180) / 360 * n))
    tile_y = int(
        math.floor((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n))
    return tile_x, tile_y


def tile_to_latlon(tile_x, tile_y, zoom):
    """Converts ZXY tile coordinates to the bounding box of the tile."""
    n = 2.0 ** zoom
    lon_deg = tile_x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def get_elevation_from_terrarium(r, g, b):
    """Decodes elevation from Terrarium PNG RGB values: (R * 256 + G + B / 256) - 32768."""
    return (r * 256 + g + b / 256) - 32768


def fetch_and_process_heightmap(west, south, east, north, zoom=13, output_dir="heightmap_data"):
    """
    Fetches Mapzen-derived Terrarium PNG tiles, processes them into elevation data,
    and saves as a 16-bit TIFF heightmap, a NumPy array, and a JPEG visual.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Calculate bounding tile coordinates for the corners of the geographic area
    tile_x_nw, tile_y_nw = latlon_to_tile(north, west, zoom)
    tile_x_se, tile_y_se = latlon_to_tile(south, east, zoom)

    # Determine the overall min/max tile indices for X and Y across the bounding box
    min_tile_x = min(tile_x_nw, tile_x_se)
    max_tile_x = max(tile_x_nw, tile_x_se)
    min_tile_y = min(tile_y_nw, tile_y_se)
    max_tile_y = max(tile_y_nw, tile_y_se)

    print(
        f"Processing tiles for zoom {zoom} (X range: {min_tile_x}-{max_tile_x}, Y range: {min_tile_y}-{max_tile_y})...")

    tile_size = 256
    num_cols = max_tile_x - min_tile_x + 1
    num_rows = max_tile_y - min_tile_y + 1

    # Initialize the NumPy array to store raw elevation data
    elevation_data_array = np.zeros((num_rows * tile_size, num_cols * tile_size), dtype=np.float32)

    base_url = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"

    # Use tqdm for a progress bar during tile fetching
    total_tiles = num_cols * num_rows
    tile_coords = [(x, y) for x in range(min_tile_x, max_tile_x + 1) for y in range(min_tile_y, max_tile_y + 1)]

    for x_idx, y_idx, (x, y) in tqdm(
            [(xi, yi, (x, y)) for xi, x in enumerate(range(min_tile_x, max_tile_x + 1))
             for yi, y in enumerate(range(min_tile_y, max_tile_y + 1))],
            total=total_tiles, desc="Downloading & Processing Tiles"
    ):
        tile_url = f"{base_url}/{zoom}/{x}/{y}.png"
        tile_path = os.path.join(output_dir,
                                 f"{zoom}_{x}_{y}.png")  # Still save individual tiles for debugging if needed

        try:
            response = requests.get(tile_url, stream=True)
            response.raise_for_status()
            with open(tile_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            img = Image.open(tile_path).convert("RGB")

            # Extract elevation data directly into the main array
            for py in range(tile_size):
                for px in range(tile_size):
                    r, g, b = img.getpixel((px, py))
                    elevation = get_elevation_from_terrarium(r, g, b)
                    elevation_data_array[y_idx * tile_size + py, x_idx * tile_size + px] = elevation

        except requests.exceptions.RequestException as e:
            # Print error without halting progress bar
            tqdm.write(f"Error fetching/processing tile {x},{y} ({tile_url}): {e}")
        except Exception as e:
            tqdm.write(f"Unexpected error with tile {x},{y}: {e}")

    # Check if any data was successfully processed before reporting stats
    if np.all(elevation_data_array == 0) and (num_cols * num_rows > 0):
        print(
            "Warning: No elevation data was successfully processed. Check coordinates, zoom level, or internet connection.")
        return

    # Calculate geographical resolution for metadata
    # Resolution in degrees per pixel
    # Get the lat/lon of the top-left pixel and bottom-right pixel of the entire stitched array
    # This is an approximation. A more precise method would involve calculating it per tile
    # and averaging, but for a broad resolution estimate, this is sufficient.

    # Get top-left and bottom-right corner of the whole stitched map
    top_left_lat, top_left_lon = tile_to_latlon(min_tile_x, min_tile_y, zoom)
    bottom_right_lat, bottom_right_lon = tile_to_latlon(max_tile_x + 1, max_tile_y + 1,
                                                        zoom)  # +1 to get the actual end of the last tile

    west = top_left_lon
    south = bottom_right_lat
    east = bottom_right_lon
    north = top_left_lat

    # Approximate resolution in degrees/pixel
    # Use the overall extent of the collected tiles
    total_lon_span = abs(bottom_right_lon - top_left_lon)
    total_lat_span = abs(top_left_lat - bottom_right_lat)  # Lat decreases as y increases

    resolution_deg_lon = total_lon_span / (num_cols * tile_size)
    resolution_deg_lat = total_lat_span / (num_rows * tile_size)

    # For a single 'resolution' value, we can use the average or pick one.
    # Mercator projection means resolution changes with latitude, so this is an average.
    resolution_deg = (resolution_deg_lon + resolution_deg_lat) / 2

    # --- Generate common filename ---
    filename_base = os.path.join(f"raster_south={south:.6f}_north={north:.6f}_west={west:.6f}_east={east:.6f}_res={resolution_deg:.6f}")

    # --- Save Raw Elevation Data as NumPy zipped archive (.npz) ---
    # np.flipud is applied here to match the user's request for how data is stored in npz
    # elevations_array.reshape(lon_mesh.shape) is implied by the 2D array already
    npz_path = f"{filename_base}.npz"
    np.savez(npz_path,
             elevations=elevation_data_array,  # Flip for common GIS/image display orientation
             south=south,
             north=north,
             west=west,
             east=east,
             resolution=resolution_deg)
    print(f"Raw elevation data (NumPy .npz archive) saved: {npz_path}")

    # --- Save Visual JPEG ---
    # Flip array for visual consistency as JPGs often display with (0,0) at top-left
    elev_for_jpg = elevation_data_array
    min_val, max_val = elev_for_jpg.min(), elev_for_jpg.max()
    scaled_jpg = ((elev_for_jpg - min_val) / (max_val - min_val) * 255).astype(
        np.uint8) if max_val > min_val else np.zeros_like(
        elev_for_jpg, dtype=np.uint8)

    jpg_path = f"{filename_base}.jpg"
    Image.fromarray(scaled_jpg).save(jpg_path, quality=95)
    print(f"Visual JPEG image saved: {jpg_path}")

    # Display some stats
    print("\n--- Elevation Data Stats ---")
    print(f"Min Elevation: {np.min(elevation_data_array):.2f}m")
    print(f"Max Elevation: {np.max(elevation_data_array):.2f}m")
    print(f"Mean Elevation: {np.mean(elevation_data_array):.2f}m")
    print(f"Approx. Resolution: {resolution_deg:.6f} degrees/pixel")


# Define your bounding box
west = 20.4529023 - 0.05
south = 40.852478 - 0.05
east = 23.034051 + 0.05
north = 42.3739044 + 0.05

# Run the function
fetch_and_process_heightmap(west, south, east, north, zoom=10)

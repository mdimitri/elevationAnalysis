import requests
import math
from PIL import Image
import numpy as np
import os
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def fetch_tile(x_idx, y_idx, x, y, zoom, tile_size, base_url, max_retries=10):
    tile_url = f"{base_url}/{zoom}/{x}/{y}.png"
    for attempt in range(max_retries):
        try:
            response = requests.get(tile_url, stream=True, timeout=10)
            response.raise_for_status()
            img = Image.open(response.raw).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            elevation = (r * 256 + g + b / 256) - 32768
            return x_idx, y_idx, elevation
        except requests.exceptions.RequestException as e:
            tqdm.write(f"Attempt {attempt+1} failed for tile {x},{y} ({tile_url}): {e}")
        except Exception as e:
            tqdm.write(f"Attempt {attempt+1} unexpected error with tile {x},{y}: {e}")
    tqdm.write(f"Failed to fetch tile {x},{y} after {max_retries} attempts.")
    return None


def fetch_and_process_heightmap(west, south, east, north, zoom=13, output_dir="heightmap_data", max_workers=10):
    os.makedirs(output_dir, exist_ok=True)

    tile_x_nw, tile_y_nw = latlon_to_tile(north, west, zoom)
    tile_x_se, tile_y_se = latlon_to_tile(south, east, zoom)

    min_tile_x = min(tile_x_nw, tile_x_se)
    max_tile_x = max(tile_x_nw, tile_x_se)
    min_tile_y = min(tile_y_nw, tile_y_se)
    max_tile_y = max(tile_y_nw, tile_y_se)

    print(f"Processing tiles for zoom {zoom} (X range: {min_tile_x}-{max_tile_x}, Y range: {min_tile_y}-{max_tile_y})...")

    tile_size = 256
    num_cols = max_tile_x - min_tile_x + 1
    num_rows = max_tile_y - min_tile_y + 1

    elevation_data_array = np.zeros((num_rows * tile_size, num_cols * tile_size), dtype=np.float32)
    base_url = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"

    # Prepare all tile download tasks
    tasks = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for xi, x_tile in enumerate(range(min_tile_x, max_tile_x + 1)):
            for yi, y_tile in enumerate(range(min_tile_y, max_tile_y + 1)):
                tasks.append(executor.submit(fetch_tile, xi, yi, x_tile, y_tile, zoom, tile_size, base_url, max_retries=10))

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Downloading & Processing Tiles"):
            result = future.result()
            if result is not None:
                xi, yi, elevation = result
                elevation_data_array[yi * tile_size:(yi + 1) * tile_size,
                                     xi * tile_size:(xi + 1) * tile_size] = elevation

    if np.all(elevation_data_array == 0) and (num_cols * num_rows > 0):
        print("Warning: No elevation data was successfully processed.")
        return

    # Calculate bounds and resolution (pixel-centered)
    top_left_lat, top_left_lon = tile_to_latlon(min_tile_x, min_tile_y, zoom)
    bottom_right_lat, bottom_right_lon = tile_to_latlon(max_tile_x + 1, max_tile_y + 1, zoom)

    res_lon = (bottom_right_lon - top_left_lon) / (num_cols * tile_size)
    res_lat = (top_left_lat - bottom_right_lat) / (num_rows * tile_size)

    # Shift to pixel center
    top_left_lat -= (res_lat / 2)
    top_left_lon += (res_lon / 2)
    bottom_right_lat += (res_lat / 2)
    bottom_right_lon -= (res_lon / 2)

    filename_base = os.path.join(output_dir,
        f"heightmap_z{zoom}_lon_{top_left_lon:.4f}_{bottom_right_lon:.4f}_lat_{bottom_right_lat:.4f}_{top_left_lat:.4f}_reslon_{abs(res_lon):.6f}_reslat_{abs(res_lat):.6f}")

    # Save NPZ
    npz_path = f"{filename_base}.npz"
    np.savez(npz_path,
             elevations=elevation_data_array,
             north=top_left_lat,
             south=bottom_right_lat,
             west=top_left_lon,
             east=bottom_right_lon,
             resolution_lon_deg=res_lon,
             resolution_lat_deg=res_lat)
    print(f"Raw elevation data saved: {npz_path}")

    # Save JPEG (north-up)
    elev_for_jpg = elevation_data_array  # no flip, as you said it's already correct
    min_val, max_val = elev_for_jpg.min(), elev_for_jpg.max()
    scaled_jpg = ((elev_for_jpg - min_val) / (max_val - min_val) * 255).astype(np.uint8) if max_val > min_val else np.zeros_like(elev_for_jpg, dtype=np.uint8)
    jpg_path = f"{filename_base}.jpg"
    Image.fromarray(scaled_jpg).save(jpg_path, quality=95)
    print(f"Visual JPEG saved: {jpg_path}")

    # Stats
    print("\n--- Elevation Data Stats ---")
    print(f"Min Elevation: {np.min(elevation_data_array):.2f}m")
    print(f"Max Elevation: {np.max(elevation_data_array):.2f}m")
    print(f"Mean Elevation: {np.mean(elevation_data_array):.2f}m")
    print(f"Resolution (Lon): {abs(res_lon):.6f}°/px")
    print(f"Resolution (Lat): {abs(res_lat):.6f}°/px")

def main():
    ## north macedonia
    west = 20.4529023 - 0.05
    south = 40.852478 - 0.05
    east = 23.034051 + 0.05
    north = 42.3739044 + 0.05

    output_dir = __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fetch_and_process_heightmap(west, south, east, north, zoom=12, output_dir=output_dir, max_workers=32)
    return

if __name__ == "__main__":
    main()

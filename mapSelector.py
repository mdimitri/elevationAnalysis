import tkinter as tk, os, math
from PIL import Image
from tqdm.auto import tqdm
import requests
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import geopandas as gpd
import geodatasets
from matplotlib.patches import Rectangle
from tkinter import ttk
from tkinter import messagebox
import threading
import pyperclip
import pickle
import matplotlib, pyproj
from shapely.ops import transform as shapely_transform
from shapely.geometry import box
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_npz_to_epsg4326_arrayInput(elev, extent):
    # extent = [top_left_lon, bottom_right_lon, bottom_right_lat, top_left_lat]
    south, north, east, west = extent[2], extent[3], extent[1], extent[0]
    width, height = elev.shape[1], elev.shape[0]
    project_to_merc = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    merc_bounds = shapely_transform(project_to_merc, box(west, south, east, north)).bounds
    src_transform = from_bounds(*merc_bounds, width, height)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        "EPSG:3857", "EPSG:4326", width, height, *merc_bounds
    )
    dst_array = np.empty((dst_height, dst_width), dtype=elev.dtype)
    reproject(
        elev, dst_array,
        src_transform=src_transform, src_crs="EPSG:3857",
        dst_transform=dst_transform, dst_crs="EPSG:4326",
        resampling=Resampling.bilinear
    )

    new_west, new_north = dst_transform.c, dst_transform.f
    return dst_array, {
        'south': new_north + dst_height * dst_transform.e,
        'north': new_north,
        'west': new_west,
        'east': new_west + dst_width * dst_transform.a,
        'resolution_lon_deg': dst_transform.a,
        'resolution_lat_deg': -dst_transform.e
    }

def ask_sure(numtiles: int, tile_size: int) -> bool:
    result = messagebox.askyesno("Are you sure?", "Request to download %d tiles of size %dx%dpx? \n Total size: %.1fMB." % (numtiles, tile_size, tile_size, 0.25*numtiles))
    return result

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

def fetch_and_process_heightmap(west, south, east, north, zoom=13, output_dir="heightmap_data", max_workers=10, parent=None):
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
    num_tiles = num_rows * num_cols

    if not ask_sure(num_tiles, tile_size):
        return

    # --- Progress window (lives on the Tk main thread) ---
    progress_win = tk.Toplevel(parent)
    progress_win.title("Downloading Tiles")
    progress_win.resizable(False, False)

    status_var = tk.StringVar(value=f"Downloading {num_tiles} tiles…")
    tk.Label(progress_win, textvariable=status_var).pack(padx=16, pady=(16, 8))

    progress_var = tk.IntVar(value=0)
    progress = ttk.Progressbar(progress_win, maximum=num_tiles, length=320, variable=progress_var, mode="determinate")
    progress.pack(padx=16, pady=(0, 16))

    progress_win.update_idletasks()
    if parent is not None:
        # center over parent
        pw = progress_win.winfo_width()
        ph = progress_win.winfo_height()
        px = parent.winfo_rootx() + (parent.winfo_width() // 2) - (pw // 2)
        py = parent.winfo_rooty() + (parent.winfo_height() // 2) - (ph // 2)
    else:
        # center on screen
        sw = progress_win.winfo_screenwidth()
        sh = progress_win.winfo_screenheight()
        pw = progress_win.winfo_width()
        ph = progress_win.winfo_height()
        px = (sw // 2) - (pw // 2)
        py = (sh // 2) - (ph // 2)

    progress_win.geometry(f"+{px}+{py}")

    # Keep the window on top of the parent, but not permanently
    if parent is not None:
        progress_win.transient(parent)

    def worker():
        try:
            elevation_data_array = np.zeros((num_rows * tile_size, num_cols * tile_size), dtype=np.float32)
            base_url = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"

            tasks = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for xi, x_tile in enumerate(range(min_tile_x, max_tile_x + 1)):
                    for yi, y_tile in enumerate(range(min_tile_y, max_tile_y + 1)):
                        tasks.append(executor.submit(
                            fetch_tile, xi, yi, x_tile, y_tile, zoom, tile_size, base_url, max_retries=10
                        ))

                completed = 0
                for future in as_completed(tasks):
                    result = future.result()
                    completed += 1
                    # update progress in UI thread
                    if parent is not None:
                        parent.after(0, progress_var.set, completed)

                    if result is not None:
                        xi, yi, elevation = result
                        elevation_data_array[yi * tile_size:(yi + 1) * tile_size,
                                             xi * tile_size:(xi + 1) * tile_size] = elevation

            # If nothing downloaded, bail out
            if np.all(elevation_data_array == 0) and num_tiles > 0:
                print("Warning: No elevation data was successfully processed.")
                if parent is not None:
                    parent.after(0, progress_win.destroy)
                return

            # Switch to a small spinner while we mosaic & save
            def start_spin():
                status_var.set("Mosaicking & saving…")
                progress.config(mode="indeterminate", maximum=100)
                progress.start(10)
            if parent is not None:
                parent.after(0, start_spin)

            # Calculate bounds & resolution (pixel-centered), then save (same as your code)
            top_left_lat, top_left_lon = tile_to_latlon(min_tile_x, min_tile_y, zoom)
            bottom_right_lat, bottom_right_lon = tile_to_latlon(max_tile_x + 1, max_tile_y + 1, zoom)

            res_lon = (bottom_right_lon - top_left_lon) / (num_cols * tile_size)
            res_lat = (top_left_lat - bottom_right_lat) / (num_rows * tile_size)

            top_left_lat -= (res_lat / 2)
            top_left_lon += (res_lon / 2)
            bottom_right_lat += (res_lat / 2)
            bottom_right_lon -= (res_lon / 2)

            filename_base = os.path.join(
                output_dir,
                f"heightmap_z{zoom}_lon_{top_left_lon:.1f}_{bottom_right_lon:.1f}_lat_{bottom_right_lat:.1f}_{top_left_lat:.1f}"
            )

            npz_path = f"{filename_base}.npz"
            np.savez(
                npz_path,
                elevations=elevation_data_array,
                north=top_left_lat,
                south=bottom_right_lat,
                west=top_left_lon,
                east=bottom_right_lon,
                resolution_lon_deg=res_lon,
                resolution_lat_deg=res_lat,
            )
            print(f"Raw elevation data saved: {npz_path}")

            # --- Popup to copy filename ---
            def show_copy_popup():
                popup = tk.Toplevel(parent)
                popup.title("File Saved")
                popup.resizable(False, False)

                tk.Label(popup, text="Heightmap saved as:", font=("Arial", 14)).pack(padx=16, pady=(16, 8))

                # Use an Entry for selectable text
                entry_var = tk.StringVar(value=f".\{filename_base}.npz")
                entry = tk.Entry(popup, textvariable=entry_var, font=("Arial", 14), width=60)
                entry.pack(padx=16, pady=(0, 8))
                entry.configure(state='readonly')  # prevent editing, but still selectable

                def copy_to_clipboard():
                    popup.clipboard_clear()
                    popup.clipboard_append(f".\{filename_base}.npz")
                    popup.update()  # keep it on clipboard
                    tk.messagebox.showinfo("Copied", "Filename copied to clipboard!")

                copy_btn = tk.Button(popup, text="Copy to clipboard", command=copy_to_clipboard)
                copy_btn.pack(pady=(0, 16))

                # center popup
                pw, ph = 800, 200
                if parent:
                    px = parent.winfo_rootx() + (parent.winfo_width() // 2) - (pw // 2)
                    py = parent.winfo_rooty() + (parent.winfo_height() // 2) - (ph // 2)
                else:
                    sw, sh = popup.winfo_screenwidth(), popup.winfo_screenheight()
                    px = (sw // 2) - (pw // 2)
                    py = (sh // 2) - (ph // 2)
                popup.geometry(f"{pw}x{ph}+{px}+{py}")

            if parent is not None:
                parent.after(0, show_copy_popup)

            elev_for_jpg = elevation_data_array
            min_val, max_val = elev_for_jpg.min(), elev_for_jpg.max()
            scaled_jpg = (
                ((elev_for_jpg - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                if max_val > min_val else np.zeros_like(elev_for_jpg, dtype=np.uint8)
            )
            jpg_path = f"{filename_base}.jpg"
            Image.fromarray(scaled_jpg).save(jpg_path, quality=95)
            print(f"Visual JPEG saved: {jpg_path}")

            print("\n--- Elevation Data Stats ---")
            print(f"Min Elevation: {np.min(elevation_data_array):.2f}m")
            print(f"Max Elevation: {np.max(elevation_data_array):.2f}m")
            print(f"Mean Elevation: {np.mean(elevation_data_array):.2f}m")
            print(f"Resolution (Lon): {abs(res_lon):.6f}°/px")
            print(f"Resolution (Lat): {abs(res_lat):.6f}°/px")

        except Exception as e:
            # Ensure errors are visible
            print(f"Error in fetch_and_process_heightmap worker: {e}")
            try:
                if parent is not None:
                    parent.after(0, messagebox.showerror, "Error", str(e))
            except Exception:
                pass
        finally:
            if parent is not None:
                parent.after(0, lambda: (progress.stop(), progress_win.destroy()))

    # run the worker in a background thread so the GUI stays responsive
    threading.Thread(target=worker, daemon=True).start()


class MapSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Height map downloader")

        self.world_gdf = None
        self.rect_patch = None
        self.start_lon_lat = None
        self.end_lon_lat = None

        # Configure root grid
        self.root.rowconfigure(1, weight=1)  # map row expands
        self.root.columnconfigure(0, weight=1)

        # --- Top: Title ---
        title_label = tk.Label(root, text="Height map downloader", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # --- Middle: Map ---
        self.fig, self.ax = plt.subplots(figsize=(7, 3.5), layout='constrained')
        self.fig.tight_layout()
        self.ax.set_aspect('equal')
        self.ax.set_facecolor("aliceblue")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget.draw()
        self.canvas = self.canvas_widget.get_tk_widget()
        self.canvas.grid(row=1, column=0, sticky="nsew")  # expands with window

        self.toolbar = NavigationToolbar2Tk(self.canvas_widget, root, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=2, column=0, sticky="ew")

        # --- Bottom: Controls ---
        self.controls_frame = tk.Frame(root)
        self.controls_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        self.coords_label = tk.Label(self.controls_frame, text="Selected Area: (N/A)", font=("Arial", 12))
        self.coords_label.pack(pady=2)

        zoom_label = tk.Label(self.controls_frame, text="Level of detail:", font=("Arial", 12))
        zoom_label.pack(pady=2)

        self.zoom_level = tk.IntVar(value=9)
        self.zoom_dropdown = ttk.Combobox(
            self.controls_frame, textvariable=self.zoom_level,
            values=list(range(8, 15)), width=5, state="readonly"
        )
        self.zoom_dropdown.pack(pady=2)

        get_height_btn = tk.Button(self.controls_frame, text="Get height", font=("Arial", 12),
                                   command=self.on_get_height)
        get_height_btn.pack(pady=5)

        controls_info_label = tk.Label(
            self.controls_frame,
            text="ℹ Left click: select rectangle | Scroll: zoom | Right click: pan",
            font=("Arial", 10),
            anchor="center",
            justify="center"
        )
        controls_info_label.pack(pady=2)

        # --- Load map ---
        self.load_and_plot_map()

        # --- Connect events ---
        self.canvas_widget.mpl_connect("button_press_event", self.on_button_press)
        self.canvas_widget.mpl_connect("motion_notify_event", self.on_mouse_drag)
        self.canvas_widget.mpl_connect("button_release_event", self.on_button_release)
        self.canvas_widget.mpl_connect("scroll_event", self.on_scroll)

        self.canvas_widget.mpl_connect("button_press_event", self.on_button_press_pan)
        self.canvas_widget.mpl_connect("motion_notify_event", self.on_mouse_drag_pan)
        self.canvas_widget.mpl_connect("button_release_event", self.on_button_release_pan)

        self.root.bind("<Configure>", self.on_window_resize)

    def fetchHeight(self, east, west, north, south, zoom):
        print(f"Fetching: East={east}, West={west}, North={north}, South={south}, Zoom={zoom}")
        output_dir = f"{zoom}_{east:.1f}_{west:.1f}_{north:.1f}_{south:.1f}"
        fetch_and_process_heightmap(west, south, east, north, zoom=zoom, output_dir=output_dir, max_workers=16, parent=self.root)

    def on_get_height(self):
        if self.rect_patch:
            # Get rectangle coordinates
            x0, y0 = self.rect_patch.get_xy()
            width = self.rect_patch.get_width()
            height = self.rect_patch.get_height()
            east = x0 + width
            west = x0
            north = y0 + height
            south = y0
            zoom = self.zoom_level.get()

            # Call your function
            self.fetchHeight(east=east, west=west, north=north, south=south, zoom=zoom)
        else:
            tk.messagebox.showwarning("No selection", "Please select a rectangle first.")

    # --- Zoom ---
    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return

        base_scale = 1.2
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale

        xdata, ydata = event.xdata, event.ydata
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])

        self.canvas_widget.draw_idle()

    # --- Pan ---
    def on_button_press_pan(self, event):
        if event.inaxes != self.ax:
            return

        # Only right-click (usually button=3)
        if event.button == 3:
            self._pan_start = (event.x, event.y)
            self._xlim_start = self.ax.get_xlim()
            self._ylim_start = self.ax.get_ylim()
        elif event.button == 1:
            # Keep existing left-click rectangle code
            self.on_button_press(event)

    def on_mouse_drag_pan(self, event):
        if event.inaxes != self.ax or not hasattr(self, "_pan_start"):
            # Fallback to rectangle dragging
            self.on_mouse_drag(event)
            return

        if event.button != 3:
            return

        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]

        # Transform pixel movement to data coordinates
        x0, x1 = self._xlim_start
        y0, y1 = self._ylim_start
        width = x1 - x0
        height = y1 - y0

        x_pixels = self.ax.bbox.width
        y_pixels = self.ax.bbox.height

        dx_data = -dx / x_pixels * width
        dy_data = -dy / y_pixels * height

        self.ax.set_xlim(x0 + dx_data, x1 + dx_data)
        self.ax.set_ylim(y0 + dy_data, y1 + dy_data)

        self.canvas_widget.draw_idle()

    def on_button_release_pan(self, event):
        if hasattr(self, "_pan_start"):
            del self._pan_start
            del self._xlim_start
            del self._ylim_start
        # Keep existing left-click release code
        self.on_button_release(event)

    def load_and_plot_map(self):
        try:
            print("Attempting to load 'naturalearth.countries.50m'...")
            base_dir = os.path.dirname(__file__)
            self.world_gdf = gpd.read_file(os.path.join(base_dir, "worldMap", "ne_50m_admin_0_countries.zip"))
            print(
                "Successfully loaded local world map. (downloaded from: https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/cultural/ne_50m_admin_0_countries.zip")
        except ValueError:
            print("'naturalearth.countries.50m' not found. Falling back to 'naturalearth.land'.")
            self.world_gdf = gpd.read_file(geodatasets.get_path('naturalearth.land'))
            print("Successfully loaded 'naturalearth.land'.")

        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_facecolor("aliceblue")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # --- load low res terrain data for guidance ---
        base_dir = os.path.dirname(__file__)
        world_map_dir = os.path.join(base_dir, "worldMap")
        os.makedirs(world_map_dir, exist_ok=True)  # Ensure the directory exists
        pickle_path = os.path.join(world_map_dir, "world_elevation_data.pkl")

        if os.path.exists(pickle_path):
            print("Loading elevation data from pickle file...")
            try:
                with open(pickle_path, 'rb') as f:
                    data = pickle.load(f)
                    elevation_data_array = data['elevations']
                    extent = data['extent']
                print("Successfully loaded elevation data from pickle.")
            except Exception as e:
                print(f"Error loading pickle file: {e}. Re-downloading.")
                elevation_data_array, extent = self._download_and_process_low_res_map(world_map_dir, pickle_path)
        else:
            print("Pickle file not found. Downloading and processing low-res elevation data...")
            elevation_data_array, extent = self._download_and_process_low_res_map(world_map_dir, pickle_path)

        print('Converting lowres elevation to epsg4326')
        elevation_data_array, meta_4326 = reproject_npz_to_epsg4326_arrayInput(elevation_data_array, extent)
        south, north, east, west = extent[2], extent[3], extent[1], extent[0]

        land = np.copy(elevation_data_array); land[land < 0] = 0
        water = np.copy(elevation_data_array); water[water > 0] = 8000

        terrain_cmap = matplotlib.colormaps['terrain']
        blues_cmap = matplotlib.colormaps['Blues_r']
        landRGB  = terrain_cmap(0.2 + np.clip(land / 8000, 0, 1)**.45)
        waterRGB = blues_cmap((np.clip(water / 6000, -1, 1) + 1)/2)
        worldRGB = np.where((elevation_data_array > 0)[..., None], landRGB, waterRGB)
        if elevation_data_array is not None:
            # Plot the elevation data as a raster image
            min_elev = np.min(elevation_data_array)
            max_elev = np.max(elevation_data_array)
            self.ax.imshow(
                worldRGB,
                extent=[west, east, south, north],
                origin='upper',
                cmap='terrain',
                vmin=min_elev,
                vmax=max_elev,
                zorder=0
            )
            print("Successfully plotted height map background.")
        # ---low res terrain map plotting end---

        # The existing code to plot the country borders goes on top
        self.world_gdf.plot(ax=self.ax, color='none', edgecolor='black', linewidth=0.5, alpha=1)

        self.ax.set_xlim(-180, 180)
        self.ax.set_ylim(-90, 90)

        self.canvas_widget.draw_idle()

    def _download_and_process_low_res_map(self, world_map_dir, pickle_path):
        """Helper function to download, process, and save low-res elevation data."""
        west, south, east, north = -180, -89.9, 180, 89.9
        zoom_level = 4
        tile_x_nw, tile_y_nw = latlon_to_tile(north, west, zoom_level)
        tile_x_se, tile_y_se = latlon_to_tile(south, east, zoom_level)
        min_tile_x = min(tile_x_nw, tile_x_se)
        max_tile_x = max(tile_x_nw, tile_x_se)
        min_tile_y = min(tile_y_nw, tile_y_se)
        max_tile_y = max(tile_y_nw, tile_y_se)
        num_cols = max_tile_x - min_tile_x + 1
        num_rows = max_tile_y - min_tile_y + 1
        tile_size = 256
        elevation_data_array = np.zeros((num_rows * tile_size, num_cols * tile_size), dtype=np.float32)
        base_url = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"

        for xi, x_tile in enumerate(range(min_tile_x, max_tile_x + 1)):
            for yi, y_tile in enumerate(range(min_tile_y, max_tile_y + 1)):
                tile_url = f"{base_url}/{zoom_level}/{x_tile}/{y_tile}.png"
                try:
                    response = requests.get(tile_url, stream=True, timeout=5)
                    response.raise_for_status()
                    img = Image.open(response.raw).convert("RGB")
                    arr = np.array(img, dtype=np.float32)
                    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                    elevation = (r * 256 + g + b / 256) - 32768
                    elevation_data_array[
                        yi * tile_size:(yi + 1) * tile_size, xi * tile_size:(xi + 1) * tile_size] = elevation
                except Exception as e:
                    print(f"Could not fetch low-res tile {x_tile},{y_tile}: {e}")

        top_left_lat, top_left_lon = tile_to_latlon(min_tile_x, min_tile_y, zoom_level)
        bottom_right_lat, bottom_right_lon = tile_to_latlon(max_tile_x + 1, max_tile_y + 1, zoom_level)
        extent = [top_left_lon, bottom_right_lon, bottom_right_lat, top_left_lat]

        # Save the data and extent to a pickle file
        data = {'elevations': elevation_data_array, 'extent': extent}
        try:
            with open(pickle_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Low-res elevation data saved to {pickle_path}")
        except Exception as e:
            print(f"Error saving pickle file: {e}")

        return elevation_data_array, extent

    # --- Left click for rectangle ---
    def on_button_press(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return  # ignore non-left clicks

        self.start_lon_lat = (event.xdata, event.ydata)
        if self.rect_patch:
            self.rect_patch.remove()
            self.rect_patch = None
        self.coords_label.config(text="Selected Area: (N/A)")
        self.canvas_widget.draw_idle()

    def on_mouse_drag(self, event):
        if event.inaxes != self.ax or self.start_lon_lat is None or event.button != 1:
            return  # ignore non-left drags

        current_lon, current_lat = event.xdata, event.ydata
        if current_lon is None or current_lat is None:
            return

        x0, y0 = self.start_lon_lat
        width = current_lon - x0
        height = current_lat - y0

        if self.rect_patch:
            self.rect_patch.set_xy((x0, y0))
            self.rect_patch.set_width(width)
            self.rect_patch.set_height(height)
        else:
            self.rect_patch = Rectangle(
                (x0, y0), width, height,
                fill=False, edgecolor='red', linewidth=2, linestyle='--'
            )
            self.ax.add_patch(self.rect_patch)

        self.canvas_widget.draw_idle()

    def on_button_release(self, event):
        if event.inaxes != self.ax or self.start_lon_lat is None or event.button != 1:
            return  # ignore non-left releases

        self.end_lon_lat = (event.xdata, event.ydata)
        if self.end_lon_lat[0] is None or self.end_lon_lat[1] is None:
            if self.rect_patch:
                self.rect_patch.remove()
                self.rect_patch = None
            self.start_lon_lat = None
            self.coords_label.config(text="Selected Area: (N/A - Invalid Release)")
            self.canvas_widget.draw_idle()
            return

        x0, y0 = self.start_lon_lat
        x1, y1 = self.end_lon_lat

        min_lon = min(x0, x1)
        max_lon = max(x0, x1)
        min_lat = min(y0, y1)
        max_lat = max(y0, y1)

        if self.rect_patch:
            self.rect_patch.set_xy((min_lon, min_lat))
            self.rect_patch.set_width(max_lon - min_lon)
            self.rect_patch.set_height(max_lat - min_lat)
        else:
            self.rect_patch = Rectangle(
                (min_lon, min_lat), 0.1, 0.1,
                fill=False, edgecolor='red', linewidth=2, linestyle='--'
            )
            self.ax.add_patch(self.rect_patch)

        self.coords_label.config(
            text=f"Selected Area: Lon {min_lon:.2f}° to {max_lon:.2f}°, Lat {min_lat:.2f}° to {max_lat:.2f}°"
        )
        print(f"Rectangle selected: Longitude ({min_lon:.2f}, {max_lon:.2f}), Latitude ({min_lat:.2f}, {max_lat:.2f})")

        self.start_lon_lat = None
        self.canvas_widget.draw_idle()

    def on_window_resize(self, event):
        self.fig.tight_layout()
        self.canvas_widget.draw_idle()

    def on_close(self):
        """Ensure app terminates cleanly when window is closed."""
        self.root.quit()
        self.root.destroy()
        os._exit(0)  # hard exit to kill any threads (like ThreadPoolExecutor)

if __name__ == "__main__":
    root = tk.Tk()
    app = MapSelectorApp(root)
    root.mainloop()

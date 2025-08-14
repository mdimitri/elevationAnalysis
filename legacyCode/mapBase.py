import argparse, os, sys, numpy as np, math, matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import minimum_filter, generate_binary_structure
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
import geopandas as gpd
from shapely.geometry import box
import requests
from shapely.geometry import Point
import pickle
from scipy.ndimage import gaussian_filter
from scipy.ndimage import sobel
from shapely.geometry import LineString
import osm2geojson

gamma = 0.5  # set your gamma here
cmap = plt.get_cmap('terrain')
x = np.linspace(0.05, 1, 256)
colors = cmap(x ** gamma)
gamma_cmap = LinearSegmentedColormap.from_list('gamma_terrain', colors)



def haversine_distance(lat1, lon1, lat2, lon2, radius=6371.0):
    """
    Calculate the great-circle distance between two points on Earth.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        Latitude and longitude in degrees.
    radius : float
        Radius of the Earth in kilometers (default: 6371 km).
        Use 3958.8 for miles.

    Returns
    -------
    distance : float
        Distance between the two points in the same units as radius.
    """
    # Convert degrees to radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c

def get_roads_from_osm(lat1, lat2, lon1, lon2):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    way["highway"]({lat1},{lon1},{lat2},{lon2});
    out geom;
    """

    response = requests.get(overpass_url, params={"data": query})
    response.raise_for_status()
    data = response.json()

    elements = data.get("elements", [])
    records = []
    for el in elements:
        highway_type = el["tags"].get("highway")
        if "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            if len(coords) > 1:
                records.append({
                    "highway": highway_type,
                    "geometry": LineString(coords)
                })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf

def get_places_from_osm(lat1, lat2, lon1, lon2):
    # Overpass query
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    node["place"]({lat1},{lon1},{lat2},{lon2});
    out;
    """

    # Send request
    response = requests.get(overpass_url, params={"data": query})
    response.raise_for_status()
    data = response.json()

    # Parse into GeoDataFrame
    elements = data.get("elements", [])
    records = []
    for el in elements:
        name = el["tags"].get("name")
        if name:  # Only keep named places
            records.append({
                "name": name,
                "place": el['tags'].get('place'),
                "lat": el["lat"],
                "lon": el["lon"]
            })

    gdf = gpd.GeoDataFrame(
        records,
        geometry=[Point(row["lon"], row["lat"]) for row in records],
        crs="EPSG:4326"
    )
    return gdf

def load_or_fetch_places(south, north, west, east, fetch_func):
    """
    Load cached places_gdf if exists, otherwise fetch and save it.

    fetch_func: function(south, north, west, east) -> GeoDataFrame
    """
    # Create a filename that encodes the bbox with rounded coords for readability
    filename = f"places_s{south:.2f}_n{north:.2f}_w{west:.2f}_e{east:.2f}.pkl"

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            places_gdf = pickle.load(f)
        print(f"Loaded cached data from {filename}")
    else:
        places_gdf = fetch_func(south, north, west, east)
        with open(filename, 'wb') as f:
            pickle.dump(places_gdf, f)
        print(f"Fetched data and saved to {filename}")

    return places_gdf

def load_or_fetch_roads(south, north, west, east, fetch_func):
    filename = f"roads_s{south:.2f}_n{north:.2f}_w{west:.2f}_e{east:.2f}.pkl"

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            roads_gdf = pickle.load(f)
        print(f"Loaded cached roads from {filename}")
    else:
        roads_gdf = fetch_func(south, north, west, east)
        with open(filename, 'wb') as f:
            pickle.dump(roads_gdf, f)
        print(f"Fetched roads and saved to {filename}")

    return roads_gdf

def get_rivers_from_osm(lat1, lat2, lon1, lon2):
    overpass_url = "https://overpass-api.de/api/interpreter"
    waterway_types = "river|stream|creek|canal|drain|ditch|tributary"
    query = f"""
    [out:json];
    (
      way["waterway"~"{waterway_types}"]({lat1},{lon1},{lat2},{lon2});
      relation["waterway"~"{waterway_types}"]({lat1},{lon1},{lat2},{lon2});
    );
    out geom;
    """

    response = requests.get(overpass_url, params={"data": query})
    response.raise_for_status()
    data = response.json()

    elements = data.get("elements", [])
    records = []
    for el in elements:
        if "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            if len(coords) > 1:
                tags = el.get("tags", {})
                records.append({
                    "geometry": LineString(coords),
                    "waterway": tags.get("waterway", "unknown"),
                    "name": tags.get("name")
                })

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf

def load_or_fetch_rivers(south, north, west, east, fetch_func):
    filename = f"rivers_s{south:.2f}_n{north:.2f}_w{west:.2f}_e{east:.2f}.pkl"

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            rivers_gdf = pickle.load(f)
        print(f"Loaded cached rivers from {filename}")
    else:
        rivers_gdf = fetch_func(south, north, west, east)
        with open(filename, 'wb') as f:
            pickle.dump(rivers_gdf, f)
        print(f"Fetched rivers and saved to {filename}")

    return rivers_gdf

def get_water_bodies_osm2geojson(lat1, lat2, lon1, lon2):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      way["natural"="water"]({lat1},{lon1},{lat2},{lon2});
      relation["natural"="water"]({lat1},{lon1},{lat2},{lon2});
      way["water"~"lake|reservoir|sea|ocean"]({lat1},{lon1},{lat2},{lon2});
      relation["water"~"lake|reservoir|sea|ocean"]({lat1},{lon1},{lat2},{lon2});
    );
    out geom;
    """
    response = requests.get(overpass_url, params={"data": query})
    response.raise_for_status()
    data = response.json()
    geojson = osm2geojson.json2geojson(data)
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    return gdf

def load_or_fetch_water_bodies(south, north, west, east, fetch_func):
    filename = f"water_bodies_s{south:.2f}_n{north:.2f}_w{west:.2f}_e{east:.2f}.pkl"
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            water_bodies_gdf = pickle.load(f)
        print(f"Loaded cached water bodies from {filename}")
    else:
        water_bodies_gdf = fetch_func(south, north, west, east)
        with open(filename, 'wb') as f:
            pickle.dump(water_bodies_gdf, f)
        print(f"Fetched water bodies and saved to {filename}")
    return water_bodies_gdf

def plot_relief_with_places(places_gdf, roads_gdf, rivers_gdf, water_bodies_gdf, map_s, south, west, north, east, dpi, scale):
    """
    Plot the provided relief raster (map_s) as base map, and overlay places with labels.

    Parameters:
    - places_gdf: GeoDataFrame with Point geometries and 'name' column.
    - map_s: 2D numpy array, relief raster aligned to bounding box.
    - south, west, north, east: floats defining bounding box coordinates.
    """
    markerScale = 1
    scatterSize = .2 * markerScale * 1000 / dpi
    textSize = .005 * markerScale * 1000 / dpi
    cbarSize = 5

    fig_width = scale*10
    fig_height = scale*10
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    # Display relief map with extent matching bounding box
    im = ax.imshow(map_s, extent=[west, east, south, north], origin='upper', cmap=gamma_cmap, vmin=0)

    # display the contours
    map_s_smooth = gaussian_filter(map_s, sigma=map_s.shape[0] / 6000)
    nrows, ncols = map_s.shape
    x = np.linspace(west, east, ncols)
    y = np.linspace(north, south, nrows)
    X, Y = np.meshgrid(x, y)
    min_val = np.min(map_s_smooth)
    max_val = np.max(map_s_smooth)

    # mini contours every 25m
    levels_mini = np.arange(min_val, max_val, 25)
    # Thin contours every 100m
    levels_thin = np.arange(min_val, max_val, 100)
    # Thick contours every 500m
    levels_thick = np.arange(min_val, max_val, 500)
    # Mini contours, very thin
    contours_mini = ax.contour(X, Y, map_s_smooth, levels=levels_mini,
                               colors='0.55', alpha=0.2, linewidths=0.025)
    # Thin contours, lighter and thinner lines
    contours_thin = ax.contour(X, Y, map_s_smooth, levels=levels_thin,
                               colors='0.55', alpha=0.3, linewidths=0.05)
    # Thick contours, darker and thicker lines
    contours_thick = ax.contour(X, Y, map_s_smooth, levels=levels_thick,
                                colors='0.5', alpha=0.3, linewidths=0.1)
    # Label only the thick contours
    # ax.clabel(contours_thin, inline=True, inline_spacing=1, fontsize=.5, fmt='%1.0f', colors='0.4')
    label_interval = 0.1  # your label spacing in map units
    min_path_length = 0.3  # minimum length of path to place multiple labels

    for level, collection in zip(contours_thin.levels, contours_thin.collections):
        for path in collection.get_paths():
            vertices = path.vertices
            if len(vertices) < 2:
                continue

            # Calculate cumulative distance along path
            deltas = np.diff(vertices, axis=0)
            seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
            cumulative_length = np.concatenate([[0], np.cumsum(seg_lengths)])
            total_length = cumulative_length[-1]

            if total_length < min_path_length:
                # For very short paths, place just one label at the midpoint or skip
                midpoint_idx = len(vertices) // 2
                x, y = vertices[midpoint_idx]
                ax.text(x, y, f"{level:.0f}", fontsize=0.3, color='0.55', alpha=0.5,
                        ha='center', va='center', zorder=1)
                continue

            # Positions where labels should be placed along longer paths
            label_positions = np.arange(0, total_length, label_interval)

            for pos in label_positions:
                idx = np.searchsorted(cumulative_length, pos)
                if idx >= len(vertices):
                    idx = len(vertices) - 1
                x, y = vertices[idx]
                ax.text(x, y, f"{level:.0f}", fontsize=0.5, color='0.55', alpha=0.6,
                        ha='center', va='center', zorder=1)


    # Plot rivers (blueish, thicker lines)
    if rivers_gdf is not None and not rivers_gdf.empty:
        # Separate rivers from smaller water features
        rivers_only = rivers_gdf[rivers_gdf["waterway"].isin(["river", "stream"])]
        smaller_waterways = rivers_gdf[rivers_gdf["waterway"].isin(["channel", "irrigation", "canal", "derelict_canal", "ditch", "drain", ""])]

        # Plot rivers with deeper blue and thicker lines
        if not rivers_only.empty:
            rivers_only.plot(
                ax=ax,
                linewidth=0.1,
                edgecolor=(0.0, 0.4, 0.7),  # deep blue
                alpha=0.7,
                zorder=2
            )

            # Add river names along the lines
            for _, row in rivers_only.iterrows():
                name = row.get("name")
                geom = row.geometry
                if not name or geom.is_empty:
                    continue
                lines = [
                    geom] if geom.geom_type == 'LineString' else geom.geoms if geom.geom_type == 'MultiLineString' else []
                for line in lines:
                    coords = np.array(line.coords)
                    if len(coords) < 2:
                        continue
                    cumlen = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))])
                    for pos in np.arange(0, cumlen[-1], 0.1):
                        i = np.searchsorted(cumlen, pos)
                        if i >= len(coords):
                            i = len(coords) - 1
                        x, y = coords[i]
                        ax.text(x, y, name, fontsize=scatterSize, color=(0, 0.3, 0.6), alpha=0.8, zorder=2,
                                ha='center', va='center', style='italic')

        # Plot smaller waterways lighter and thinner
        if not smaller_waterways.empty:
            smaller_waterways.plot(
                ax=ax,
                linewidth=0.05,
                edgecolor=(0.3, 0.6, 0.9),  # lighter blue
                alpha=0.5,
                zorder=2
            )

    # --- Plot water bodies as filled polygons ---
    if water_bodies_gdf is not None and not water_bodies_gdf.empty:
        water_bodies_gdf.plot(
            ax=ax,
            facecolor=(0.7, 0.88, 0.96),
            edgecolor=(0.42, 0.7, 0.84),
            linewidth=0.1,
            zorder=3,
        )

    # cbar = plt.colorbar(im)
    # cbar.locator = MaxNLocator(nbins=30)  # request ~10 ticks
    # cbar.update_ticks()
    # cbar.ax.tick_params(labelsize=cbarSize)



    # Plot roads first (under places and labels)
    whatRoadsToPlot = {"motorway", "trunk", "primary", "secondary", "tertiary", "residential", "unclassified", "service"}
    if roads_gdf is not None and not roads_gdf.empty:
        # Filter roads by type
        roads_subset = roads_gdf[roads_gdf["highway"].isin(whatRoadsToPlot)]
        if not roads_subset.empty:
            highway_width = {
                "motorway": 0.4,
                "trunk": 0.3,
                "primary": 0.3,
                "secondary": 0.2,
                "tertiary": 0.15,
                "unclassified": 0.075,
                "residential": 0.075,
                "service": 0.05,
                "footway": 0.02
            }

            # Save current axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            widths = roads_subset["highway"].map(highway_width).fillna(0.05)

            # Plot fill
            roads_subset.plot(
                ax=ax,
                linewidth=widths,
                edgecolor=(0.3, 0.3, 0.3),
                alpha=0.75,
                zorder=4
            )
            roads_subset.plot(
                ax=ax,
                linewidth=widths * 0.6,
                edgecolor=(0.7, 0.7, 0.5),
                alpha=1,
                zorder=5
            )

            # Restore axis limits and aspect ratio
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal', adjustable='box')




    # Plot points on top
    # Define marker sizes for each place type
    size_scale_markers = {
        "country": scatterSize * 10,
        "region": scatterSize * 9,
        "city": scatterSize * 8,
        "town": scatterSize * 7,
        "suburb": scatterSize * 3,
        "quarter": scatterSize * 3,
        "neighbourhood": scatterSize * 2.5,
        "village": scatterSize * 1.5,
        "hamlet": scatterSize * 1.5,
        "isolated_dwelling": scatterSize * 1.2,
        "farm": scatterSize * 1.2,
        "square": scatterSize * 1.2,
        "locality": scatterSize * 1.2,
        "city_block": scatterSize,
        "natural=ridge": scatterSize  # unlikely to need emphasis
    }
    size_scale_labels = {
        "country": scatterSize * 14,
        "region": scatterSize * 13,
        "city": scatterSize * 12,
        "town": scatterSize * 11,
        "suburb": scatterSize * 3,
        "quarter": scatterSize * 3,
        "neighbourhood": scatterSize * 2.5,
        "village": scatterSize * 1.5,
        "hamlet": scatterSize * 1.5,
        "isolated_dwelling": scatterSize * 1.2,
        "farm": scatterSize * 1.2,
        "square": scatterSize * 1.2,
        "locality": scatterSize * 1.2,
        "city_block": scatterSize,
        "natural=ridge": scatterSize  # unlikely to need emphasis
    }
    whatToPlot = {"city", "town", "village"}
    # Plot each type with its own size
    for place_type in whatToPlot:
        subset = places_gdf[places_gdf["place"] == place_type]
        if not subset.empty:
            ax.scatter(
                subset.geometry.x,
                subset.geometry.y,
                marker="o",
                color=[0.1, 0.1, 0.1],
                s=size_scale_markers.get(place_type, scatterSize),
                zorder=6,
                linewidths=0
            )
            # Labels — proportional to importance
            proportional_text_size = (
                    size_scale_labels.get(place_type, textSize)
            )
            for idx, row in subset.iterrows():
                ax.text(
                    row.geometry.x,
                    row.geometry.y,
                    row["name"],
                    fontsize=proportional_text_size,
                    ha="center",
                    va="bottom",
                    color=[0.1, 0.1, 0.1],
                    zorder=7
                )

    # # Add labels
    # for idx, row in places_gdf.iterrows():
    #     ax.text(row.geometry.x, row.geometry.y, row['name'],
    #             fontsize=textSize, ha='center', va='bottom', color=[0.3,0.3,0.3], zorder=3)

    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')
    # ax.set_title('Relief Map with Places')
    fig.axes[0].axis('off')

    mng = plt.get_current_fig_manager()
    try:
        # For TkAgg backend (common)
        mng.window.wm_geometry(f"{fig_width * dpi}x{fig_height * dpi}+0+0")
    except Exception:
        pass

    fig.canvas.draw_idle()
    return fig
def main():
    # mat = load_mat(r'C:\Users\mdimitri\OneDrive - UGent\Desktop\Terrain party\Macedonia\mk_corrected.mat')
    map = np.load('mk_corrected.npy').T
    # sea level is -28510.299
    # 2489m is 94022.3
    # normalize
    map -= -28510.299
    map /= ((94022.3+28510.299) / 2489)

    subsample = 3
    map_s = map[0::subsample, 0::subsample]
    lat = np.load('latitudes.npy')
    lon = np.load('longitudes.npy')

    # get the geo data
    north = lat[0][0]
    south = lat[-1][0]
    east = lon[-1][0]  # min longitude
    west = lon[0][0]  # max longitude
    places_gdf = load_or_fetch_places(south, north, west, east, get_places_from_osm)
    roads_gdf = load_or_fetch_roads(south, north, west, east, get_roads_from_osm)
    rivers_gdf = load_or_fetch_rivers(south, north, west, east, get_rivers_from_osm)
    water_bodies_gdf = load_or_fetch_water_bodies(south, north, west, east, get_water_bodies_osm2geojson)

    # places_gdf = get_places_from_osm(south, north, west, east) # lat1, lat2, lon1, lon2

    scale=4 # does not scale the text, markers, etc.
    dpi=700 # scales text and markers
    fig = plot_relief_with_places(places_gdf, roads_gdf, rivers_gdf, water_bodies_gdf, map_s, south, west, north, east, dpi=dpi, scale=scale)


    fig.savefig('baseMap_%d_%d_%ddpi.png' % (subsample, scale, dpi), dpi=dpi)
    return

if __name__ == "__main__":
    main()
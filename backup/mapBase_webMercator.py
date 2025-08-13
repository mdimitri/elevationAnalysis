import argparse, os, sys, numpy as np, math, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
import requests, pickle
import osm2geojson
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import time
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pyproj
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

gamma = 0.5
cmap = plt.get_cmap('terrain')
x = np.linspace(0.05, 1, 256)
colors = cmap(x ** gamma)
gamma_cmap = LinearSegmentedColormap.from_list('gamma_terrain', colors)


def haversine_distance(lat1, lon1, lat2, lon2, radius=6371.0):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c
def reproject_npz_to_epsg4326(npz_path):
    data = np.load(npz_path)
    elev = data['elevations']
    south, north, west, east = data['south'].item(), data['north'].item(), data['west'].item(), data['east'].item()
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


def _fetch_data_from_osm(query, bounds, geometry_handler, desc):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = query.format(lat1=bounds[0], lon1=bounds[2], lat2=bounds[1], lon2=bounds[3])
    with tqdm(total=100, desc=f"Fetching {desc}",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]", dynamic_ncols=True,
              leave=False) as pbar:
        try:
            response = requests.get(overpass_url, params={"data": query})
            response.raise_for_status()
            pbar.update(100)
            elements = response.json().get("elements", [])
            return geometry_handler(elements)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {desc}: {e}")
            # Ensure an empty GeoDataFrame still has the expected columns for 'places'
            if desc == "points of interest":
                return gpd.GeoDataFrame(columns=['name', 'type', 'geometry'], crs="EPSG:4326")
            else:
                return gpd.GeoDataFrame()


def get_roads_from_osm(lat1, lat2, lon1, lon2):
    query = f'[out:json];way["highway"]({lat1},{lon1},{lat2},{lon2});out geom;'
    return _fetch_data_from_osm(query, (lat1, lat2, lon1, lon2), lambda elements: gpd.GeoDataFrame([
        {"highway": el["tags"].get("highway"),
         "geometry": LineString([(pt["lon"], pt["lat"]) for pt in el["geometry"]])}
        for el in tqdm(elements, desc="Processing road elements", dynamic_ncols=True, leave=False) if
        "geometry" in el and len(el["geometry"]) > 1
    ], crs="EPSG:4326"), "roads")


def get_places_from_osm(lat1, lat2, lon1, lon2):
    query = f"""
    [out:json];
    (
      node["place"]({lat1},{lon1},{lat2},{lon2});
      node["amenity"~"monastery|church|hospital|school|restaurant|cafe|bar|park"]({lat1},{lon1},{lat2},{lon2});
      node["historic"~"castle|monument|ruins|archaeological_site"]({lat1},{lon1},{lat2},{lon2});
      node["tourism"~"attraction|museum|artwork|viewpoint"]({lat1},{lon1},{lat2},{lon2});
      node["shop"]({lat1},{lon1},{lat2},{lon2});
    );
    out;
    """

    def places_geometry_handler(elements):
        records = []
        for el in tqdm(elements, desc="Processing POI elements", dynamic_ncols=True, leave=False):
            # Only process elements that have a geometry (lat/lon) and ideally a name
            if "lat" in el and "lon" in el:
                poi_name = el["tags"].get("name")
                # Prioritize specific amenity/historic/tourism tags, fallback to 'place', then 'shop', then 'unknown'
                poi_type = el["tags"].get("amenity") or \
                           el["tags"].get("historic") or \
                           el["tags"].get("tourism") or \
                           el["tags"].get("place") or \
                           el["tags"].get("shop") or \
                           "unknown"  # Ensure 'type' is always assigned

                records.append({
                    "name": poi_name,  # Safely get the name, can be None
                    "type": poi_type,  # This is the crucial 'type' column
                    "geometry": Point(el["lon"], el["lat"])
                })

        # Ensure the GeoDataFrame has the 'name', 'type', and 'geometry' columns even if 'records' is empty
        if not records:
            return gpd.GeoDataFrame(columns=['name', 'type', 'geometry'], crs="EPSG:4326")
        return gpd.GeoDataFrame(records, crs="EPSG:4326")

    return _fetch_data_from_osm(query, (lat1, lat2, lon1, lon2), places_geometry_handler, "points of interest")


def get_rivers_from_osm(lat1, lat2, lon1, lon2):
    waterway_types = "river|stream|creek|canal|drain|ditch|tributary"
    query = f'[out:json];(way["waterway"~"{waterway_types}"]({lat1},{lon1},{lat2},{lon2});relation["waterway"~"{waterway_types}"]({lat1},{lon1},{lat2},{lon2}););out geom;'
    return _fetch_data_from_osm(query, (lat1, lat2, lon1, lon2), lambda elements: gpd.GeoDataFrame([
        {"geometry": LineString([(pt["lon"], pt["lat"]) for pt in el["geometry"]]),
         "waterway": el["tags"].get("waterway", "unknown"), "name": el["tags"].get("name")}
        for el in tqdm(elements, desc="Processing river elements", dynamic_ncols=True, leave=False) if
        "geometry" in el and len(el["geometry"]) > 1
    ], crs="EPSG:4326"), "rivers")


def get_water_bodies_osm2geojson(lat1, lat2, lon1, lon2):
    query = f'[out:json];(way["natural"="water"]({lat1},{lon1},{lat2},{lon2});relation["natural"="water"]({lat1},{lon1},{lat2},{lon2});way["water"~"lake|reservoir|sea|ocean"]({lat1},{lon1},{lat2},{lon2});relation["water"~"lake|reservoir|sea|ocean"]({lat1},{lon1},{lat2},{lon2}););out geom;'
    return _fetch_data_from_osm(query, (lat1, lat2, lon1, lon2), lambda elements: gpd.GeoDataFrame.from_features(
        osm2geojson.json2geojson({"elements": elements})["features"], crs="EPSG:4326"), "water bodies")

def get_country_boundaries_from_osm(lat1, lat2, lon1, lon2):
    query = f"""
    [out:json];
    (
      relation["admin_level"="2"]({lat1},{lon1},{lat2},{lon2});
    );
    out geom;
    """

    def boundaries_geometry_handler(elements):
        if not elements:
            print("No elements found.")
            return gpd.GeoDataFrame(columns=['name', 'geometry'], crs="EPSG:4326")

        print(f"Elements count: {len(elements)}")

        try:
            geojson = osm2geojson.json2geojson({"elements": elements}, filter_used_refs=False)
            features = geojson.get("features", [])

            print(f"Features count after conversion: {len(features)}")
            if not features:
                print("No features found in GeoJSON.")
                return gpd.GeoDataFrame(columns=['name', 'geometry'], crs="EPSG:4326")

            gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

            if 'name' not in gdf.columns:
                gdf['name'] = None
            else:
                gdf['name'] = gdf['name'].fillna(
                    gdf['properties'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
                )

            return gdf[['name', 'geometry']]

        except Exception as e:
            print(f"Exception during GeoJSON processing: {e}")
            return gpd.GeoDataFrame(columns=['name', 'geometry'], crs="EPSG:4326")

        except Exception as e:
            print(f"Failed to process OSM data: {e}")
            # Return an empty GeoDataFrame on failure
            return gpd.GeoDataFrame(columns=['name', 'geometry'], crs="EPSG:4326")

    return _fetch_data_from_osm(query, (lat1, lat2, lon1, lon2), boundaries_geometry_handler, "country boundaries")


def load_or_fetch(filename_prefix, south, north, west, east, fetch_func):
    filename = f"{filename_prefix}_s{south:.2f}_n{north:.2f}_w{west:.2f}_e{east:.2f}.pkl"
    if os.path.exists(filename):
        print(f"Loading cached data for {filename_prefix} from {filename}")
        with open(filename, 'rb') as f:
            data_gdf = pickle.load(f)
        print(f"Loaded {len(data_gdf)} records.")
        return data_gdf
    else:
        print(f"Cache not found. Fetching {filename_prefix} from OSM.")
        data_gdf = fetch_func(south, north, west, east)
        with open(filename, 'wb') as f:
            pickle.dump(data_gdf, f)
        print(f"Fetched and saved {len(data_gdf)} records to {filename}")
        return data_gdf


def plot_relief_with_features(places_gdf, roads_gdf, rivers_gdf, water_bodies_gdf, country_boundaries_gdf, map_s, south, west, north, east, dpi,
                              scale):
    fig_width, fig_height = scale * 10, scale * 10
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
    ax.imshow(map_s, extent=[west, east, south, north], origin='upper', cmap=gamma_cmap, vmin=0, interpolation='bilinear')

    # #print("Plotting contours...")
    # map_s_smooth = gaussian_filter(map_s, sigma=map_s.shape[0] / 6000)
    # x, y = np.linspace(west, east, map_s.shape[1]), np.linspace(north, south, map_s.shape[0])
    # X, Y = np.meshgrid(x, y)
    # min_val, max_val = np.min(map_s_smooth), np.max(map_s_smooth)
    #
    # levels_mini = np.arange(min_val, max_val, 25)
    # levels_thin = np.arange(min_val, max_val, 100)
    # levels_thick = np.arange(min_val, max_val, 500)
    #
    # contours_mini = ax.contour(X, Y, map_s_smooth, levels=levels_mini,
    #                            colors='0.55', alpha=0.2, linewidths=0.025)
    # contours_thin = ax.contour(X, Y, map_s_smooth, levels=levels_thin,
    #                            colors='0.55', alpha=0.3, linewidths=0.05)
    # contours_thick = ax.contour(X, Y, map_s_smooth, levels=levels_thick,
    #                             colors='0.5', alpha=0.3, linewidths=0.1)
    #
    # print("Labeling thin contours...")
    # label_interval = 0.1
    # min_path_length = 0.3
    #
    # for level, collection in zip(
    #         tqdm(contours_thin.levels, desc="Drawing contour labels", dynamic_ncols=True, leave=False),
    #         contours_thin.collections):
    #     for path in collection.get_paths():
    #         vertices = path.vertices
    #         if len(vertices) < 2:
    #             continue
    #
    #         deltas = np.diff(vertices, axis=0)
    #         seg_lengths = np.sqrt((deltas ** 2).sum(axis=1))
    #         cumulative_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    #         total_length = cumulative_length[-1]
    #
    #         if total_length < min_path_length:
    #             midpoint_idx = len(vertices) // 2
    #             x_mid, y_mid = vertices[midpoint_idx]
    #             ax.text(x_mid, y_mid, f"{level:.0f}", fontsize=0.3, color='0.55', alpha=0.5,
    #                     ha='center', va='center', zorder=1)
    #             continue
    #
    #         label_positions = np.arange(0, total_length, label_interval)
    #
    #         for pos in label_positions:
    #             idx = np.searchsorted(cumulative_length, pos)
    #             if idx >= len(vertices):
    #                 idx = len(vertices) - 1
    #             x_pos, y_pos = vertices[idx]
    #             ax.text(x_pos, y_pos, f"{level:.0f}", fontsize=0.5, color='0.55', alpha=0.6,
    #                     ha='center', va='center', zorder=1)

    # print("Plotting water bodies...")
    if water_bodies_gdf is not None and not water_bodies_gdf.empty:
        water_bodies_gdf.plot(ax=ax, facecolor=(0.7, 0.88, 0.96), edgecolor=(0.42, 0.7, 0.84), linewidth=0.1, zorder=3)

    print("Plotting rivers and their labels...")
    label_interval = 0.1
    if rivers_gdf is not None and not rivers_gdf.empty:
        rivers_only = rivers_gdf[rivers_gdf["waterway"].isin(["river", "stream"])]
        smaller_waterways = rivers_gdf[
            rivers_gdf["waterway"].isin(["channel", "irrigation", "canal", "derelict_canal", "ditch", "drain", ""])]

        if not rivers_only.empty:
            rivers_only.plot(
                ax=ax,
                linewidth=0.05,
                edgecolor=(0.0, 0.4, 0.7),
                alpha=0.7,
                zorder=2
            )

            placed_river_labels = []
            min_label_distance_map_units = 0.006

            for _, row in tqdm(rivers_only.iterrows(), total=len(rivers_only), desc="Labeling rivers",
                               dynamic_ncols=True, leave=False):
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

                    for pos in np.arange(0, cumlen[-1], label_interval):
                        idx = np.searchsorted(cumlen, pos)
                        if idx >= len(coords):
                            idx = len(coords) - 1
                        x_coord, y_coord = coords[idx]

                        is_too_close = False
                        for px, py in placed_river_labels:
                            if math.sqrt((x_coord - px) ** 2 + (y_coord - py) ** 2) < min_label_distance_map_units:
                                is_too_close = True
                                break

                        if not is_too_close:
                            ax.text(max(west, min(x_coord, east)), max(south, min(y_coord, north)), name, fontsize=0.2 * 1000 / dpi, color=(0, 0.3, 0.6), alpha=0.8,
                                    zorder=4,
                                    ha='center', va='center', style='italic')
                            placed_river_labels.append((x_coord, y_coord))

        if not smaller_waterways.empty:
            smaller_waterways.plot(
                ax=ax,
                linewidth=0.025,
                edgecolor=(0.3, 0.6, 0.9),
                alpha=0.5,
                zorder=2
            )

    #print("Plotting roads...")
    if roads_gdf is not None and not roads_gdf.empty:
        road_styles = {
            "motorway": {"width": 0.4, "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5), "alpha_fill": 0.75,
                         "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "trunk": {"width": 0.3, "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5), "alpha_fill": 0.75,
                      "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "primary": {"width": 0.3, "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5), "alpha_fill": 0.75,
                        "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "secondary": {"width": 0.2, "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5),
                          "alpha_fill": 0.75, "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "tertiary": {"width": 0.15, "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5),
                         "alpha_fill": 0.75, "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "residential": {"width": 0.075, "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5),
                            "alpha_fill": 0.75, "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "unclassified": {"width": 0.075, "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5),
                             "alpha_fill": 0.75, "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "service": {"width": 0.05, "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5), "alpha_fill": 0.75,
                        "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},

            "footway": {"width": 0.05, "color": (0.5, 0.5, 0.5), "alpha": 0.8, "zorder": 6, "linestyle": '--'},
            "path": {"width": 0.04, "color": (0.6, 0.6, 0.6), "alpha": 0.7, "zorder": 6, "linestyle": ':'},
            "cycleway": {"width": 0.06, "color": (0.4, 0.6, 0.4), "alpha": 0.8, "zorder": 6, "linestyle": '-'},
            "bridleway": {"width": 0.04, "color": (0.6, 0.4, 0.2), "alpha": 0.7, "zorder": 6, "linestyle": '-.'},
            "steps": {"width": 0.03, "color": (0.5, 0.5, 0.5), "alpha": 0.8, "zorder": 6, "linestyle": ':'},
            "pedestrian": {"width": 0.1, "color": (0.5, 0.5, 0.5), "alpha": 0.8, "zorder": 5, "linestyle": '-'},
            "track": {"width": 0.05, "color": (0.7, 0.6, 0.5), "alpha": 0.6, "zorder": 3, "linestyle": '-'},
            "construction": {"width": 0.05, "color": (0.8, 0.5, 0.1), "alpha": 0.5, "zorder": 2, "linestyle": ':'},
            "proposed": {"width": 0.05, "color": (0.5, 0.5, 0.8), "alpha": 0.4, "zorder": 2, "linestyle": '--'},
        }

        for road_type in tqdm(roads_gdf["highway"].unique(), desc="Drawing roads", dynamic_ncols=True, leave=False):
            subset = roads_gdf[roads_gdf["highway"] == road_type]
            if not subset.empty:
                style = road_styles.get(road_type, {
                    "width": 0.05, "fill_color": (0.4, 0.4, 0.4), "line_color": (0.6, 0.6, 0.6),
                    "alpha_fill": 0.6, "alpha_line": 0.8, "zorder_fill": 3, "zorder_line": 4, "linestyle": '-'
                })

                if "color" in style:
                    subset.plot(
                        ax=ax,
                        linewidth=style["width"],
                        edgecolor=style["color"],
                        alpha=style["alpha"],
                        linestyle=style["linestyle"],
                        zorder=style["zorder"]
                    )
                else:
                    subset.plot(
                        ax=ax,
                        linewidth=style["width"],
                        edgecolor=style["fill_color"],
                        alpha=style["alpha_fill"],
                        zorder=style["zorder_fill"],
                        linestyle=style["linestyle"]
                    )
                    subset.plot(
                        ax=ax,
                        linewidth=style["width"] * 0.6,
                        edgecolor=style["line_color"],
                        alpha=style["alpha_line"],
                        zorder=style["zorder_line"],
                        linestyle=style["linestyle"]
                    )

    print("Plotting places and labels...")
    if places_gdf is not None and not places_gdf.empty:
        # Define styles for settlement places (city, town, village)
        # These now use the 'type' column, as it's consistently generated
        settlement_types_map = {
            "city": {"size_scale": 8, "text_scale_factor": 1.5},
            "town": {"size_scale": 7, "text_scale_factor": 1.5},
            "village": {"size_scale": 1.5, "text_scale_factor": 1.5}
        }

        # Define which POI categories you want to plot from the 'type' column
        whichPois = [
            # "monastery", "church", "castle", "monument", "ruins",
            # "archaeological_site"
        ]

        # Filter places_gdf based on its 'type' column
        settlement_places = places_gdf[places_gdf["type"].isin(settlement_types_map.keys())]
        other_pois = places_gdf[places_gdf["type"].isin(whichPois)]

        scatterSize = .2 * 1000 / dpi

        # --- Logic for Cities, Towns, Villages (using 'type' for filtering) ---
        if not settlement_places.empty:
            for place_type, scales in tqdm(settlement_types_map.items(), desc="Drawing settlements", dynamic_ncols=True,
                                           leave=False):
                subset = settlement_places[settlement_places["type"] == place_type]
                if not subset.empty:
                    marker_size = (scatterSize * scales["size_scale"]) ** 2
                    label_size = (scatterSize * (scales["size_scale"] + 4)) * scales["text_scale_factor"]
                    ax.scatter(subset.geometry.x, subset.geometry.y, s=marker_size, c=[0.1, 0.1, 0.1], zorder=6,
                               linewidths=0)
                    for _, row in subset.iterrows():
                        if row.get("name"):  # Only label if a name exists
                            ax.text(row.geometry.x, row.geometry.y, row["name"], fontsize=label_size, ha="center",
                                    va="bottom",
                                    color=[0.1, 0.1, 0.1], zorder=7)

        # --- Logic for other Points of Interest (POIs) (using 'type' for filtering) ---
        print("Plotting other points of interest...")
        if not other_pois.empty:
            poi_marker_styles = {
                "monastery": {"marker": "X", "color": (0.6, 0.2, 0.8), "size_scale": 0.8, "zorder": 6},
                "church": {"marker": "+", "color": (0.8, 0.4, 0.2), "size_scale": 0.7, "zorder": 6},
                "hospital": {"marker": "s", "color": (0.8, 0.1, 0.1), "size_scale": 0.9, "zorder": 6},
                "school": {"marker": "P", "color": (0.1, 0.5, 0.1), "size_scale": 0.7, "zorder": 6},
                "restaurant": {"marker": "*", "color": (0.9, 0.6, 0.0), "size_scale": 0.6, "zorder": 6},
                "cafe": {"marker": "o", "color": (0.7, 0.5, 0.3), "size_scale": 0.5, "zorder": 6},
                "bar": {"marker": "d", "color": (0.3, 0.3, 0.7), "size_scale": 0.5, "zorder": 6},
                "park": {"marker": "H", "color": (0.1, 0.7, 0.1), "size_scale": 0.9, "zorder": 6},
                "castle": {"marker": "^", "color": (0.4, 0.4, 0.4), "size_scale": 1.0, "zorder": 6},
                "monument": {"marker": "v", "color": (0.5, 0.5, 0.5), "size_scale": 0.8, "zorder": 6},
                "ruins": {"marker": "X", "color": (0.7, 0.7, 0.7), "size_scale": 0.7, "zorder": 6},
                "archaeological_site": {"marker": "D", "color": (0.6, 0.3, 0.0), "size_scale": 0.7, "zorder": 6},
                "attraction": {"marker": "P", "color": (0.8, 0.5, 0.0), "size_scale": 0.8, "zorder": 6},
                "museum": {"marker": "s", "color": (0.1, 0.6, 0.6), "size_scale": 0.8, "zorder": 6},
                "artwork": {"marker": "p", "color": (0.5, 0.1, 0.5), "size_scale": 0.6, "zorder": 6},
                "viewpoint": {"marker": "^", "color": (0.0, 0.5, 0.8), "size_scale": 0.7, "zorder": 6},
                "shop": {"marker": "s", "color": (0.7, 0.7, 0.1), "size_scale": 0.6, "zorder": 6},
                "bakery": {"marker": "P", "color": (0.8, 0.7, 0.5), "size_scale": 0.6, "zorder": 6},
                "pedestrian": {"marker": "_", "color": (0.1, 0.1, 0.1), "size_scale": 0.5, "zorder": 5},
                "track": {"marker": "2", "color": (0.7, 0.6, 0.5), "size_scale": 0.4, "zorder": 3},
                "construction": {"marker": "*", "color": (0.8, 0.5, 0.1), "size_scale": 0.5, "zorder": 2},
                "proposed": {"marker": ".", "color": (0.5, 0.5, 0.8), "size_scale": 0.3, "zorder": 2},
            }

            default_poi_style = {"marker": ".", "color": (0.5, 0.5, 0.5), "size_scale": 0.4, "zorder": 5}

            # Loop through the filtered POIs to plot them
            for _, row in tqdm(other_pois.iterrows(), total=len(other_pois), desc="Drawing POIs", dynamic_ncols=True,
                               leave=False):
                poi_type = row.get("type", "unknown")
                style = poi_marker_styles.get(poi_type, default_poi_style)

                marker_size =  (scatterSize * 4 * style["size_scale"]) ** 2
                label_size = (scatterSize * (style["size_scale"] + 2)) * 0.5

                ax.scatter(row.geometry.x, row.geometry.y,
                           s=marker_size,
                           marker=style["marker"],
                           color=style["color"],
                           zorder=style["zorder"],
                           linewidths=0)

                # if row.get("name"):
                #     ax.text(row.geometry.x, row.geometry.y, row["name"],
                #             fontsize=label_size,
                #             ha="center", va="bottom",
                #             color=style["color"],
                #             zorder=style["zorder"] + 1)

    # --- Plot country boundaries ---
    print("Plotting country boundaries...")
    if country_boundaries_gdf is not None and not country_boundaries_gdf.empty:
        country_boundaries_gdf.plot(
            ax=ax,
            facecolor='none',
            edgecolor=(0.7, 0.2, 0.2),
            linewidth=1,
            alpha=.5,
            linestyle='-',
            zorder=10
        )

    fig.axes[0].axis('off')
    plt.tight_layout()

    # query_centroid = Point((west + east) / 2, (south + north) / 2)
    # best_match = country_boundaries_gdf.iloc[country_boundaries_gdf.geometry.distance(query_centroid).idxmin()]
    # minx, miny, maxx, maxy = best_match.geometry.bounds
    # ax.set(xlim=(minx-.1, maxx+.1), ylim=(miny-.1, maxy+.1))
    ax.set(xlim=(west, east), ylim=(south, north))

    mng = plt.get_current_fig_manager()
    try:
        mng.window.wm_geometry(f"{fig_width * dpi}x{fig_height * dpi}+0+0")
    except Exception:
        pass

    fig.canvas.draw_idle()
    print("Plotting complete.")
    return fig, ax


def main():
    print("Starting map generation process...")

    elev_4326, meta_4326 = reproject_npz_to_epsg4326(
        r"2025-08-11_11-14-53\heightmap_z9_lon_20.3920_23.2018_lat_40.4480_42.5521_reslon_0.002747_reslat_0.002057.npz")

    map_data = elev_4326
    south = meta_4326['south']
    north = meta_4326['north']
    west = meta_4326['west']
    east = meta_4326['east']

    # data = np.load(r"2025-08-11_11-14-53\heightmap_z9_lon_20.3920_23.2018_lat_40.4480_42.5521_reslon_0.002747_reslat_0.002057.npz")
    # map_data = data['elevations']  # 2D numpy array
    # south = data['south'].item()  # scalar float
    # north = data['north'].item()
    # west = data['west'].item()
    # east = data['east'].item()
    # # resolution = data['resolution'].item()


    # map_data = np.load('mk_corrected.npy').T
    # map_data -= -28510.299
    # map_data /= ((94022.3 + 28510.299) / 2489)
    # subsample = 2
    # map_s = map_data[::subsample, ::subsample]
    # # lat = np.load('latitudes.npy')
    # # lon = np.load('longitudes.npy')
    # # south = lat[-1][0]
    # # north = lat[0][0]
    # # west = lon[0][0]
    # # east = lon[-1][0]

    subsample = 1
    map_s = map_data[::subsample, ::subsample]


    print(f"Map boundaries: North={north:.2f}, South={south:.2f}, West={west:.2f}, East={east:.2f}")

    places_gdf = load_or_fetch("places", south, north, west, east, get_places_from_osm)
    roads_gdf = load_or_fetch("roads", south, north, west, east, get_roads_from_osm)
    rivers_gdf = load_or_fetch("rivers", south, north, west, east, get_rivers_from_osm)
    water_bodies_gdf = load_or_fetch("water_bodies", south, north, west, east, get_water_bodies_osm2geojson)
    country_boundaries_gdf = load_or_fetch("country_boundaries", south, north, west, east, get_country_boundaries_from_osm)

    scale, dpi = 2, 300 # 1, 2, 1600
    fig, ax = plot_relief_with_features(places_gdf, roads_gdf, rivers_gdf, water_bodies_gdf,country_boundaries_gdf, map_s, south, west, north,
                                    east, dpi=dpi, scale=scale)

    output_filename = f'baseMap_{subsample}_{scale}_{dpi}dpi.png'
    print(f"Saving map to {output_filename}...")
    start_time = time.time()
    fig.savefig(output_filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
    end_time = time.time()
    print(f"Map saved successfully in {end_time - start_time:.2f} seconds.")
    print("Process finished.")


if __name__ == "__main__":
    main()
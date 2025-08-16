import argparse, os, sys, numpy as np, math, matplotlib.pyplot as plt, re
from scipy.ndimage import gaussian_filter
import geopandas as gpd
import pandas as pd
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
from skimage import exposure
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import cKDTree

# colors = [
#     '#2f4f4f',  # dark slate gray (deep water)
#     '#4f6d7a',  # muted steel blue (shallow water)
#     '#7b9e87',  # soft muted green (low land)
#     '#a3b18a',  # olive green (plains)
#     '#c9d6aa',  # pale moss green (foothills)
#     '#ddd6a4',  # muted beige/yellow (low mountains)
#     '#cfc3a6',  # soft brown/gray (mid mountains)
#     '#bfb8a5',  # muted taupe (higher mountains)
#     '#d9d8d7',  # light gray (sub peaks)
#     '#ececec',  # very light gray (high peaks)
#     '#ffffff',  # white (highest peaks)
# ]
# cmap = LinearSegmentedColormap.from_list('contrasty_terrain', colors, N=1024)

gamma = 0.5
cmap = plt.get_cmap('terrain')
x = np.linspace(0.0, 1, 5000)
colors = cmap(x ** gamma)
gamma_cmap = LinearSegmentedColormap.from_list('gamma_terrain', colors, N=5000)
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
    # overpass_url = "https://lz4.overpass-api.de/api/interpreter"
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


def get_structures_from_osm(lat1, lat2, lon1, lon2):
    def structures_geometry_handler(elements):
        if not elements:
            return gpd.GeoDataFrame(columns=["type", "name", "geometry"], crs="EPSG:4326")

        geojson = osm2geojson.json2geojson({"elements": elements})
        gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")

        def get_type(row):
            # Check known columns, fallback to 'unknown'
            if "building" in row and row["building"]:
                return row["building"]
            if "man_made" in row and row["man_made"]:
                return row["man_made"]
            if "amenity" in row and row["amenity"]:
                return row["amenity"]
            if "landuse" in row and row["landuse"]:
                return row["landuse"]
            if "waterway" in row and row["waterway"]:
                return row["waterway"]
            return "unknown"

        # Add 'type' column based on existing columns
        gdf["type"] = gdf.apply(get_type, axis=1)

        # Add 'name' column if missing
        if "name" not in gdf.columns:
            gdf["name"] = None

        # Ensure 'name' column exists (some features might have 'name')
        # If 'name' column exists but has missing values, fill with None
        else:
            gdf["name"] = gdf["name"].fillna(None)

        return gdf[["type", "name", "geometry"]]

    splits = 1
    # Prepare 8×8 tiling
    lats = np.linspace(lat1, lat2, splits+1)
    lons = np.linspace(lon1, lon2, splits+1)

    all_tiles = []
    with tqdm(total=splits**2, desc="Fetching structures in tiles", dynamic_ncols=True) as pbar:
        for i in range(splits):
            for j in range(splits):
                tile_south = lats[i]
                tile_north = lats[i + 1]
                tile_west = lons[j]
                tile_east = lons[j + 1]

                query = """
                [out:json][timeout:180];
                (
                  way["building"]({lat1},{lon1},{lat2},{lon2});
                  relation["building"]({lat1},{lon1},{lat2},{lon2});
                  way["man_made"]({lat1},{lon1},{lat2},{lon2});
                  relation["man_made"]({lat1},{lon1},{lat2},{lon2});
                  way["amenity"="parking"]({lat1},{lon1},{lat2},{lon2});
                  way["landuse"="construction"]({lat1},{lon1},{lat2},{lon2});
                  way["waterway"="dam"]({lat1},{lon1},{lat2},{lon2});
                  relation["waterway"="dam"]({lat1},{lon1},{lat2},{lon2});
                );
                out body;
                >;
                out skel qt;
                """

                gdf_tile = _fetch_data_from_osm(
                    query,
                    (tile_south, tile_north, tile_west, tile_east),
                    structures_geometry_handler,
                    "structures"
                )

                if not gdf_tile.empty:
                    all_tiles.append(gdf_tile)

                pbar.update(1)

    if not all_tiles:
        return gpd.GeoDataFrame(columns=["type", "name", "geometry"], crs="EPSG:4326")

    # Merge and drop duplicates
    combined = gpd.GeoDataFrame(pd.concat(all_tiles, ignore_index=True), crs="EPSG:4326")
    combined = combined.drop_duplicates(subset=["geometry", "type", "name"])

    return combined


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
    # Query gets ways and relations + expands members (nodes and ways) with >;
    query = (
        f'[out:json][timeout:180];'
        f'(way["waterway"~"{waterway_types}"]({lat1},{lon1},{lat2},{lon2});'
        f'relation["waterway"~"{waterway_types}"]({lat1},{lon1},{lat2},{lon2}););'
        f'out body;>;out skel qt;'
    )

    def parse_elements(elements):
        # Collect geometries by osm id, for ways and relations separately
        ways = {}
        nodes = {}
        relations = {}

        # First collect all nodes and ways for geometry reconstruction
        for el in tqdm(elements, desc="Indexing elements", dynamic_ncols=True, leave=False):
            if el['type'] == 'node':
                nodes[el['id']] = (el['lon'], el['lat'])
            elif el['type'] == 'way':
                ways[el['id']] = el
            elif el['type'] == 'relation':
                relations[el['id']] = el

        # Helper to build LineString from way node refs
        def way_to_linestring(way):
            pts = [nodes[nid] for nid in way['nodes'] if nid in nodes]
            return LineString(pts) if len(pts) > 1 else None

        # Build geometries for ways
        way_geoms = {}
        for wid, way in tqdm(ways.items(), desc="Building way geometries", dynamic_ncols=True, leave=False):
            geom = way_to_linestring(way)
            if geom:
                way_geoms[wid] = geom

        # Build geometries for relations by merging member ways
        rel_geoms = {}
        for rid, rel in tqdm(relations.items(), desc="Building relation geometries", dynamic_ncols=True, leave=False):
            member_lines = []
            for mem in rel.get('members', []):
                if mem['type'] == 'way' and mem['ref'] in way_geoms:
                    member_lines.append(way_geoms[mem['ref']])
            if member_lines:
                from shapely.ops import linemerge
                merged = linemerge(member_lines)
                rel_geoms[rid] = merged

        # Compose output features from ways and relations
        features = []
        # Ways first (that are not part of relations)
        rel_way_ids = {mem['ref'] for rel in relations.values() for mem in rel.get('members', []) if
                       mem['type'] == 'way'}
        for wid, geom in tqdm(way_geoms.items(), desc="Composing way features", dynamic_ncols=True, leave=False):
            if wid not in rel_way_ids:
                tags = ways[wid].get('tags', {})
                features.append({
                    "geometry": geom,
                    "waterway": tags.get("waterway", "unknown"),
                    "name": tags.get("name")
                })
        # Relations
        for rid, geom in tqdm(rel_geoms.items(), desc="Composing relation features", dynamic_ncols=True, leave=False):
            tags = relations[rid].get('tags', {})
            features.append({
                "geometry": geom,
                "waterway": tags.get("waterway", "unknown"),
                "name": tags.get("name")
            })

        return gpd.GeoDataFrame(features, crs="EPSG:4326")

    return _fetch_data_from_osm(query, (lat1, lat2, lon1, lon2), parse_elements, "rivers")


def get_water_bodies_osm2geojson(lat1, lat2, lon1, lon2):
    query = (
        f'[out:json][timeout:180];'
        f'('
        f'way["natural"="water"]({lat1},{lon1},{lat2},{lon2});'
        f'relation["natural"="water"]({lat1},{lon1},{lat2},{lon2});'
        f'way["water"~"lake|reservoir|sea|ocean"]({lat1},{lon1},{lat2},{lon2});'
        f'relation["water"~"lake|reservoir|sea|ocean"]({lat1},{lon1},{lat2},{lon2});'
        f'way["natural"="coastline"]({lat1},{lon1},{lat2},{lon2});'
        f')->.a;'
        f'(.a;>;rel.a;>;);'
        f'out meta;'
    )
    def parse_elements(elements):
        geojson = osm2geojson.json2geojson({"elements": elements})
        return gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")

    return _fetch_data_from_osm(query, (lat1, lat2, lon1, lon2), parse_elements, "water bodies")

def get_railroads_osm2geojson(lat1, lat2, lon1, lon2):
    query = (
        f'[out:json][timeout:180];'
        f'('
        f'way["railway"~"rail|light_rail|subway"]({lat1},{lon1},{lat2},{lon2});'
        f'relation["railway"~"rail|light_rail|subway"]({lat1},{lon1},{lat2},{lon2});'
        f');'
        f'out body;'
        f'>;'
        f'out skel qt;'
    )

    def parse_elements(elements):
        if not elements:
            # Return an empty GeoDataFrame with EPSG:4326 CRS and no features
            return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

        geojson = osm2geojson.json2geojson({"elements": elements})
        return gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")

    return _fetch_data_from_osm(query, (lat1, lat2, lon1, lon2), parse_elements, "railroads")


def get_airports_osm2geojson(lat1, lat2, lon1, lon2):
    query = (
        f'[out:json][timeout:180];'
        f'('
        f'node["aeroway"="aerodrome"]({lat1},{lon1},{lat2},{lon2});'
        f'way["aeroway"="aerodrome"]({lat1},{lon1},{lat2},{lon2});'
        f'relation["aeroway"="aerodrome"]({lat1},{lon1},{lat2},{lon2});'
        f');'
        f'out body;'
        f'>;'
        f'out skel qt;'
    )

    def parse_elements(elements):
        geojson = osm2geojson.json2geojson({"elements": elements})
        return gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")

    return _fetch_data_from_osm(query, (lat1, lat2, lon1, lon2), parse_elements, "airports")
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

def get_mountain_peaks_osm2geojson(lat1, lat2, lon1, lon2):
    query = (
        f'[out:json][timeout:180];'
        f'('
        f'node["natural"="peak"]["name"]({lat1},{lon1},{lat2},{lon2});'
        f'way["natural"="peak"]["name"]({lat1},{lon1},{lat2},{lon2});'
        f'relation["natural"="peak"]["name"]({lat1},{lon1},{lat2},{lon2});'
        f')->.a;'
        f'(.a;>;rel.a;>;);'
        f'out meta;'
    )

    def parse_elements(elements):
        geojson = osm2geojson.json2geojson({"elements": elements})
        if not geojson["features"]:
            return gpd.GeoDataFrame(columns=["name", "geometry"], crs="EPSG:4326")
        return gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")

    return _fetch_data_from_osm(query, (lat1, lat2, lon1, lon2), parse_elements, "mountain peaks")


def load_or_fetch(filename_prefix, rasterPath, south, north, west, east, fetch_func):
    filename = f"{filename_prefix}_s{south:.2f}_n{north:.2f}_w{west:.2f}_e{east:.2f}.pkl"
    filename = os.path.join(os.path.dirname(rasterPath), filename)
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


def plot_relief_with_features(places_gdf, roads_gdf, structures_gdf, rivers_gdf, water_bodies_gdf, mountain_peaks_gdf, railroads_gdf, airports_gdf, country_boundaries_gdf, map_s, south, west, north, east, dpi,
                              resolutionFactor, resolution, exagerateTerrain):
    fig_width, fig_height = resolutionFactor * 10, resolutionFactor * 10
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    settlementsFontSize   = 1 * resolutionFactor# master for the largest font, labels of smaller settlements take a fraction of this size
    airportsFontSize      = 0.5 * resolutionFactor
    contoursFontSize      = 0.2 * resolutionFactor
    mountainPeaksFontSize = 0.3 * resolutionFactor
    riverLabelFontSize    = 0.3 * resolutionFactor
    waterBodiesFontSize   = 0.4 * resolutionFactor

    airportMarkerSize       = 1 * resolutionFactor
    settlementMarkerSize    = .5 * resolutionFactor
    mountainPeaksMarkerSize = .2 * resolutionFactor

    if exagerateTerrain:
        print("Terrain exaggeration...")
        # # increase colorization in locally flat regions
        pixelsPerDegree = map_s.shape[0] / np.abs(south-north)
        kernel_size = 0.0125 / 2 # in degrees
        kernel_size = int(kernel_size * pixelsPerDegree) # in pixels
        unsharp_mask = lambda img, sigma=1, strength=1.2: np.clip(
            (((img - img.min()) / (img.max() - img.min())) + strength * (
                        ((img - img.min()) / (img.max() - img.min())) - gaussian_filter(
                    (img - img.min()) / (img.max() - img.min()), sigma))) * (img.max() - img.min()) + img.min(),
            img.min(), img.max()
        )
        ax.imshow(unsharp_mask(np.copy(map_s), sigma=kernel_size, strength=1.0), extent=[west, east, south, north], origin='upper', cmap=gamma_cmap, vmin=-150, interpolation='bilinear')
    else:
        ax.imshow(map_s, extent=[west, east, south, north], origin='upper',
                  cmap=gamma_cmap, vmin=-150, interpolation='bilinear')

    print("Plotting contours...")
    map_s_smooth = map_s # gaussian_filter(map_s, sigma=map_s.shape[0] / 6000)
    x, y = np.linspace(west, east, map_s.shape[1]), np.linspace(north, south, map_s.shape[0])
    X, Y = np.meshgrid(x, y)
    min_val, max_val = np.min(map_s_smooth), np.max(map_s_smooth)
    min_val = 0
    max_val = max_val // 100 * 100 + 100

    levels_mini = np.arange(min_val, max_val, 20)
    levels_thin = np.arange(min_val, max_val, 100)

    print("Drawing mini contours with very thin lines...");
    contours_mini = ax.contour(X, Y, map_s_smooth, levels=levels_mini, colors='0.55', alpha=0.4, linewidths=0.03)
    print(f"Mini contours drawn: {len(contours_mini.collections)} levels")
    print("Drawing thin contours with thin lines...");
    contours_thin = ax.contour(X, Y, map_s_smooth, levels=levels_thin, colors='0.55', alpha=0.4, linewidths=0.05)
    print(f"Thin contours drawn: {len(contours_thin.collections)} levels")

    max_distance_km = 2.0  # spacing between labels

    for level, collection in tqdm(zip(contours_thin.levels, contours_thin.collections),
                                  total=len(contours_thin.levels),
                                  desc="Drawing contour labels", dynamic_ncols=True, leave=False):
        collection.set_rasterized(True)
        # Keep track of label positions per contour level
        placed_labels = []
        for path in collection.get_paths():
            vertices = path.vertices
            if len(vertices) < 2:
                continue
            deltas = [
                haversine_distance(lat1, lon1, lat2, lon2)
                for (lon1, lat1), (lon2, lat2) in zip(vertices[:-1], vertices[1:])
            ]
            cum_dist = 0.0
            for (x1, y1), (x2, y2), seg_len in zip(vertices[:-1], vertices[1:], deltas):
                cum_dist += seg_len
                if cum_dist >= max_distance_km:
                    if cum_dist <= max_distance_km * 1.5:  # avoid labeling at broken contours
                        x_mid, y_mid = (x1 + x2) / 2, (y1 + y2) / 2
                        # Check distance to all previously placed labels of this level
                        if all(haversine_distance(y_mid, x_mid, py, px) >= max_distance_km
                               for py, px in placed_labels):
                            # Calculate slope angle
                            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                            if angle > 90:
                                angle -= 180
                            elif angle < -90:
                                angle += 180
                            ax.text(
                                x_mid, y_mid, f"{level:.0f}",
                                fontsize=contoursFontSize,
                                color="0.55", alpha=0.5,
                                ha="center", va="center",
                                rotation=angle, rotation_mode="anchor",
                                zorder=1, rasterized=True,
                            )
                            placed_labels.append((y_mid, x_mid))  # store as (lat, lon)
                    cum_dist = 0.0
    print("Finished labeling contours.")

    print("Plotting structures...")
    if structures_gdf is not None and not structures_gdf.empty:
        structures_gdf.plot(ax=ax, edgecolor='0.4', facecolor='0.6', alpha=0.5, linewidth=0.05)
        ax.collections[-1].set_rasterized(True)

    print("Plotting water bodies...")
    if water_bodies_gdf is not None and not water_bodies_gdf.empty:
        water_bodies_gdf.plot(ax=ax, facecolor=(0.7, 0.88, 0.96), edgecolor=(0.42, 0.7, 0.84), linewidth=0.1, zorder=3)
        ax.collections[-1].set_rasterized(True)
        def has_valid_water_name(tags):
            if not isinstance(tags, dict):
                return False
            name = tags.get("name")
            if not name:
                return False
            natural = tags.get("natural", "").lower()
            water = tags.get("water", "").lower()
            wtype = tags.get("type", "").lower()
            # Check for relevant water bodies:
            relevant_types = ["lake", "reservoir", "artificial", "glacial", "pond", "reservoir", "sea", "ocean"]
            if natural == "water":
                # Accept if water or type tag matches relevant keywords
                if any(t in water for t in relevant_types) or any(
                        t in wtype for t in relevant_types) or water == "" or wtype == "":
                    return True
            return False
        water_names_gdf = water_bodies_gdf[water_bodies_gdf["tags"].apply(has_valid_water_name)]
        for _, row in water_names_gdf.iterrows():
            label_point = row.geometry.centroid
            if (west <= label_point.x <= east) and (south <= label_point.y <= north):
                name = row["tags"]["name"].replace("/", "\n")
                name = "\n".join(re.split(r"\s+", name.strip())) if len(re.split(r"\s+", name.strip())) == 2 else name
                ax.text(label_point.x, label_point.y, name,
                        fontsize=waterBodiesFontSize,
                        fontfamily='serif',
                        style='italic',
                        color=(0, 0.3, 0.6),
                        alpha=0.8,
                        ha='center',
                        va='center',
                        rasterized = True,
                        zorder=4)

    print("Plotting rivers and their labels...")
    label_step_km = 2.5  # candidate labels every 500m
    max_distance_km = 2.5  # minimum distance between river labels
    segment_length_km = 0.5  # 500 meters for label rotation calculation

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
            ax.collections[-1].set_rasterized(True)

            placed_river_labels = []

            for _, row in tqdm(rivers_only.iterrows(), total=len(rivers_only), desc="Labeling rivers",
                               dynamic_ncols=True, leave=False):
                name, geom = row.get("name"), row.geometry
                if not name or geom.is_empty:
                    continue
                words = re.split(r"[ -]", name)
                name = "\n".join(" ".join(words[i:i + 2]) for i in range(0, len(words), 2))

                lines = [
                    geom] if geom.geom_type == 'LineString' else geom.geoms if geom.geom_type == 'MultiLineString' else []
                for line in lines:
                    coords = np.array(line.coords)
                    if len(coords) < 2:
                        continue

                    # compute cumulative distance along the line in km
                    cumlen_km = [0.0]
                    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
                        cumlen_km.append(cumlen_km[-1] + haversine_distance(y1, x1, y2, x2))
                    cumlen_km = np.array(cumlen_km)

                    # window = 5  # smoothing for angle
                    pos = 0.0
                    while pos <= cumlen_km[-1]:
                        idx = np.searchsorted(cumlen_km, pos)
                        idx = min(idx, len(coords) - 1)
                        x, y = coords[idx]

                        # Determine start and end indices for rotation using a ~100m segment
                        start_cum = max(cumlen_km[idx] - segment_length_km / 2, 0)
                        end_cum = min(cumlen_km[idx] + segment_length_km / 2, cumlen_km[-1])
                        start_idx = np.searchsorted(cumlen_km, start_cum)
                        end_idx = np.searchsorted(cumlen_km, end_cum)

                        p1 = coords[start_idx]
                        p2 = coords[end_idx]

                        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
                        if angle < -90:
                            angle += 180
                        elif angle > 90:
                            angle -= 180

                        # check distance to all previously placed river labels
                        if all(haversine_distance(y, x, py, px) >= max_distance_km for py, px in placed_river_labels):
                            ax.text(
                                max(west, min(x, east)),
                                max(south, min(y, north)),
                                name,
                                fontsize=riverLabelFontSize,
                                color=(0, 0.3, 0.6),
                                alpha=0.8,
                                zorder=4,
                                ha='center',
                                va='top',
                                style='italic',
                                fontfamily='serif',
                                rotation=angle,
                                rasterized=True,
                            )
                            placed_river_labels.append((y, x))  # store as (lat, lon)

                        pos += label_step_km  # move to next candidate along river

        if not smaller_waterways.empty:
            smaller_waterways.plot(
                ax=ax,
                linewidth=0.025,
                edgecolor=(0.3, 0.6, 0.9),
                alpha=0.5,
                zorder=2
            )
            ax.collections[-1].set_rasterized(True)

    print("Plotting roads...")
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
                    ax.collections[-1].set_rasterized(True)
                else:
                    subset.plot(
                        ax=ax,
                        linewidth=style["width"],
                        edgecolor=style["fill_color"],
                        alpha=style["alpha_fill"],
                        zorder=style["zorder_fill"],
                        linestyle=style["linestyle"]
                    )
                    ax.collections[-1].set_rasterized(True)
                    subset.plot(
                        ax=ax,
                        linewidth=style["width"] * 0.6,
                        edgecolor=style["line_color"],
                        alpha=style["alpha_line"],
                        zorder=style["zorder_line"],
                        linestyle=style["linestyle"]
                    )
                    ax.collections[-1].set_rasterized(True)

    # Railroads: plot as thin dark gray lines
    print("Plotting railroads...")
    if railroads_gdf is not None and not railroads_gdf.empty:
        railroads_gdf.plot(
            ax=ax,
            color='black',
            linewidth=0.2,
            alpha=0.3,
            linestyle=(0, (2, 2)),  # long dashes: 2 on, 2 off
            zorder=5
        )
        ax.collections[-1].set_rasterized(True)
        # Optionally add a gray shadow line beneath for style/depth:
        railroads_gdf.plot(
            ax=ax,
            color='gray',
            linewidth=0.2,
            alpha=0.4,
            zorder=4
        )
        ax.collections[-1].set_rasterized(True)

    print("Plotting places and labels...")
    if places_gdf is not None and not places_gdf.empty:
        # Define styles for settlement places (city, town, village)
        # These now use the 'type' column, as it's consistently generated
        settlement_types_map = {
            "city":    {"settlementMarkerSize": settlementMarkerSize,         "settlementsFontSize": settlementsFontSize,         "style":'normal', "color":[0.1,0.1,0.1]},
            "town":    {"settlementMarkerSize": settlementMarkerSize * 0.8,   "settlementsFontSize": settlementsFontSize * 0.8,   "style":'normal', "color":[0.1,0.1,0.1]},
            "village": {"settlementMarkerSize": settlementMarkerSize * 0.4,   "settlementsFontSize": settlementsFontSize * 0.5,   "style":'normal', "color":[0.1,0.1,0.1]},
            "suburb":  {"settlementMarkerSize": settlementMarkerSize * 0,     "settlementsFontSize": settlementsFontSize * 0.3,   "style":'italic', "color":[0.25,0.25,0.25]},
            "hamlet":  {"settlementMarkerSize": settlementMarkerSize * 0,     "settlementsFontSize": settlementsFontSize * 0.3,   "style":'italic', "color":[0.25,0.25,0.25]}
        }

        # Define which POI categories you want to plot from the 'type' column
        whichPois = [
            # "monastery", "church", "castle", "monument", "ruins",
            # "archaeological_site"
        ]

        # Filter places_gdf based on its 'type' column
        settlement_places = places_gdf[places_gdf["type"].isin(settlement_types_map.keys())]
        # other_pois = places_gdf[places_gdf["type"].isin(whichPois)]

        # --- Logic for Cities, Towns, Villages (using 'type' for filtering) ---
        if not settlement_places.empty:
            pbar = tqdm(settlement_types_map.items(), dynamic_ncols=True, leave=False)
            for place_type, scales in pbar:
                pbar.set_description(f"Drawing settlements {place_type}"); pbar.update(1)
                subset = settlement_places[settlement_places["type"] == place_type]
                if not subset.empty:
                    marker_size = scales["settlementMarkerSize"] ** 2
                    label_size = scales["settlementsFontSize"]
                    text_style = scales['style']
                    color = scales['color']
                    ax.scatter(subset.geometry.x, subset.geometry.y, s=marker_size, color=(0.1, 0.1, 0.1), zorder=6,
                               linewidths=0)
                    for _, row in subset.iterrows():
                        if row.get("name"):  # Only label if a name exists
                            ax.text(row.geometry.x, row.geometry.y, row["name"], fontsize=label_size, ha="center",
                                    va="bottom", rasterized = True, style=text_style,
                                    color=color, zorder=7)

    # Airports: plot as dark blue plane markers (triangle up or custom marker)
    print("Plotting airports...")
    if airports_gdf is not None and not airports_gdf.empty:
        marker_size = airportMarkerSize

        for _, row in airports_gdf.iterrows():
            geom = row.geometry
            # Get centroid for polygons, or use point directly
            if geom.geom_type == 'Point':
                x, y = geom.x, geom.y
            else:
                centroid = geom.centroid
                x, y = centroid.x, centroid.y

            ax.text(x, y, "✈", fontsize=marker_size, ha='center', va='top',
                    color='darkblue', zorder=7, alpha=.8, rasterized = True)

            # Optionally label airports by name
            words = re.split(r"[ -]", row.get('tags', {}).get('name', ''))
            name = "\n".join(" ".join(words[i:i + 2]) for i in range(0, len(words), 2))
            if name:
                ax.text(x, y, name,
                        fontsize=airportsFontSize,
                        ha='center', va='bottom',
                        color='darkblue', fontfamily='serif', style='italic',
                        zorder=8, alpha=.8, rasterized = True)

    # Plot mountain peaks as dark green triangles with village marker size
    print("Plotting mountain peaks...")
    if not mountain_peaks_gdf.empty:
        peaks_settings_map = {
            "mountain_peak": {"size_scale": 6.0, "text_scale_factor": 0.25},
        }
        ax.scatter(mountain_peaks_gdf.geometry.x, mountain_peaks_gdf.geometry.y, s=(mountainPeaksMarkerSize) ** 2, c='darkgreen',
                   marker='^', zorder=6, linewidths=0)
        for _, row in mountain_peaks_gdf.iterrows():
            if row.get("tags")['name']:
                name = row.get("tags").get('name', '')
                ele = row.get("tags").get('ele', None)
                if ele is not None:
                    label_text = f"{name}\n{ele} m"
                else:
                    label_text = name
                ax.text(row.geometry.x, row.geometry.y, label_text, fontsize=mountainPeaksFontSize, ha="center", va="bottom",
                        color='darkgreen', fontfamily='serif', style='italic', zorder=7, rasterized = True)

    # --- Plot country boundaries ---
    print("Plotting country boundaries...")
    if country_boundaries_gdf is not None and not country_boundaries_gdf.empty:
        country_boundaries_gdf.plot( # red thick line
            ax=ax,
            facecolor='none',
            edgecolor=(0.7, 0.2, 0.2),
            linewidth=1,
            alpha=.5,
            linestyle='-',
            zorder=4
        )
        ax.collections[-1].set_rasterized(True)
        country_boundaries_gdf.plot( # black dashed thin line
            ax=ax,
            facecolor='none',
            edgecolor=(0.1, 0.1, 0.1),
            linewidth=.25,
            alpha=.9,
            linestyle=(0, (3,5,1,5)), # dashdotted
            joinstyle='round',
            zorder=5
        )
        ax.collections[-1].set_rasterized(True)

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
    # rasterPath = r"./2025-08-14_11-00-31/heightmap_z11_lon_20.3910_23.2028_lat_40.7142_42.5528_reslon_0.000687_reslat_0.000513.npz" # NMK lowres
    rasterPath = r"./2025-08-14_16-17-43/heightmap_z12_lon_20.3908_23.1151_lat_40.7807_42.4882_reslon_0.000343_reslat_0.000257.npz"  # NMK hires
    ### hires settings
    ### We have to use a scaling trick in order to render small fonts (less than 1pt)
    # resolutionFactor, dpi = 5, int(640)
    # resolutionFactor, dpi = 4, int(800)
    # resolutionFactor, dpi = 3, int(1066)
    # resolutionFactor, dpi = 2, int(1600)
    resolutionFactor, dpi = 1.5, int(1500) # good middle ground
    ### lowres settings
    # resolutionFactor, dpi = 2, int(640)


    print("Starting map generation process...")
    #mapzen tiles usually come in as Web Mercator, projcet to lat-lon rectangular projection
    elev_4326, meta_4326 = reproject_npz_to_epsg4326(rasterPath)

    map_data = elev_4326
    south = meta_4326['south']
    north = meta_4326['north']
    west = meta_4326['west']
    east = meta_4326['east']
    resolution = meta_4326['resolution_lon_deg']

    print("Raster and metadata loaded!")
    subsample = 1
    map_s = map_data[::subsample, ::subsample]

    print(f"Map boundaries: North={north:.2f}, South={south:.2f}, West={west:.2f}, East={east:.2f}")

    places_gdf             = load_or_fetch("places", rasterPath, south, north, west, east, get_places_from_osm)
    roads_gdf              = load_or_fetch("roads", rasterPath, south, north, west, east, get_roads_from_osm)
    structures_gdf         = load_or_fetch("structures", rasterPath, south, north, west, east, get_structures_from_osm)
    rivers_gdf             = load_or_fetch("rivers", rasterPath, south, north, west, east, get_rivers_from_osm)
    water_bodies_gdf       = load_or_fetch("water_bodies", rasterPath, south, north, west, east, get_water_bodies_osm2geojson)
    mountain_peaks_gdf     = load_or_fetch("mountain_peaks", rasterPath, south, north, west, east, get_mountain_peaks_osm2geojson)
    railroads_gdf          = load_or_fetch("railroads", rasterPath, south, north, west, east, get_railroads_osm2geojson)
    airports_gdf           = load_or_fetch("airports", rasterPath, south, north, west, east, get_airports_osm2geojson)
    country_boundaries_gdf = load_or_fetch("country_boundaries", rasterPath, south, north, west, east, get_country_boundaries_from_osm)

    # Apply color exaggeration
    exagerateTerrain = True
    fig, ax = plot_relief_with_features(places_gdf, roads_gdf, structures_gdf, rivers_gdf, water_bodies_gdf, mountain_peaks_gdf, railroads_gdf, airports_gdf, country_boundaries_gdf, map_s, south, west, north,
                                    east, dpi=dpi, resolutionFactor=resolutionFactor, resolution=resolution, exagerateTerrain=exagerateTerrain)

    output_filename = (
        f'baseMap_{subsample}_{resolutionFactor}_{dpi}dpi_{resolution}_ex={exagerateTerrain}'
        f'_E={format(east, ".3f").replace(".", ",")}'
        f'_W={format(west, ".3f").replace(".", ",")}'
        f'_N={format(north, ".3f").replace(".", ",")}'
        f'_S={format(south, ".3f").replace(".", ",")}.png'
    )
    print(f"Saving map to {output_filename}...")
    start_time = time.time()
    fig.savefig(output_filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
    end_time = time.time()
    print(f"Map saved successfully in {end_time - start_time:.2f} seconds.")
    print("Process finished.")


if __name__ == "__main__":
    main()
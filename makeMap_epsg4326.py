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
# from pyproj import datadir
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from skimage import exposure
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import cKDTree
from shapely.ops import polygonize, unary_union  # Added for polygonize
import colorsys
import matplotlib.patheffects as patheffects
from matplotlib.colors import LightSource

# projDataDirPath = datadir.get_data_dir()
# os.environ['PROJ_DATA'] = projDataDirPath

## origianl CM terrain
# terrain_colors = [
#     (0.00, (0.192,0.208,0.608)), # deep blue
#     (0.152, (0.0, 0.604, 0.98)), # sea color
#     (0.262, (0.047, 0.808, 0.408)), # green
#     (0.42, (0.659, 0.929, 0.529)), # bright green
#     (0.51, (0.973, 0.965, 0.584)), # yellow
#     (0.75, (0.502, 0.361, 0.329)), # brown
#     (1.00, (1.0, 1.0, 1.0)), # white
# ]
terrain_colors = [
    (0.000, (0.192,0.208,0.608)), # deep blue
    (0.050, (0.0, 0.604, 0.98)), # sea color
    (0.300, (0.247, 0.608, 0.308)), # darker green
    (0.400, (0.347, 0.708, 0.408)), # green
    (0.500, (0.689, 0.829, 0.529)), # bright green
    (0.550, (0.973, 0.935, 0.584)), # yellow
    (0.750, (0.502, 0.361, 0.329)), # brown
    (0.950, (0.9, 0.9, 1.0)), # white
    (1.000, (1.0, 1.0, 1.0)), # white
]
f = 1.15  # value boost factor
terrain_colors_boosted = [
    (h, colorsys.hsv_to_rgb(*(*colorsys.rgb_to_hsv(*rgb)[:2], min(1.0, colorsys.rgb_to_hsv(*rgb)[2]*f))))
    for h, rgb in terrain_colors
]
cmap = LinearSegmentedColormap.from_list("compressed_snow_terrain", terrain_colors_boosted)

gamma = 0.45
# cmap = plt.get_cmap('terrain')
x = np.linspace(0.0, 1, 5000)
colors = cmap(x ** gamma)
gamma_cmap = LinearSegmentedColormap.from_list('gamma_terrain', colors, N=5000)


def sanitize_filename(filename):
    """
    Removes invalid characters from a string to make it a valid filename.
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')
    return filename

def haversine_distance(lat1, lon1, lat2, lon2, radius=6371.0):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c

def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


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
            # For water bodies, ensure 'tags' column is present even if empty
            elif desc == "water bodies":
                return gpd.GeoDataFrame(columns=['geometry', 'tags'], crs="EPSG:4326")
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
    lats = np.linspace(lat1, lat2, splits + 1)
    lons = np.linspace(lon1, lon2, splits + 1)

    all_tiles = []
    with tqdm(total=splits ** 2, desc="Fetching structures in tiles", dynamic_ncols=True) as pbar:
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
    # query = f'[out:json];way["highway"~"motorway|primary|secondary|tertiary|trunk|residential|unclassified|service"]({lat1},{lon1},{lat2},{lon2});out geom;'
    return _fetch_data_from_osm(query, (lat1, lat2, lon1, lon2), lambda elements: gpd.GeoDataFrame([
        {"highway": el["tags"].get("highway"),
         "geometry": LineString([(pt["lon"], pt["lat"]) for pt in el["geometry"]])}
        for el in tqdm(elements, desc="Processing road elements", dynamic_ncols=True, leave=False) if
        "geometry" in el and len(el["geometry"]) > 1
    ], crs="EPSG:4326"), "roads")


def get_places_from_osm(lat1, lat2, lon1, lon2):
    # query = f"""
    # [out:json];
    # (
    #   node["place"]({lat1},{lon1},{lat2},{lon2});
    #   node["amenity"~"monastery|church|hospital|school|restaurant|cafe|bar|park"]({lat1},{lon1},{lat2},{lon2});
    #   node["historic"~"castle|monument|ruins|archaeological_site"]({lat1},{lon1},{lat2},{lon2});
    #   node["tourism"~"attraction|museum|artwork|viewpoint"]({lat1},{lon1},{lat2},{lon2});
    #   node["shop"]({lat1},{lon1},{lat2},{lon2});
    # );
    # out;
    # """
    query = f"""
    [out:json];
    (
      node["place"]({lat1},{lon1},{lat2},{lon2});      
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
        f'node["aeroway"~"aerodrome|runway|apron|taxiway"]({lat1},{lon1},{lat2},{lon2});'
        f'way["aeroway"~"aerodrome|runway|apron|taxiway"]({lat1},{lon1},{lat2},{lon2});'
        f'relation["aeroway"~"aerodrome|runway|apron|taxiway"]({lat1},{lon1},{lat2},{lon2});'
        f');'
        f'out body;'
        f'>;'
        f'out skel qt;'
    )

    def parse_elements(elements):
        geojson = osm2geojson.json2geojson({"elements": elements})
        if not geojson["features"]:
            return gpd.GeoDataFrame(columns=["name", "geometry"], crs="EPSG:4326")
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


def plot_relief_with_features(places_gdf, roads_gdf, structures_gdf, rivers_gdf, water_bodies_gdf, mountain_peaks_gdf,
                              railroads_gdf, airports_gdf, country_boundaries_gdf, map_s, south, west, north, east, dpi,
                              resolutionFactor, resolution, exagerateTerrain):
    fig_width, fig_height = resolutionFactor * 10, resolutionFactor * 10
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    settlementsFontSize = 1 * resolutionFactor  # master for the largest font, labels of smaller settlements take a fraction of this size
    airportsFontSize = 0.5 * resolutionFactor
    contoursFontSize = 0.2 * resolutionFactor
    mountainPeaksFontSize = 0.2 * resolutionFactor
    riverLabelFontSize = 0.3 * resolutionFactor
    waterBodiesFontSize = 0.4 * resolutionFactor

    airportMarkerSize = 1 * resolutionFactor
    settlementMarkerSize = .5 * resolutionFactor
    mountainPeaksMarkerSize = .2 * resolutionFactor

    # pre smooth the terrain map to avoid exaggerating noise
    # smoothed = lambda arr, w: gaussian_filter(arr, sigma=w / 6, truncate=w / (2 * (w / 6)))
    # map_s_smooth = smoothed(map_s, 3)
    map_s_smooth = map_s

    if exagerateTerrain:
        print("Terrain exaggeration by hill shading...")
        ls = LightSource(azdeg=315, altdeg=45)
        map_s_colored = (255*ls.shade(map_s_smooth, blend_mode='soft', vert_exag=1, fraction=.5, cmap=gamma_cmap, vmin=-np.max(map_s) * 0.05)).astype(np.uint8)
        ax.imshow(map_s_colored, extent=[west, east, south, north], origin='upper', interpolation='bilinear')
    else:
        ax.imshow(map_s, extent=[west, east, south, north], origin='upper',
                  cmap=gamma_cmap, interpolation='bilinear', vmin=-np.max(map_s)*0.05)

    print("Plotting contours...")
    x, y = np.linspace(west, east, map_s.shape[1]), np.linspace(north, south, map_s.shape[0])
    X, Y = np.meshgrid(x, y)
    min_val, max_val = np.min(map_s_smooth), np.max(map_s_smooth)
    min_val = 0
    max_val = max_val // 100 * 100 + 100

    levels_mini = np.arange(min_val, max_val, 20)
    levels_thin = np.arange(min_val, max_val, 100)

    print("Drawing mini contours with very thin lines...");
    contours_mini = ax.contour(X, Y, map_s_smooth, levels=levels_mini, colors='0.65', alpha=0.4, linewidths=0.02)
    print(f"Mini contours drawn: {len(contours_mini.collections)} levels")
    print("Drawing thin contours with thin lines...");
    contours_thin = ax.contour(X, Y, map_s_smooth, levels=levels_thin, colors='0.65', alpha=0.5, linewidths=0.05)
    print(f"Thin contours drawn: {len(contours_thin.collections)} levels")

    max_distance_km = 2.0  # spacing between labels
    for level_idx, (level, collection) in enumerate(
            tqdm(zip(contours_thin.levels, contours_thin.collections),
                 total=len(contours_thin.levels),
                 desc="Drawing contour labels",
                 position=0,  # Explicitly set position for the outer bar
                 leave=True,
                 dynamic_ncols=False)):

        collection.set_rasterized(True)
        placed_labels = []

        # Count all segments for this level
        total_segments = sum(
            max(len(path.vertices) - 1, 0) for path in collection.get_paths()
        )
        seg_bar = tqdm(total=total_segments,
                       desc=f"{level}m",
                       position=1,  # Place the inner bar on the next line
                       leave=False,
                       dynamic_ncols=False,
                       mininterval=0.1)

        for path_idx, path in enumerate(collection.get_paths()):
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
                seg_bar.update(1)

                if cum_dist >= max_distance_km:
                    if cum_dist <= max_distance_km * 1.5:
                        x_mid, y_mid = (x1 + x2) / 2, (y1 + y2) / 2
                        if all(haversine_distance(y_mid, x_mid, py, px) >= max_distance_km
                               for py, px in placed_labels):
                            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                            if angle > 90:
                                angle -= 180
                            elif angle < -90:
                                angle += 180
                            ax.text(
                                x_mid, y_mid, f"{level:.0f}",
                                fontsize=contoursFontSize,
                                color="0.65", alpha=0.6,
                                ha="center", va="center",
                                rotation=angle, rotation_mode="anchor",
                                zorder=1, rasterized=True,
                            )
                            placed_labels.append((y_mid, x_mid))
                    cum_dist = 0.0

        seg_bar.close()
    print("Finished labeling contours.")

    print("Plotting structures...")
    if structures_gdf is not None and not structures_gdf.empty:
        structures_gdf.plot(ax=ax, edgecolor='0.4', facecolor='0.6', alpha=0.5, linewidth=0.05)
        ax.collections[-1].set_rasterized(True)

    print("Plotting water bodies...")
    if water_bodies_gdf is not None and not water_bodies_gdf.empty:
        water_bodies_gdf.plot(ax=ax, facecolor=(0.7, 0.88, 0.96),
                              edgecolor=(0.21, 0.55, 0.77),
                              linewidth=0.1, zorder=3, rasterized=True)

        def has_valid_water_name(tags):
            if not isinstance(tags, dict):
                return False
            name = tags.get("name")
            if not name:
                return False
            natural = tags.get("natural", "").lower()
            water = tags.get("water", "").lower()
            wtype = tags.get("type", "").lower()
            relevant = {"lake", "reservoir", "artificial", "glacial", "pond", "sea", "ocean"}
            return natural == "water" and (water in relevant or wtype in relevant or not water or not wtype)

        # Filter for label-worthy water bodies, sort by size (descending)
        water_names_gdf = water_bodies_gdf[water_bodies_gdf["tags"].apply(has_valid_water_name)]
        # Re-project to a projected CRS
        water_names_gdf_projected = water_names_gdf.to_crs("EPSG:3857")

        # Calculate the area in square meters and assign it back to the original GeoDataFrame
        water_names_gdf = water_names_gdf.assign(area=water_names_gdf_projected.geometry.area)

        # Sort by the new, correct area
        water_names_gdf = water_names_gdf.sort_values("area", ascending=False)

        min_water_label_distance_km = 2.0  # tweak for density control
        placed_coords_deg = []  # Store placed coordinates in degrees (lat, lon)

        # Lists to store data for cKDTree and text plotting
        texts_for_plot = []
        coords_deg_for_tree = []  # Store coordinates in degrees (lat, lon) for the tree

        for _, row in water_names_gdf.iterrows():
            geom = row.geometry.representative_point()
            x, y = geom.x, geom.y  # x is longitude, y is latitude

            if not (west <= x <= east and south <= y <= north):
                continue

            name = row["tags"]["name"].replace("/", "\n")
            words = re.split(r"\s+", name.strip())
            if len(words) == 2:
                name = "\n".join(words)

            texts_for_plot.append((x, y, name, row['area']))  # Add area to the tuple
            coords_deg_for_tree.append((y, x))  # Store lat, lon in degrees

        if texts_for_plot:
            coords_deg_for_tree_np = np.array(coords_deg_for_tree)

            # New logic to handle font size adaptation
            areas = [t[3] for t in texts_for_plot]
            min_area = min(areas)
            max_area = max(areas)

            # Check to prevent division by zero if all areas are the same
            area_range = max_area - min_area

            # Convert coordinates to radians for the Haversine KDTree
            coords_rad_for_tree = np.radians(coords_deg_for_tree_np)
            tree = cKDTree(coords_rad_for_tree)

            # Earth's radius in kilometers
            earth_radius_km = 6371.0
            # Convert desired distance in km to radians
            radius_rad = min_water_label_distance_km / earth_radius_km

            for (x_lon, y_lat, label, area) in texts_for_plot:
                # Get the original lat, lon in degrees for this point
                current_lat_deg, current_lon_deg = y_lat, x_lon

                # Convert the current point to radians for querying the KDTree
                current_lat_rad = np.radians(current_lat_deg)
                current_lon_rad = np.radians(current_lon_deg)

                # Query the tree for points within the specified radius (in radians)
                # This will return indices of points within the Haversine distance
                idxs = tree.query_ball_point([current_lat_rad, current_lon_rad], r=radius_rad)

                # Check if any of the nearby points (including itself) have already been placed
                # We compare the original degree coordinates for this check
                if all(tuple(coords_deg_for_tree_np[i]) not in placed_coords_deg for i in idxs):
                    # Calculate the scaling factor based on area
                    if area_range > 0:
                        normalized_area = (area - min_area) / area_range
                        # Scaling factor from 1.0 to 3.0
                        area_scale_factor = 1.0 + 2*normalized_area
                    else:
                        # If all areas are the same, use a factor of 1.0
                        area_scale_factor = 1.0

                    # Adjust the font size
                    scaled_font_size = waterBodiesFontSize * area_scale_factor

                    ax.text(x_lon, y_lat, label,
                            fontsize=scaled_font_size,  # Use the scaled font size
                            fontfamily="serif", style="italic",
                            color=(0, 0.3, 0.6), alpha=0.8,
                            ha="center", va="center",
                            path_effects=[
                                patheffects.withStroke(linewidth=waterBodiesFontSize * 0.1, foreground=(0.2, 0.5, 0.8),
                                                       capstyle="round")],
                            rasterized=True, zorder=4)
                    # Add the degree coordinates of the placed label to the list
                    placed_coords_deg.append((current_lat_deg, current_lon_deg))

    print("Plotting rivers and their labels...")
    # Parameters are now in meters for consistency with projected CRS
    label_step_m = 2500  # candidate labels every 2500 meters (2.5km)
    max_distance_m = 2500  # minimum distance between river labels
    segment_length_m = 1500  # for label rotation calculation

    if rivers_gdf is not None and not rivers_gdf.empty:
        projected_rivers_gdf = rivers_gdf.to_crs(epsg=32631)

        rivers_only = projected_rivers_gdf[projected_rivers_gdf["waterway"].isin(["river", "stream"])]
        smaller_waterways = projected_rivers_gdf[
            projected_rivers_gdf["waterway"].isin(["channel", "irrigation", "canal", "derelict_canal", "ditch", "drain", ""])]

        if not rivers_only.empty:
            rivers_only.to_crs(epsg=4326).plot(  # Plot back in the original CRS for correct map display
                ax=ax,
                linewidth=0.1,
                edgecolor=(0.21, 0.55, 0.77),
                alpha=0.7,
                zorder=2
            )
            ax.collections[-1].set_rasterized(True)

            # Vectorized label placement
            valid_rivers = rivers_only[rivers_only['name'].notna() & (rivers_only.geometry.length > 0)]

            candidate_labels = []
            for _, row in tqdm(valid_rivers.iterrows(), total=len(valid_rivers), desc="Generating river labels", dynamic_ncols=True, leave=False):
                name, geom = row.get("name"), row.geometry

                lines = []
                if geom.geom_type == 'LineString':
                    lines = [geom]
                elif geom.geom_type == 'MultiLineString':
                    lines = list(geom.geoms)
                else:
                    continue

                for line in lines:
                    length_m = line.length
                    # Use numpy to generate positions along the line efficiently
                    for dist in np.arange(label_step_m, length_m, label_step_m):
                        try:
                            point_on_line = line.interpolate(dist, normalized=False)
                            if not point_on_line.is_empty:

                                # Get a segment for angle calculation
                                start_dist = max(0, dist - segment_length_m / 2)
                                end_dist = min(length_m, dist + segment_length_m / 2)
                                start_point = line.interpolate(start_dist, normalized=False)
                                end_point = line.interpolate(end_dist, normalized=False)

                                x, y = point_on_line.x, point_on_line.y
                                angle = np.degrees(np.arctan2(end_point.y - start_point.y, end_point.x - start_point.x))
                                if angle < -90:
                                    angle += 180
                                elif angle > 90:
                                    angle -= 180

                                candidate_labels.append({
                                    'x': x,
                                    'y': y,
                                    'name': name,
                                    'angle': angle
                                })
                        except:
                            continue

            # Proximity filtering using a spatial index for massive speedup
            placed_coords = np.empty((0, 2))  # Initialize as an empty NumPy array with 2 columns
            indices_to_keep = []

            # Filter candidate labels for unique positions
            if candidate_labels:
                candidate_df = gpd.GeoDataFrame(candidate_labels, geometry=[Point(d['x'], d['y']) for d in candidate_labels], crs="EPSG:32631")

                # Check each candidate against already placed labels
                for i, row in candidate_df.iterrows():
                    y, x = row.geometry.y, row.geometry.x

                    # Use a simple Euclidean distance check here, as the CRS is now projected
                    if placed_coords.size == 0:
                        dist_to_placed = np.array([])
                    else:
                        dist_to_placed = np.linalg.norm(placed_coords - np.array([y, x]), axis=1)

                    if dist_to_placed.size == 0 or dist_to_placed.min() >= max_distance_m:
                        indices_to_keep.append(i)
                        placed_coords = np.vstack([placed_coords, [y, x]])

                # Plot the filtered labels
                for idx in tqdm(indices_to_keep, desc="Plotting river labels", dynamic_ncols=True, leave=False):
                    label = candidate_df.loc[idx]
                    words = re.split(r"[ -]", label['name'])
                    name = "\n".join(" ".join(words[i:i + 2]) for i in range(0, len(words), 2))

                    # Create a temporary GeoDataFrame with the single point to use .to_crs()
                    point_gdf = gpd.GeoDataFrame([label], crs=candidate_df.crs)

                    # Convert the label's projected coordinates back to lat/lon for plotting on the basemap
                    latlon_point = point_gdf.to_crs(epsg=4326)

                    ax.text(
                        max(west, min(latlon_point.geometry.x.iloc[0], east)),
                        max(south, min(latlon_point.geometry.y.iloc[0], north)),
                        name,
                        fontsize=riverLabelFontSize,
                        color=(0.11, 0.45, 0.67),
                        path_effects=[patheffects.withStroke(linewidth=waterBodiesFontSize * 0.1, foreground=(0.21, 0.55, 0.77), capstyle="round")],
                        alpha=0.8,
                        zorder=6,
                        ha='center',
                        va='top',
                        style='italic',
                        fontfamily='serif',
                        rotation=label['angle'],
                        rasterized=True,
                    )

        if not smaller_waterways.empty:
            smaller_waterways.to_crs(epsg=4326).plot(
                ax=ax,
                linewidth=0.08,
                edgecolor=(0.21, 0.55, 0.77),
                alpha=0.7,
                zorder=2
            )
            ax.collections[-1].set_rasterized(True)

    print("Plotting roads...")
    if roads_gdf is not None and not roads_gdf.empty:
        road_styles = {
            "motorway":    {"width": 0.4,   "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5), "alpha_fill": 0.75, "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "trunk":       {"width": 0.3,   "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5), "alpha_fill": 0.75, "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "primary":     {"width": 0.3,   "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5), "alpha_fill": 0.75, "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "secondary":   {"width": 0.25,   "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5), "alpha_fill": 0.75, "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "tertiary":    {"width": 0.2,  "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5), "alpha_fill": 0.75, "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "residential": {"width": 0.1, "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5), "alpha_fill": 0.75, "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "unclassified":{"width": 0.1, "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5), "alpha_fill": 0.75, "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},
            "service":     {"width": 0.1,  "fill_color": (0.3, 0.3, 0.3), "line_color": (0.7, 0.7, 0.5), "alpha_fill": 0.75, "alpha_line": 1, "zorder_fill": 4, "zorder_line": 5, "linestyle": '-'},

            "footway":      {"width": 0.1, "color": (0.5, 0.5, 0.5), "alpha": 0.8, "zorder": 4, "linestyle": '--'},
            "path":         {"width": 0.1, "color": (0.6, 0.6, 0.6), "alpha": 0.7, "zorder": 4, "linestyle": ':'},
            "cycleway":     {"width": 0.1, "color": (0.4, 0.6, 0.4), "alpha": 0.8, "zorder": 4, "linestyle": '-'},
            "bridleway":    {"width": 0.1, "color": (0.6, 0.4, 0.2), "alpha": 0.7, "zorder": 4, "linestyle": '-.'},
            "steps":        {"width": 0.07, "color": (0.5, 0.5, 0.5), "alpha": 0.8, "zorder": 4, "linestyle": ':'},
            "pedestrian":   {"width": 0.1 ,  "color": (0.5, 0.5, 0.5), "alpha": 0.8, "zorder": 4, "linestyle": '-'},
            "track":        {"width": 0.1, "color": (0.7, 0.6, 0.5), "alpha": 0.6, "zorder": 3, "linestyle": '-'},
            "construction": {"width": 0.1, "color": (0.8, 0.5, 0.1), "alpha": 0.5, "zorder": 2, "linestyle": ':'},
            "proposed":     {"width": 0.1, "color": (0.5, 0.5, 0.8), "alpha": 0.4, "zorder": 2, "linestyle": '--'},
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
    min_label_distance_km = 0.5
    placed_labels_coords = np.empty((0, 2))  # (lat, lon) in degrees

    if places_gdf is not None and not places_gdf.empty:
        settlement_types_map = {
            "city": {"settlementMarkerSize": settlementMarkerSize, "settlementsFontSize": settlementsFontSize,
                     "style": 'normal', "color": [0.1, 0.1, 0.1]},
            "town": {"settlementMarkerSize": settlementMarkerSize * 0.8,
                     "settlementsFontSize": settlementsFontSize * 0.8, "style": 'normal', "color": [0.1, 0.1, 0.1]},
            "village": {"settlementMarkerSize": settlementMarkerSize * 0.4,
                        "settlementsFontSize": settlementsFontSize * 0.5, "style": 'normal', "color": [0.1, 0.1, 0.1]},
            "suburb": {"settlementMarkerSize": 0,
                       "settlementsFontSize": settlementsFontSize * 0.3, "style": 'italic',
                       "color": [0.25, 0.25, 0.25]},
            "hamlet": {"settlementMarkerSize": 0,
                       "settlementsFontSize": settlementsFontSize * 0.3, "style": 'italic', "color": [0.25, 0.25, 0.25]}
        }

        settlement_places = places_gdf[places_gdf["type"].isin(settlement_types_map.keys())].copy()
        settlement_places = settlement_places[settlement_places['name'].notna()].copy()

        if not settlement_places.empty:
            # Sort by size to ensure larger places are plotted first, giving them priority for labels
            order = list(settlement_types_map.keys())
            settlement_places['type_order'] = settlement_places['type'].apply(lambda x: order.index(x))
            settlement_places = settlement_places.sort_values(by='type_order', ascending=True)

            candidate_labels = []
            for _, row in tqdm(settlement_places.iterrows(), total=len(settlement_places), desc="Filtering labels on proximity", dynamic_ncols=True, leave=False):
                x, y = row.geometry.x, row.geometry.y

                # Use cKDTree for efficient proximity checking
                if placed_labels_coords.size == 0 or haversine_vectorized(y, x, placed_labels_coords[:, 0], placed_labels_coords[:, 1]).min() >= min_label_distance_km:
                    candidate_labels.append(row)
                    placed_labels_coords = np.vstack([placed_labels_coords, [y, x]])

            # Plot markers and text for filtered labels
            for row in tqdm(candidate_labels, desc="Plotting labels", dynamic_ncols=True, leave=False):
                place_type = row['type']
                scales = settlement_types_map[place_type]
                marker_size = scales["settlementMarkerSize"] ** 2
                label_size = scales["settlementsFontSize"]
                text_style = scales['style']
                color = scales['color']

                x, y = row.geometry.x, row.geometry.y

                if marker_size > 0:
                    ax.scatter(x, y, s=marker_size, color=(0.1, 0.1, 0.1), zorder=6, linewidths=0)

                ax.text(
                    x, y, row["name"],
                    fontsize=label_size,
                    ha="center",
                    va="bottom",
                    rasterized=True,
                    style=text_style,
                    color=color,
                    path_effects=[patheffects.withStroke(linewidth=label_size*0.05, foreground=(0.9, 0.9, 0.9), capstyle="round")],
                    zorder=8
                )


    print("Plotting airports and their structures...")
    if airports_gdf is not None and not airports_gdf.empty:
        # Separate features by aeroway type for different plotting styles
        aerodrome_gdf = airports_gdf[airports_gdf['tags'].apply(lambda x: x.get('aeroway') == 'aerodrome')]
        runway_gdf = airports_gdf[airports_gdf['tags'].apply(lambda x: x.get('aeroway') == 'runway')]
        apron_gdf = airports_gdf[airports_gdf['tags'].apply(lambda x: x.get('aeroway') == 'apron')]
        taxiway_gdf = airports_gdf[airports_gdf['tags'].apply(lambda x: x.get('aeroway') == 'taxiway')]

        # Plot Aprons as filled gray areas (Polygons)
        if not apron_gdf.empty:
            apron_gdf.plot(
                ax=ax,
                facecolor=(0.9, 0.9, 0.9),  # Light gray
                edgecolor=(0.7, 0.7, 0.7),
                linewidth=0.1,
                alpha=0.5,
                zorder=5,
                rasterized=True
            )
            ax.collections[-1].set_rasterized(True)

        # Plot Runways as thick dark gray lines (LineStrings or Polygons)
        if not runway_gdf.empty:
            runway_gdf.plot(
                ax=ax,
                facecolor='none',
                edgecolor=(0.3, 0.3, 0.3),  # Dark gray
                linewidth=.4,
                alpha=0.5,
                zorder=6,
                rasterized=True
            )
            ax.collections[-1].set_rasterized(True)

        # Plot Taxiways as thinner dark gray dashed lines
        if not taxiway_gdf.empty:
            taxiway_gdf.plot(
                ax=ax,
                facecolor='none',
                edgecolor=(0.5, 0.5, 0.5),  # Gray
                linewidth=0.2,
                alpha=0.5,
                linestyle='--',
                zorder=6,
                rasterized=True
            )
            ax.collections[-1].set_rasterized(True)

        # Plot Aerodrome as a symbol at the centroid of the feature (Point or Polygon)
        if not aerodrome_gdf.empty:
            for _, row in aerodrome_gdf.iterrows():
                geom = row.geometry
                # Use centroid for Polygons, or use point directly
                if geom.geom_type == 'Point':
                    x, y = geom.x, geom.y
                else:
                    centroid = geom.centroid
                    x, y = centroid.x, centroid.y

                ax.text(x, y, "✈", fontsize=airportMarkerSize, ha='center', va='top',
                        color='darkblue', zorder=7, alpha=.8, rasterized=True)

                # Optionally label airports by name
                words = re.split(r"[ -]", row.get('tags', {}).get('name', ''))
                name = "\n".join(" ".join(words[i:i + 2]) for i in range(0, len(words), 2))
                if name:
                    ax.text(x, y, name,
                            fontsize=airportsFontSize,
                            ha='center', va='bottom',
                            color='darkblue', fontfamily='serif', style='italic',
                            zorder=8, alpha=.8, rasterized=True)

    # Plot mountain peaks as dark green triangles with village marker size
    print("Plotting mountain peaks (Haversine KDTree)...")
    min_peak_label_distance_km = 1.0
    placed_coords = []
    if mountain_peaks_gdf is not None and not mountain_peaks_gdf.empty:
        texts = []
        coords_deg = []

        # Collect coordinates and texts
        # Use vectorized operations to apply functions and filter data
        mountain_peaks_gdf['geom_to_use'] = mountain_peaks_gdf.apply(
            lambda row: row.geometry if row.geometry.geom_type == "Point" else row.geometry.representative_point(),
            axis=1
        )

        # Extract coordinates and tags in a vectorized manner
        x_coords = mountain_peaks_gdf['geom_to_use'].apply(lambda geom: geom.x)
        y_coords = mountain_peaks_gdf['geom_to_use'].apply(lambda geom: geom.y)
        names = mountain_peaks_gdf['tags'].apply(lambda tags: tags.get("name", ""))
        eles = mountain_peaks_gdf['tags'].apply(lambda tags: tags.get("ele"))

        # Filter out rows with no name
        valid_peaks = mountain_peaks_gdf[names != ""]
        x_coords_filtered = valid_peaks['geom_to_use'].apply(lambda geom: geom.x)
        y_coords_filtered = valid_peaks['geom_to_use'].apply(lambda geom: geom.y)
        names_filtered = valid_peaks['tags'].apply(lambda tags: tags.get("name", ""))
        eles_filtered = valid_peaks['tags'].apply(lambda tags: tags.get("ele"))

        # Use a list comprehension to build the texts list efficiently
        texts = [(x, y, f"{name}\n{ele} m" if ele else name)
                 for x, y, name, ele in zip(x_coords_filtered, y_coords_filtered, names_filtered, eles_filtered)]

        # Use vectorized plotting
        ax.scatter(x_coords, y_coords, s=mountainPeaksMarkerSize ** 2, c="darkgreen",
                   marker="^", zorder=6, linewidths=0)

        # Build coords_deg list with list comprehension
        coords_deg = [(y, x) for x, y in zip(x_coords_filtered, y_coords_filtered)]
        # Convert to radians for haversine
        coords_rad = np.radians(coords_deg)
        tree = cKDTree(coords_rad)
        # Radius in radians
        radius_rad = min_peak_label_distance_km / 6371.0
        # Place labels avoiding close neighbors
        for (x, y, label), (lat_deg, lon_deg) in tqdm(zip(texts, coords_deg),
                                                      total=len(texts),
                                                      desc="Mountain peak labels",
                                                      dynamic_ncols=True):
            lat_rad, lon_rad = np.radians([lat_deg, lon_deg])
            idxs = tree.query_ball_point([lat_rad, lon_rad], r=radius_rad)
            if all(tuple(coords_deg[i]) not in placed_coords for i in idxs):
                ax.text(x, y, label,
                        fontsize=mountainPeaksFontSize, ha="center", va="bottom",
                        color="darkgreen", fontfamily="serif", style="italic",
                        zorder=7, rasterized=True)
                placed_coords.append((lat_deg, lon_deg))

    # --- Plot country boundaries ---
    print("Plotting country boundaries...")
    if country_boundaries_gdf is not None and not country_boundaries_gdf.empty:
        country_boundaries_gdf.plot(  # red thick line
            ax=ax,
            facecolor='none',
            edgecolor=(0.7, 0.2, 0.2),
            linewidth=1,
            alpha=.5,
            linestyle='-',
            zorder=4
        )
        ax.collections[-1].set_rasterized(True)
        country_boundaries_gdf.plot(  # black dashed thin line
            ax=ax,
            facecolor='none',
            edgecolor=(0.1, 0.1, 0.1),
            linewidth=.25,
            alpha=.9,
            linestyle=(0, (3, 5, 1, 5)),  # dashdotted
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

    # rasterPath = r".\11_23.1_20.4_42.5_40.6\heightmap_z11_lon_20.2_23.2_lat_40.6_42.6.npz"  # NMK zoom 11
    # rasterPath = r".\12_23.1_20.3_42.4_40.7\heightmap_z12_lon_20.3_23.1_lat_40.6_42.5.npz"  # NMK zoom 12
    rasterPath = r".\13_23.1_20.4_42.5_40.6\heightmap_z13_lon_20.3_23.1_lat_40.6_42.5.npz"  # NMK zoom 13
    # rasterPath = r".\14_23.0_20.4_42.4_40.7\heightmap_z14_lon_20.4_23.0_lat_40.7_42.4.npz" # NMK zoom 14
    # rasterPath = r".\11_16.6_13.3_47.0_45.3\heightmap_z11_lon_13.2_16.7_lat_45.2_47.0.npz" # Slovenia
    # rasterPath = r".\11_23.2_13.3_46.9_40.7\heightmap_z11_lon_13.2_23.2_lat_40.6_47.0.npz" # ex YU
    # rasterPath = r".\9_22.1_21.8_41.6_41.2\heightmap_z9_lon_21.1_22.5_lat_41.0_42.0.npz" # tikvesko ezero debug

    ### hires settings
    ### We have to use a scaling trick in order to render small fonts (less than 1pt)
    # resolutionFactor, dpi = 5, int(640)
    # resolutionFactor, dpi = 4, int(800)
    # resolutionFactor, dpi = 3, int(1066)
    # resolutionFactor, dpi = 2, int(1600)
    resolutionFactor, dpi = 2.0, int(1500)  # good middle ground
    ### lowres settings
    # resolutionFactor, dpi = 2, int(640)

    print("Starting map generation process...")
    # mapzen tiles usually come in as Web Mercator, projcet to lat-lon rectangular projection
    elev_4326, meta_4326 = reproject_npz_to_epsg4326(rasterPath)

    map_data = elev_4326
    south = meta_4326['south']
    north = meta_4326['north']
    west = meta_4326['west']
    east = meta_4326['east']
    resolution = meta_4326['resolution_lon_deg']

    print("Raster and metadata loaded!")
    subsample = 1
    from skimage.transform import rescale
    map_s = rescale(map_data, 1/subsample, anti_aliasing=True)
    map_s[map_s <= -1] = - np.max(map_s) * 1  # we're not going to draw underwater structures, set water to -3% max height (so that 0asl is rendered green in terrain cmap)

    print(f"Map boundaries: North={north:.2f}, South={south:.2f}, West={west:.2f}, East={east:.2f}")

    places_gdf = load_or_fetch("places", rasterPath, south, north, west, east, get_places_from_osm)
    roads_gdf = load_or_fetch("roads", rasterPath, south, north, west, east, get_roads_from_osm)
    structures_gdf = load_or_fetch("structures", rasterPath, south, north, west, east, get_structures_from_osm)
    rivers_gdf =  load_or_fetch("rivers", rasterPath, south, north, west, east, get_rivers_from_osm)
    water_bodies_gdf = load_or_fetch("water_bodies", rasterPath, south, north, west, east, get_water_bodies_osm2geojson)
    mountain_peaks_gdf = load_or_fetch("mountain_peaks", rasterPath, south, north, west, east, get_mountain_peaks_osm2geojson)
    railroads_gdf = load_or_fetch("railroads", rasterPath, south, north, west, east, get_railroads_osm2geojson)
    airports_gdf = load_or_fetch("airports", rasterPath, south, north, west, east, get_airports_osm2geojson)
    country_boundaries_gdf = load_or_fetch("country_boundaries", rasterPath, south, north, west, east, get_country_boundaries_from_osm)



    # Apply color exaggeration
    exagerateTerrain = True
    fig, ax = plot_relief_with_features(places_gdf, roads_gdf, structures_gdf, rivers_gdf, water_bodies_gdf,
                                        mountain_peaks_gdf, railroads_gdf, airports_gdf, country_boundaries_gdf, map_s,
                                        south, west, north,
                                        east, dpi=dpi, resolutionFactor=resolutionFactor, resolution=resolution,
                                        exagerateTerrain=exagerateTerrain)

    output_filename = (
        f'baseMap_{subsample}_{resolutionFactor}_{dpi}dpi_{resolution}_ex={exagerateTerrain}'
        f'_E={format(east, ".3f").replace(".", ",")}'
        f'_W={format(west, ".3f").replace(".", ",")}'
        f'_N={format(north, ".3f").replace(".", ",")}'
        f'_S={format(south, ".3f").replace(".", ",")}.png'
    )
    print(f"Saving map to {output_filename}...")
    start_time = time.time()
    fig.savefig(sanitize_filename(output_filename), dpi=dpi, bbox_inches='tight', pad_inches=0)
    end_time = time.time()
    print(f"Map saved successfully in {end_time - start_time:.2f} seconds.")
    print("Process finished.")


if __name__ == "__main__":
    main()

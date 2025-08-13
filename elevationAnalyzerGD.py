import argparse, os, sys, numpy as np, math, matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import minimum_filter, generate_binary_structure
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection


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
def find_min_positions(map_s, xy, radius):
    def min_pos(py, px):
        y_min, y_max = max(py - radius, 0), min(py + radius + 1, map_s.shape[0])
        x_min, x_max = max(px - radius, 0), min(px + radius + 1, map_s.shape[1])
        patch = map_s[y_min:y_max, x_min:x_max]
        y_rel, x_rel = np.ogrid[y_min - py:y_max - py, x_min - px:x_max - px]
        mask = (x_rel**2 + y_rel**2) <= radius**2
        min_idx = np.argmin(np.where(mask, patch, np.inf))
        min_y, min_x = np.unravel_index(min_idx, patch.shape)
        return y_min + min_y, x_min + min_x
    return np.array([min_pos(py, px) for py, px in xy.astype(int)])

def follow_steepest_descent(map_s, peaks, steps=10, window_radius=2, min_gradient=0.1):
    positions = []
    min_vals = []
    min_positions = []

    h, w = map_s.shape

    for py, px in peaks.astype(int):
        path = [(py, px)]
        current_pos = (py, px)
        current_val = map_s[py, px]
        min_val = current_val
        min_pos = current_pos

        for i in range(steps):
            if i <= 1: # in the first two steps use a larger window radius
                window_radius_i = 3*window_radius
            else:
                window_radius_i = window_radius

            y, x = current_pos

            # 5x5 window bounds
            y_min = max(y - window_radius_i, 0)
            y_max = min(y + window_radius_i + 1, h)
            x_min = max(x - window_radius_i, 0)
            x_max = min(x + window_radius_i + 1, w)

            patch = map_s[y_min:y_max, x_min:x_max]
            rel_min_idx = np.argmin(patch)
            rel_dy, rel_dx = np.unravel_index(rel_min_idx, patch.shape)

            dy = (y_min + rel_dy) - y
            dx = (x_min + rel_dx) - x

            # Skip if no descent
            if (dy == 0 and dx == 0):
                break

            # Clip movement to 1-pixel step
            step_y = np.clip(dy, -1, 1)
            step_x = np.clip(dx, -1, 1)
            next_y = y + step_y
            next_x = x + step_x

            next_val = map_s[next_y, next_x]
            drop = current_val - next_val

            if np.abs(drop) < min_gradient:# or drop <= 0:
                break  # too flat or uphill

            current_pos = (next_y, next_x)
            current_val = next_val
            path.append(current_pos)

            if current_val < min_val:
                min_val = current_val
                min_pos = current_pos

        positions.append(path)
        min_vals.append(min_val)
        min_positions.append(min_pos)

    return positions, np.array(min_positions), np.array(min_vals)
def main():
    # mat = load_mat(r'C:\Users\mdimitri\OneDrive - UGent\Desktop\Terrain party\Macedonia\mk_corrected.mat')
    map = np.load('mk_corrected.npy').T
    # sea level is -28510.299
    # 2489m is 94022.3
    # normalize
    map -= -28510.299
    map /= ((94022.3+28510.299) / 2489)

    map_s = map[0::1, 0::1]
    lat = np.load('latitudes.npy')
    lon = np.load('longitudes.npy')
    distance_km = haversine_distance(lat[0], lon[0], lat[-1], lon[-1]) # the diagonal
    pixelSize = 1000 * distance_km / (map_s.shape[0]**2+map_s.shape[1]**2)**0.5 # in meters

    targetRadius = 1000 # in m
    targetRadiusPx = targetRadius / pixelSize

    xy  = peak_local_max(map_s, min_distance=int(targetRadiusPx))
    # xy_ = peak_local_max(np.max(map_s) - map_s, min_distance=int(targetRadiusPx))

    peaks = map_s[xy[:, 0].astype(np.int32), xy[:, 1].astype(np.int32)]

    flattnesLimit = 1  # 1% 1m in 100m
    descentRadius = 5000 # total distance in m
    descentLocalWindow = 350 # in m

    descentRadiusPx = descentRadius / pixelSize
    min_gradient = int(flattnesLimit/100 * pixelSize)
    window_radius = int(np.ceil(descentLocalWindow / pixelSize))
    paths, min_positions, min_vals = follow_steepest_descent(map_s, xy, steps=int(descentRadiusPx), window_radius=window_radius, min_gradient=min_gradient)
    diffs = peaks - min_vals
    sorted_indices = np.argsort(diffs)[::-1]

    im = plt.imshow(map_s, cmap=gamma_cmap, vmin=0)
    cbar = plt.colorbar(im)
    cbar.locator = MaxNLocator(nbins=30)  # request ~10 ticks
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=5)
    plt.pause(0.01)

    for i in np.arange(0, len(paths)):
        if diffs[sorted_indices[i]] < 200:
            continue
        path = paths[sorted_indices[i]]
        plt.scatter(path[0][1], path[0][0], s=.03, c='red')
        ys, xs = zip(*path)
        # plt.plot(xs, ys, linewidth=.1, color='black')

        # build segments
        segs = np.stack([np.column_stack([xs[:-1], ys[:-1]]),
                         np.column_stack([xs[1:], ys[1:]])], axis=1)
        # decreasing widths from 1 to 0.1
        widths = np.linspace(.15, 0.03, len(segs))
        lc = LineCollection(segs, colors='black', linewidths=widths)
        plt.gca().add_collection(lc)

        plt.text(path[0][1], path[0][0], f"{diffs[sorted_indices[i]]:.0f}", color='black',
                 fontsize=.3)
        # plt.pause(0.01)


    plt.pause(0.01)
    plt.axis('off')
    plt.savefig('prominences.png', dpi=3000)
    return

if __name__ == "__main__":
    main()
import argparse, os, sys, numpy as np, math, matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import minimum_filter, generate_binary_structure
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
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
def main():
    # mat = load_mat(r'C:\Users\mdimitri\OneDrive - UGent\Desktop\Terrain party\Macedonia\mk_corrected.mat')
    map = np.load('mk_corrected.npy').T
    # sea level is -28510.299
    # 2489m is 94022.3
    # normalize
    map -= -28510.299
    map /= ((94022.3+28510.299) / 2489)

    map_s = map[0::5, 0::5]
    lat = np.load('latitudes.npy')
    lon = np.load('longitudes.npy')
    distance_km = haversine_distance(lat[0], lon[0], lat[-1], lon[-1]) # the diagonal
    pixelSize = 1000 * distance_km / (map_s.shape[0]**2+map_s.shape[1]**2)**0.5 # in meters

    targetRadius = 2000 # in m
    targetRadiusPx = targetRadius / pixelSize

    xy  = peak_local_max(map_s, min_distance=int(targetRadiusPx))
    # xy_ = peak_local_max(np.max(map_s) - map_s, min_distance=int(targetRadiusPx))

    peaks = map_s[xy[:, 0].astype(np.uint16), xy[:, 1].astype(np.uint16)]
    # pits  = map_s[xy_[:, 0].astype(np.uint16), xy_[:, 1].astype(np.uint16)]
    # todo for each peak follow the steepest gradient down for 5km and compute the drop
    # find the lowest value within the search radius
    targetRadius = 5000 # in m
    targetRadiusPx = targetRadius / pixelSize
    y, x = np.ogrid[-int(targetRadiusPx):int(targetRadiusPx) + 1, -int(targetRadiusPx):int(targetRadiusPx) + 1]
    footprint = (x ** 2 + y ** 2) <= int(targetRadiusPx) ** 2
    # Compute the minimum filter over the whole array with this footprint
    min_filtered = minimum_filter(map_s, footprint=footprint, mode='mirror')
    # Extract minimums at the peak locations
    minimums = min_filtered[xy[:, 0].astype(np.uint16), xy[:, 1].astype(np.uint16)]
    diffs = peaks - minimums
    sorted_indices = np.argsort(diffs)[::-1]

    min_positions = find_min_positions(map_s, xy, int(targetRadiusPx))


    # tree = cKDTree(xy_)
    # distances, indices = tree.query(xy, k=1)

    # diffs = peaks - pits[indices]
    # sorted_indices = np.argsort(diffs)[::-1]

    im = plt.imshow(map_s, cmap=gamma_cmap, vmin=0)
    cbar = plt.colorbar(im)
    cbar.locator = MaxNLocator(nbins=30)  # request ~10 ticks
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=5)
    # plot the top 10 points
    for i in np.arange(0, len(diffs)):
        if diffs[sorted_indices[i]] < 100:
            continue
        # plt.scatter(min_positions[sorted_indices[i], 1], min_positions[sorted_indices[i], 0], s=5, c='black')
        plt.plot([xy[sorted_indices[i], 1], min_positions[sorted_indices[i], 1]],
                 [xy[sorted_indices[i], 0], min_positions[sorted_indices[i], 0]], c='black', linewidth=.1)
        plt.scatter(xy[sorted_indices[i], 1], xy[sorted_indices[i], 0], s=.05, c='red')
        plt.text(xy[sorted_indices[i], 1], xy[sorted_indices[i], 0], f"{diffs[sorted_indices[i]]:.0f}", color='black',
                 fontsize=.6)
        # plt.scatter(xy_[indices[sorted_indices[i]], 1], xy_[indices[sorted_indices[i]], 0], s=5, c='cyan')
        # plt.plot([xy[sorted_indices[i], 1], xy_[indices[sorted_indices[i]], 1]], [xy[sorted_indices[i], 0], xy_[indices[sorted_indices[i]], 0]], c='black',linewidth=1)
    # plt.scatter(xy[:, 1], xy[:, 0], s=1, c=peaks, cmap='hot')
    # plt.scatter(xy_[:, 1], xy_[:, 0], s=1, c='cyan')
    #     print(diffs[sorted_indices[i]])
    #     plt.pause(0.01)
    plt.pause(0.01)
    plt.axis('off')
    plt.savefig('prominences.png', dpi=2000)
    return

if __name__ == "__main__":
    main()
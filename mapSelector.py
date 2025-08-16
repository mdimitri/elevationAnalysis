import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import geopandas as gpd
import geodatasets
from matplotlib.patches import Rectangle

class MapSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("World Map Rectangle Selector (Geopandas)")
        self.root.state('zoomed')

        self.world_gdf = None
        self.rect_patch = None
        self.start_lon_lat = None
        self.end_lon_lat = None

        self.fig, self.ax = plt.subplots(figsize=(8, 6), layout='constrained')
        self.ax.set_aspect('equal')
        self.ax.set_facecolor("aliceblue")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget.draw()
        self.canvas = self.canvas_widget.get_tk_widget()
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas_widget, self.root, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.coords_label = tk.Label(root, text="Selected Area: (N/A)", font=("Arial", 12))
        self.coords_label.pack(pady=10)

        self.load_and_plot_map()

        self.canvas_widget.mpl_connect("button_press_event", self.on_button_press)
        self.canvas_widget.mpl_connect("motion_notify_event", self.on_mouse_drag)
        self.canvas_widget.mpl_connect("button_release_event", self.on_button_release)

        self.root.bind("<Configure>", self.on_window_resize)

    def load_and_plot_map(self):
        try:
            print("Attempting to load 'naturalearth.countries.50m'...")
            self.world_gdf = gpd.read_file(r".\worldMap\ne_50m_admin_0_countries.zip")
            print("Successfully loaded local world map. (downloaded from: https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/50m/cultural/ne_50m_admin_0_countries.zip")
        except ValueError:
            print("'naturalearth.countries.50m' not found. Falling back to 'naturalearth.land'.")
            self.world_gdf = gpd.read_file(geodatasets.get_path('naturalearth.land'))
            print("Successfully loaded 'naturalearth.land'.")

        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_facecolor("aliceblue")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.world_gdf.plot(ax=self.ax, color='lightgreen', edgecolor='black', linewidth=0.5)

        self.ax.set_xlim(-180, 180)
        self.ax.set_ylim(-90, 90)

        self.canvas_widget.draw_idle()

    def on_button_press(self, event):
        if event.inaxes == self.ax:
            self.start_lon_lat = (event.xdata, event.ydata)
            if self.rect_patch:
                self.rect_patch.remove()
                self.rect_patch = None
            self.coords_label.config(text="Selected Area: (N/A)")
            self.canvas_widget.draw_idle()

    def on_mouse_drag(self, event):
        if event.inaxes == self.ax and self.start_lon_lat:
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
        if event.inaxes == self.ax and self.start_lon_lat:
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
            print(
                f"Rectangle selected: Longitude ({min_lon:.2f}, {max_lon:.2f}), Latitude ({min_lat:.2f}, {max_lat:.2f})")

            self.start_lon_lat = None
            self.canvas_widget.draw_idle()

    def on_window_resize(self, event):
        self.fig.tight_layout()
        self.canvas_widget.draw_idle()

if __name__ == "__main__":
    root = tk.Tk()
    app = MapSelectorApp(root)
    root.mainloop()

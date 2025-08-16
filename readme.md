# Easy Heightmap Downloader

This Python tool helps you get **elevation data** for any area you choose on Earth. It takes this data and turns it into two easy-to-use files: a **data file (.npz)** and a **picture (.jpg)**. It uses a free online source called the Terrarium project for the elevation info.

## ✨ What It Does

* **Finds Locations:** Converts regular map coordinates (latitude and longitude) into a special map grid (ZXY tiles).

* **Gets Data Fast:** Downloads many small elevation map pieces (tiles) at the same time to save you time.

* **Decodes Height:** Turns the colors from the downloaded tiles into actual heights in meters.

* **Puts It Together:** Stitches all the small map pieces into one big elevation map.

* **Saves Files:** Creates a data file for numbers (`.npz`) and a picture for viewing (`.jpg`).

* **Shows Progress:** Lets you see how much of the work is done while it's running.

## 🛠️ How It Works

This script works by:

1. **Using a Map Grid:** It uses a common map system where the world is split into squares called "tiles."

2. **Downloading Tiles:** It gets these special tiles from the Terrarium project. These tiles have hidden height information in their colors.

3. **Reading Heights:** It figures out the height for each spot by reading the red, green, and blue colors in the tile.

4. **Working Together:** It uses your computer's power to download and process many tiles at once, making it faster.

5. **Making Your Map:** Once all the tiles are ready, it combines them into one large height map. This map is then saved as a raw data file and a simple image.

## 🚀 Get Started

First, make sure you have Python (version 3.7 or newer) on your computer. Then, open your command line or terminal and install these needed tools:

pip install requests Pillow numpy tqdm


## 📝 How to Use

Just run the `main.py` file. The script already has a small area (part of North Macedonia) set up as an example.

1. **Get the Script:** Download or copy this script to your computer.

2. **Change the Area (Optional):** Open the script file (like `your_script_name.py`). Find the `main()` section. Here, you can change the numbers for `west`, `south`, `east`, and `north` to pick a different area you want to map. You can also change `zoom` for more detail (higher numbers mean more detail but take longer).

def main():
# Example area (change these numbers for your location!)
west = 20.4529023 - 0.05
south = 40.852478 - 0.05
east = 23.034051 + 0.05
north = 42.3739044 + 0.05

   # It will create a new folder for your map data, named by today's date and time.
   output_dir = __import__("datetime").datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
   # Runs the main process with your settings
   fetch_and_process_heightmap(west, south, east, north, zoom=12, output_dir=output_dir, max_workers=32)


* `west`, `south`, `east`, `north`: These are the edges of your map area (longitude and latitude).

* `zoom`: How much detail you want (e.g., `12` is less detailed, `14` is more).

* `max_workers`: How many parts of your computer work on downloading at the same time. You can change this based on your internet speed.

3. **Run It!:** Open your terminal or command prompt, go to where you saved the script, and type:

python your_script_name.py


A new folder will appear (named with the current date and time), and your elevation data will be saved inside it. You'll see updates as it works!

## 📦 What You Get

After it finishes, you'll find a new folder (for example, `2023-10-27_15-30-45`). Inside, there will be two files:

1. **`heightmap_...npz` (Data File):**

* This file holds the actual height numbers (in meters).

* You can load it into Python using `np.load('path/to/file.npz')`.

* It contains the `elevations` (the height data itself) and other details like `north`, `south`, `west`, `east` (your map's edges), and `resolution_lon_deg`, `resolution_lat_deg` (how detailed each pixel is).

2. **`heightmap_...jpg` (Picture File):**

* This is a grayscale image of your heightmap. Darker areas are lower, lighter areas are higher. It's just for quickly looking at the map.

## ⚠️ Important Notes

* **Internet Needed:** You must have an active internet connection for this to work.

* **Retries:** If a download fails, the script will try again a few times.

* **Big Downloads:** Very large areas or very high detail levels can take a long time and use a lot of space on your computer.

* **Data Quality:** The elevation data comes from various sources, so it might not be perfect everywhere.
# Sheep counter

Counts white sheep in a top-down drone or aerial photo. It prints the count and saves a copy of the photo with a dot on every sheep it found, so you can check the result by eye.

## Install

Requires Python 3.8 or newer.

```
pip install -r requirements.txt
```

On Windows, use `py -m pip install -r requirements.txt`.

## Usage

```
python count_sheep.py sheep.jpg
```

This prints:

- **Sheep counted**: the confident detections (red dots)
- **Including borderline**: confident + uncertain detections (red + orange dots)
- **Sensitivity**: the count at other confidence cut-offs, so you can see how stable the number is

It also saves `sheep_counted.jpg` next to the photo.

### Other photos: set the sheep size

The detector needs to know roughly how big one sheep is in pixels. The defaults (22 × 9 px) fit the photo the script was developed on. For another photo, zoom in, measure a typical sheep's body length and width in pixels, and pass them in:

```
python count_sheep.py other.jpg --sheep-length 40 --sheep-width 16
```

Always open the annotated image afterwards and check the dots sit on the sheep.

## Checking accuracy

The automatic count is an estimate. To measure how accurate it is on your photo:

1. Make random sample tiles to count by hand:
   ```
   python count_sheep.py sheep.jpg --tiles 20
   ```
   This creates a `sheep_tiles` folder with a clean and a marked image of each tile, plus `tiles.csv`.
2. Count the sheep whose body centre is inside the yellow box, using the **clean** images so the dots don't bias you. Type the numbers into the `manual_count` column of `tiles.csv` (Excel works).
3. Get an adjusted total with a 95% confidence interval:
   ```
   python count_sheep.py --evaluate sheep_tiles/tiles.csv
   ```

The adjustment is a ratio estimate: hand count ÷ script count across the sampled tiles, applied to the script's total. Count 20 or more tiles for a usable interval.

## All options

| Option | Default | Meaning |
|---|---|---|
| `--sheep-length` | 22 | Typical sheep body length in pixels |
| `--sheep-width` | 9 | Typical sheep body width in pixels |
| `--threshold` | 0.18 | Confidence cut-off for the main count (lower = more sheep) |
| `--borderline` | 0.15 | Lower cut-off for the "including borderline" count |
| `--out FILE` | `<photo>_counted.jpg` | Where to save the annotated image |
| `--csv FILE` | – | Save every detection (x, y, score, angle) to a CSV file |
| `--tiles N` | 0 | Make N random tiles for hand counting |
| `--tile-size` | 4.5 × sheep length | Tile size in pixels |
| `--seed` | 1 | Random seed for tile selection |
| `--evaluate CSV` | – | Compute adjusted total from a hand-counted `tiles.csv` |

## How it works

1. **Whiteness map.** Sheep are removed from the image with a morphological opening, leaving the background. Each pixel is then scored by how far it is from its local background towards pure white. This makes sheep stand out equally on light ground and dark bushes.
2. **Shape matching.** A sheep-shaped template (a bright ellipse with darker surroundings) is matched at 12 orientations. Each pixel keeps its best score and angle.
3. **Picking sheep.** Local peaks are taken strongest first. A peak is dropped if it falls inside the footprint of a sheep already accepted. The footprint follows the sheep's orientation, so two sheep side by side are both kept, but one long sheep is not counted twice.

Only `numpy` and `opencv-python` are needed. File paths with non-ASCII characters (æ, ø, å) work on Windows.

## Limitations

- Only light-coloured sheep are detected. Dark sheep, dogs and people are not counted.
- Sheep pressed tightly together can merge into one detection, so very dense patches are slightly undercounted. Bright rocks or bushes are occasionally detected as sheep.
- Only what is inside the photo is counted.
- More pixels per sheep gives a better count. An original drone file usually beats a resized or screenshotted copy.

Photos, CSV files and tile folders are excluded by `.gitignore`, so client data isn't committed by accident.

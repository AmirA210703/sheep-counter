#!/usr/bin/env python3
"""
count_sheep.py - count white sheep in a top-down (drone / aerial) photo.

INSTALL (once)
    pip install opencv-python numpy

BASIC USE
    python count_sheep.py sheep.jpg

    Prints the count and saves sheep_counted.jpg next to the photo, with a dot
    on every detected sheep (red = confident, orange = borderline).

    The photo can have any file name. If you leave the name out, the script
    counts the only photo in the folder. On Windows you can also double-click
    the script.

OTHER PHOTOS
    The detector needs to know roughly how big one sheep is in pixels. The
    defaults (22 x 9 px) fit the original photo. For another photo, zoom in,
    measure a typical sheep's body length and width in pixels (Paint shows the
    pixel position at the bottom of the window), and pass them in:

        python count_sheep.py other.jpg --sheep-length 40 --sheep-width 16

    Open the annotated image afterwards and check the dots look right.

CHECKING ACCURACY (for audit use)
    1) Make random sample tiles to count by hand:
           python count_sheep.py sheep.jpg --tiles 20
       This creates a folder with a "clean" and a "marked" image of each tile
       plus tiles.csv. Count by hand the sheep whose body centre is inside the
       yellow box, using the clean image so the dots don't bias you, and
       type the numbers into the manual_count column of tiles.csv.
    2) Get an adjusted total with a 95% confidence interval:
           python count_sheep.py --evaluate sheep_tiles/tiles.csv

OPTIONS
    --threshold 0.18    confidence cut-off for the main count (lower = more)
    --borderline 0.15   lower cut-off for the "including borderline" count
    --csv FILE          also save every detection (x, y, score, angle)
    --out FILE          where to save the annotated image

LIMITATIONS
    * Only light-coloured sheep are detected. Dark sheep, dogs and people are
      not counted.
    * Sheep standing on top of or tightly against each other can merge into
      one detection, so very dense patches are slightly undercounted.
      Bright rocks or bushes can occasionally be detected as sheep.
    * Only what is inside the photo is counted.
"""

import argparse
import csv
import math
import os
import random
import sys

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is missing. Install it with:  pip install opencv-python numpy")


# --------------------------------------------------------------------------
# File helpers (work with non-ASCII paths such as æ/ø/å on Windows)
# --------------------------------------------------------------------------
def read_image(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as e:
        sys.exit(f"Could not open {path}: {e}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read {path} as an image.")
    return img


def write_image(path, img):
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, 92] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        sys.exit(f"Could not save {path}")
    buf.tofile(path)


def find_photo():
    """Used when no photo is given: take the only photo in the current folder
    (or the script's folder), ignoring images this script produced."""
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
    folders = dict.fromkeys([os.getcwd(), os.path.dirname(os.path.abspath(__file__))])
    for folder in folders:
        photos = sorted(f for f in os.listdir(folder)
                        if f.lower().endswith(exts) and "_counted" not in f.lower())
        if len(photos) == 1:
            return os.path.join(folder, photos[0])
        if len(photos) > 1:
            sys.exit("Found several photos: " + ", ".join(photos) +
                     "\nSay which one to count, e.g.  python count_sheep.py " + photos[0])
    sys.exit("No photo found. Put the photo in this folder, or run:  python count_sheep.py path/to/photo.jpg")


def odd(n):
    n = int(round(n))
    return n if n % 2 else n + 1


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def whiteness_map(gray, L):
    """How far each pixel is from its local background towards pure white.
    Works on both light ground and dark bushes."""
    k = odd(1.6 * L)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)  # image with sheep removed
    bg = cv2.GaussianBlur(bg, (0, 0), L / 4.4)
    return np.clip((gray - bg) / np.maximum(255 - bg, 30), -1, 1.5)


def oriented_response(r, L, W, step=15):
    """Match an elongated sheep-shaped template (bright ellipse with darker
    surroundings) at 12 orientations; keep the best score and angle."""
    a_long = max(2, int(0.43 * L))
    a_short = max(1, int(0.39 * W))
    ring = max(2, int(round(0.14 * L)))
    size = 2 * (a_long + ring + 4) + 1
    c = size // 2
    best = np.full(r.shape, -9, np.float32)
    best_ang = np.zeros(r.shape, np.int16)
    for ang in range(0, 180, step):
        inner = np.zeros((size, size), np.uint8)
        outer = np.zeros((size, size), np.uint8)
        cv2.ellipse(inner, (c, c), (a_short, a_long), ang, 0, 360, 1, -1)
        cv2.ellipse(outer, (c, c), (a_short + ring, a_long + ring), ang, 0, 360, 1, -1)
        inner = inner > 0
        surround = (outer > 0) & ~inner
        k = np.zeros((size, size), np.float32)
        k[inner] = 1.0 / inner.sum()
        k[surround] = -1.0 / surround.sum()
        resp = cv2.filter2D(r, -1, k)
        better = resp > best
        best[better] = resp[better]
        best_ang[better] = ang
    return best, best_ang


def pick_sheep(R, A, min_score, L, W):
    """Take local peaks, strongest first, and drop any peak that falls inside
    the footprint of a sheep already accepted (orientation-aware, so two sheep
    side by side are both kept, but one long sheep is not counted twice)."""
    peak = cv2.dilate(R, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero((R == peak) & (R > min_score))
    scores = R[ys, xs]
    order = np.argsort(-scores, kind="stable")
    along_r, across_r = round(0.6 * L), 0.61 * W
    cell = int(math.ceil(along_r)) + 3
    trig = {a: (math.sin(math.radians(a)), math.cos(math.radians(a))) for a in range(0, 180)}
    grid, kept = {}, []
    for i in order:
        y, x, s, a = int(ys[i]), int(xs[i]), float(scores[i]), int(A[ys[i], xs[i]])
        cy, cx = y // cell, x // cell
        clash = False
        for gy in (cy - 1, cy, cy + 1):
            for gx in (cx - 1, cx, cx + 1):
                for (ky, kx, ka) in grid.get((gy, gx), ()):
                    dx, dy = x - kx, y - ky
                    for ang in (ka, a):
                        sn, cs = trig[ang]
                        along = -dx * sn + dy * cs
                        across = dx * cs + dy * sn
                        if (along / along_r) ** 2 + (across / across_r) ** 2 < 1:
                            clash = True
                            break
                    if clash:
                        break
                if clash:
                    break
            if clash:
                break
        if not clash:
            grid.setdefault((cy, cx), []).append((y, x, a))
            kept.append((x, y, s, a))
    return np.array(kept, dtype=np.float64).reshape(-1, 4)  # x, y, score, angle


def detect(img, L, W, min_score):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    r = whiteness_map(gray, L)
    R, A = oriented_response(r, L, W)
    return pick_sheep(R, A, min_score, L, W)


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
def annotate(img, det, thr, low, L):
    out = img.copy()
    rad = max(2, round(L / 7))
    for x, y, s, _ in det:
        if s >= thr:
            cv2.circle(out, (int(x), int(y)), rad, (0, 0, 255), -1)
        elif s >= low:
            cv2.circle(out, (int(x), int(y)), rad, (0, 165, 255), -1)
    n_hi = int((det[:, 2] >= thr).sum())
    n_lo = int((det[:, 2] >= low).sum())
    text = f"Sheep: {n_hi:,} (red)   incl. borderline: {n_lo:,} (red + orange)"
    h, w = out.shape[:2]
    scale = max(0.6, w / 1500)
    th = max(1, int(scale * 2))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    y0 = h - int(15 * scale)
    cv2.rectangle(out, (10, y0 - tht - int(12 * scale)), (20 + tw, y0 + int(10 * scale)), (0, 0, 0), -1)
    cv2.putText(out, text, (15, y0), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)
    return out


def make_tiles(img, det, thr, n, size, seed, outdir, L, image_path):
    H, Wd = img.shape[:2]
    pts = det[det[:, 2] >= thr]
    total = len(pts)
    cells = []
    for y in range(0, H - size + 1, size):
        for x in range(0, Wd - size + 1, size):
            inside = (pts[:, 0] >= x) & (pts[:, 0] < x + size) & (pts[:, 1] >= y) & (pts[:, 1] < y + size)
            c = int(inside.sum())
            if c > 0:
                cells.append((x, y, c))
    if not cells:
        sys.exit("No sheep detected, so no tiles to sample.")
    rng = random.Random(seed)
    chosen = rng.sample(cells, min(n, len(cells)))
    os.makedirs(outdir, exist_ok=True)
    margin = int(L)
    zoom = max(2, round(700 / (size + 2 * margin)))
    rad = max(3, round(zoom * L / 8))
    rows = []
    for i, (x, y, c) in enumerate(chosen, 1):
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(Wd, x + size + margin), min(H, y + size + margin)
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        box = ((x - x0) * zoom, (y - y0) * zoom, (x - x0 + size) * zoom, (y - y0 + size) * zoom)
        clean = crop.copy()
        cv2.rectangle(clean, box[:2], box[2:], (0, 255, 255), 2)
        marked = clean.copy()
        for px, py, s, _ in pts:
            if x0 <= px < x1 and y0 <= py < y1:
                cv2.circle(marked, (int((px - x0) * zoom + zoom / 2), int((py - y0) * zoom + zoom / 2)),
                           rad, (0, 0, 255), -1)
        name = f"tile_{i:02d}"
        write_image(os.path.join(outdir, name + "_clean.png"), clean)
        write_image(os.path.join(outdir, name + "_marked.png"), marked)
        rows.append({"tile": name, "x": x, "y": y, "size": size, "machine_count": c,
                     "manual_count": "", "total_machine_count": total,
                     "threshold": thr, "image": os.path.basename(image_path)})
    csv_path = os.path.join(outdir, "tiles.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("sep=,\n")  # makes Excel split the columns correctly in any language setting
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return csv_path, len(chosen), len(cells)


T_TABLE = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
           10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 25: 2.06, 30: 2.04, 60: 2.00}


def t_value(df):
    keys = [k for k in sorted(T_TABLE) if k <= df]
    return T_TABLE[keys[-1]] if df < 120 else 1.96


def evaluate(csv_path):
    # utf-8-sig handles the BOM Excel adds when it saves a CSV
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    if lines and lines[0].lower().startswith("sep="):
        lines = lines[1:]
    header = lines[0] if lines else ""
    delim = ";" if header.count(";") > header.count(",") else ","  # Danish Excel saves with ;
    rows = list(csv.DictReader(lines, delimiter=delim))
    filled = [r for r in rows if str(r.get("manual_count", "")).strip() != ""]
    if len(filled) < 2:
        sys.exit("Fill in manual_count for at least 2 tiles (20 or more is recommended).")
    m = np.array([float(r["machine_count"]) for r in filled])
    h = np.array([float(str(r["manual_count"]).replace(",", ".")) for r in filled])
    total = float(rows[0]["total_machine_count"])
    n = len(filled)
    ratio = h.sum() / m.sum()
    resid = h - ratio * m
    se = math.sqrt(resid.var(ddof=1) / n) / m.mean()
    t = t_value(n - 1)
    est, lo, hi = ratio * total, (ratio - t * se) * total, (ratio + t * se) * total
    print(f"Tiles hand-counted:        {n}")
    print(f"Sheep in those tiles:      hand {int(h.sum())}  vs  script {int(m.sum())}")
    print(f"Hand / script ratio:       {ratio:.3f}")
    print(f"Script total:              {int(total):,}")
    print(f"Adjusted estimate:         {est:,.0f}")
    print(f"95% confidence interval:   {lo:,.0f} - {hi:,.0f}")
    if n < 20:
        print("Note: fewer than 20 tiles, so the interval is rough. Count more tiles to narrow it.")
    print("Note: tiles are sampled only where the script found at least one sheep, "
          "so sheep in completely missed areas are not covered by this check.")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Count white sheep in a top-down photo.",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="See the top of this file for full instructions.")
    p.add_argument("image", nargs="?", help="photo to count (default: the only photo in this folder)")
    p.add_argument("--sheep-length", type=float, default=22, help="typical sheep length in pixels (default 22)")
    p.add_argument("--sheep-width", type=float, default=9, help="typical sheep width in pixels (default 9)")
    p.add_argument("--threshold", type=float, default=0.18, help="confidence cut-off (default 0.18)")
    p.add_argument("--borderline", type=float, default=0.15, help="lower cut-off for borderline (default 0.15)")
    p.add_argument("--out", help="annotated image path (default: <photo>_counted.jpg)")
    p.add_argument("--csv", help="save all detections to this CSV file")
    p.add_argument("--tiles", type=int, default=0, help="number of random tiles to make for hand counting")
    p.add_argument("--tile-size", type=int, help="tile size in pixels (default: 4.5 x sheep length)")
    p.add_argument("--seed", type=int, default=1, help="random seed for tile selection (default 1)")
    p.add_argument("--evaluate", metavar="TILES_CSV", help="compute adjusted total from hand-counted tiles.csv")
    a = p.parse_args()

    if a.evaluate:
        evaluate(a.evaluate)
        return
    if not a.image:
        a.image = find_photo()
    if a.borderline > a.threshold:
        p.error("--borderline must be lower than --threshold")

    L, W = a.sheep_length, a.sheep_width
    img = read_image(a.image)
    h, w = img.shape[:2]
    print(f"Image: {os.path.basename(a.image)} ({w} x {h} px), sheep size assumed {L:g} x {W:g} px")
    print("Detecting sheep...", flush=True)

    sens = [0.12, 0.15, 0.18, 0.21, 0.24]
    det = detect(img, L, W, min(sens + [a.threshold, a.borderline]) - 1e-6)

    n_hi = int((det[:, 2] >= a.threshold).sum())
    n_lo = int((det[:, 2] >= a.borderline).sum())
    print()
    print(f"  Sheep counted (threshold {a.threshold:g}):            {n_hi:,}")
    print(f"  Including borderline (threshold {a.borderline:g}):    {n_lo:,}")
    print()
    print("  Sensitivity (count at other thresholds):")
    for t in sens:
        print(f"    {t:.2f}: {int((det[:, 2] >= t).sum()):,}")

    stem = os.path.splitext(a.image)[0]
    out = a.out or stem + "_counted.jpg"
    write_image(out, annotate(img, det, a.threshold, a.borderline, L))
    print(f"\nAnnotated image saved: {out}")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["x", "y", "score", "angle_deg", "counted"])
            for x, y, s, ang in det:
                if s >= a.borderline:
                    wr.writerow([int(x), int(y), round(s, 4), int(ang),
                                 "yes" if s >= a.threshold else "borderline"])
        print(f"Detections saved:      {a.csv}")

    if a.tiles > 0:
        size = a.tile_size or int(round(4.5 * L))
        outdir = stem + "_tiles"
        csv_path, k, avail = make_tiles(img, det, a.threshold, a.tiles, size, a.seed, outdir, L, a.image)
        print(f"Sample tiles saved:    {outdir}  ({k} of {avail} tiles that contain sheep)")
        print(f"  Fill in manual_count in {csv_path}, then run:")
        print(f"  python count_sheep.py --evaluate \"{csv_path}\"")


if __name__ == "__main__":
    if os.name == "nt" and len(sys.argv) == 1:
        # Started by double-click on Windows: keep the window open to show the result
        try:
            main()
        except SystemExit as e:
            if isinstance(e.code, str):
                print(e.code)
        except Exception as e:
            print(f"Error: {e}")
        input("\nPress Enter to close...")
    else:
        main()

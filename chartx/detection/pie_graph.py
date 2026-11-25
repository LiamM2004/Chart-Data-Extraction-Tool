"""Pie graph detection. Expects a PREPROCESSED image: deskewed, white board, no glare/shadow.

Finds the sector dividers with the ReVision unroll (Savva et al. 2011) and reads the labels
and percentages inside every sector with PaddleOCR. The steps:

0. coloured sectors -> line drawing: a pie with coloured fills has no drawn dividers, so every
   fill gets one label by its hue, the fill is painted white and the borders between labels black
1. text removal: PaddleOCR boxes -> erase the ink inside the boxes, but keep the long straight
   lines and the pie rim. Result: clean ink mask
2. Hough circle (centre + radius R) on the clean image
3. ink centre: the point where the dividers meet
4. unroll: cv2.warpPolar turns the circle around the ink centre into a rectangle.
   Every column is one angle, every row is one distance from the centre
5. histogram: for every angle, the share of ink pixels between BAND_IN and BAND_OUT (share of R).
   A divider is ink along the whole band -> a high peak
6. every run of angles above PEAK = one divider, at the middle of the run
7. sector size = angle between two dividers / 360
8. the same OCR boxes as in step 1 are used for the labels
9. one text per sector: every box with its centre inside the circle goes to the sector it lies in.
   The boxes of a sector are read like a page (lines top to bottom, each line left to right)
10. the percentage is read from the sector text. The OCR values are used only if every sector
   has exactly one and they add up to 100 (+- SUM_TOL)

All angles are in degrees, clockwise from 3 o'clock (the same as cv2.warpPolar).

PaddleOCR is only imported when a pie graph is detected: loading it is slow.

Preprocess a photo (chartx.main.preprocess) and print the sectors of its pie graph:
    python -m chartx.detection.pie_graph photo.jpg
"""
import functools
import math
import re
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

WORK_SIZE = 1000          # longest image side after resizing: all px settings below are tuned for it
INK_C = 15                # pixel must be this much darker than its neighbourhood to count as ink
# text removal
BOX_PAD = 4               # grow each box a little, so the ends of letters are inside
LINE_BAND = 13            # width in px of the protected band around each line: drawn lines wiggle
LINE_EXTEND = 20          # make each found line this many px longer at both ends (closes gaps at bends)
TEXT_LINE_MIN = 0.1       # a straight line at least this long (share of the shorter image side) is protected
RIM_BAND = 25             # width in px of the protected band around the Hough circle
EDGE_PAD = 5              # also erase this many px around each text stroke (ends of letters in the bands)
# ink centre
INSIDE = 0.9              # only look for lines inside this share of R (keeps the rim out)
DIV_LINE_MIN = 0.4        # a divider segment must be at least this long (share of R)
LINE_GAP = 0.05           # small breaks in a drawn line that are still joined (share of R)
MAX_END_DIST = 0.3        # one end of a divider must be this close to the Hough centre (share of R)
MAX_END_DIST_2 = 0.15     # second round: one end must be this close to the first estimate (share of R)
# unroll and histogram
ANGLES = 720              # columns of the unrolled image: 0.5 degree each
BAND_IN = 0.2             # measure ink from this share of R ...
BAND_OUT = 0.9            # ... to this share of R
PEAK = 0.4                # a divider: at least 40 % of the band is ink (tested 30-60 %, 40 % is best)
# labels
SUM_TOL = 1.0             # the OCR values must add up to 100 +- this (rounding, e.g. 25+22+20+17+17 = 101)
PERCENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
# coloured sectors -> line drawing
CHROMA_MIN = 15           # "coloured": max - min of B, G, R (averaged over FLAT_BLUR px) is above this.
                          # IMG_1294 (cleaned): pale fills 25-34, black text <= 9, board 0. HSV saturation does
                          # not work: it is high for dark pixels, so black text looked coloured
FLAT_BLUR = 15            # px: median blur that turns a hatched fill into one flat colour
FILL_MIN = 0.01           # a coloured area must be at least this share of the image. The fills of a pie touch,
                          # so they are one big area (IMG_1294: 5.8 %); colour noise in scans (pie_11, pie_19) <= 0.22 %
HUE_BIN = 30              # fills are told apart by hue in bins of this size (hue 0-179, so 6 bins).
                          # IMG_1294: blue 108-111, green 77-84, pink 164-173. A red near 0/179 would be split


# Object used for storing data for a sector
@dataclass(frozen=True)
class Sector:
    index: int                      # 1-based, clockwise from 3 o'clock
    start_angle: float              # first divider (degrees, clockwise from 3 o'clock)
    end_angle: float                # next divider clockwise
    value: float                    # calculated from the angle between the dividers (%)
    ocr_value: Optional[float]      # written percentage read by OCR (None if not trusted)
    label: Optional[str]            # all OCR text inside the sector (None if there is none)


# ---------------------------------------------------------------- OCR

# Create PaddleOCR once (loading the models is slow)
@functools.lru_cache(maxsize=None)
def _paddle():
    """PaddleOCR with page rotation and unwarping switched off, otherwise the box coordinates
    would belong to a changed image, not to ours. The other settings are the defaults."""
    from paddleocr import PaddleOCR   # imported here: loading it is slow, and only the pie detector needs it
    # enable_mkldnn=False: paddlepaddle 3.3.1 crashes in its oneDNN CPU path on Linux x86
    # ("ConvertPirAttribute2RuntimeAttribute not support"). Macs with Apple chips do not use oneDNN anyway
    settings = dict(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                    enable_mkldnn=False)
    return PaddleOCR(**settings)


# Read all the text in an image
def _read_text(ocr, image):
    """Run PaddleOCR and return a list of (4 corner points, text, confidence)."""
    page = ocr.predict(image)[0]
    return [(poly.tolist(), text, score)
            for poly, text, score in zip(page["rec_polys"], page["rec_texts"], page["rec_scores"])]


# ---------------------------------------------------------------- 0. coloured sectors -> line drawing

# Turn coloured (hatched) sectors into a black line drawing
def _colour_to_lines(image):
    """Pies with coloured sectors have no drawn dividers: a divider is only the edge between two colours.
    This paints the coloured area white and draws the colour edges in black, so the image looks like
    a black ink drawing. An image with no coloured area comes back unchanged."""
    # 1. the coloured pixels: the biggest minus the smallest of B, G, R is high (grey, black and
    #    white get about 0). Averaging first bridges the white gaps between the hatching strokes
    chroma = cv2.blur(image.max(axis=2) - image.min(axis=2), (FLAT_BLUR, FLAT_BLUR))
    coloured = (chroma > CHROMA_MIN).astype(np.uint8)

    # 2. keep only the big coloured areas: small spots are colour noise of a scan
    _, blobs, stats, _ = cv2.connectedComponentsWithStats(coloured)
    big = stats[:, cv2.CC_STAT_AREA] >= FILL_MIN * coloured.size
    big[0] = False                                        # blob 0 is the background
    coloured = big[blobs]                                 # True/False per pixel
    if not coloured.any():
        return image                                      # no coloured sectors: nothing to do

    # 3. one label per fill: the hue (0-179) in bins of HUE_BIN -> 1, 2, ... 6, and 0 = not coloured.
    #    The first median blur flattens the hatching (the hue is the same in the light and dark
    #    streaks), the second one gives single pixels the label around them
    hue = cv2.cvtColor(cv2.medianBlur(image, FLAT_BLUR), cv2.COLOR_BGR2HSV)[:, :, 0]
    labels = cv2.medianBlur(np.where(coloured, hue // HUE_BIN + 1, 0).astype(np.uint8), FLAT_BLUR)

    # 4. a black ink drawing: fills white, dividers black. A divider is where the label changes:
    #    the morphological gradient (biggest - smallest label in a 5x5 window) is not 0 there.
    #    Black ink (the rim, the text) is not coloured, so it stays as it is
    out = image.copy()
    out[coloured] = 255
    out[cv2.morphologyEx(labels, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0] = 0
    return out


# ---------------------------------------------------------------- 1. text removal

# Create a mask of the dark ink
def _ink_mask(gray):
    """Binary image: 255 where the pixel is dark ink."""
    block = int(WORK_SIZE / 20) | 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, block, INK_C)


# Find the long straight lines, so the text removal does not erase them
def _long_lines(ink, boxes):
    """Mask of long straight lines (pie dividers and rim pieces). Text strokes are too short to be found.
    A line that lies fully inside the text boxes is skipped: it is the edge of a word, not a chart line."""
    h, w = ink.shape
    length = int(TEXT_LINE_MIN * min(h, w))
    lines = cv2.HoughLinesP(ink, 1, np.pi / 360, threshold=length // 2,
                            minLineLength=length, maxLineGap=5)
    mask = np.zeros_like(ink)
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4).astype(float):
            one = np.zeros_like(ink)
            cv2.line(one, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
            if not ((one > 0) & (boxes == 0)).any():   # no pixel of the line is outside the boxes
                continue
            # make the segment a bit longer at both ends: where a drawn line bends, Hough finds
            # two pieces with a small gap between them, and this closes the gap
            length = np.hypot(x2 - x1, y2 - y1)
            step_x = (x2 - x1) / length                # a 1 px step along the line
            step_y = (y2 - y1) / length
            start = (int(x1 - step_x * LINE_EXTEND), int(y1 - step_y * LINE_EXTEND))
            end = (int(x2 + step_x * LINE_EXTEND), int(y2 + step_y * LINE_EXTEND))
            cv2.line(mask, start, end, 255, LINE_BAND)
    return mask


# Find the pie rim, so the text removal does not erase it
def _pie_circle(gray):
    """Mask of a band around the pie rim (Hough circle)."""
    mask = np.zeros_like(gray)
    circle = _find_circle(gray)
    if circle is not None:
        # wide band, because a hand-drawn rim is not a perfect circle
        cv2.circle(mask, circle[:2], circle[2], 255, RIM_BAND)
    return mask


# Erase the text from the ink mask
def _remove_text(gray, results):
    """Erase the ink inside the OCR boxes, but keep the long straight lines and the rim.
    Returns the ink mask without the text: 255 = ink."""
    boxes = np.zeros(gray.shape, np.uint8)
    for corners, _, _ in results:
        cv2.fillPoly(boxes, [np.array(corners, np.int32)], 255)
    boxes = cv2.dilate(boxes, np.ones((2 * BOX_PAD + 1, 2 * BOX_PAD + 1), np.uint8))

    ink = _ink_mask(gray)
    text = cv2.bitwise_and(ink, boxes)
    keep = cv2.bitwise_or(_long_lines(ink, boxes), _pie_circle(gray))
    text = cv2.bitwise_and(text, cv2.bitwise_not(keep))
    # grow the text a little, so the ends of letters inside the protected bands are erased too
    text = cv2.dilate(text, np.ones((2 * EDGE_PAD + 1, 2 * EDGE_PAD + 1), np.uint8))
    return cv2.bitwise_and(ink, cv2.bitwise_not(text))


# ---------------------------------------------------------------- 2.-3. circle and ink centre

# Find the pie circle
def _find_circle(gray):
    """Hough circle transform. Returns (x, y, r) of the strongest circle, or None."""
    h, w = gray.shape
    blurred = cv2.GaussianBlur(cv2.medianBlur(gray, 5), (9, 9), 2)
    # start strict, and lower the number of votes needed if no circle is found
    for votes in (60, 45, 30, 20):
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=max(h, w),
                                   param1=100, param2=votes,
                                   minRadius=int(0.15 * min(h, w)), maxRadius=int(0.6 * max(h, w)))
        if circles is not None:
            x, y, r = circles.reshape(-1, circles.shape[-1])[0][:3]
            return int(round(x)), int(round(y)), int(round(r))
    return None


# Find the divider lines that start near the circle centre
def _find_segments(ink, cx, cy, r):
    """Long straight segments inside the circle that start near the Hough centre (the dividers)."""
    inside = np.zeros_like(ink)
    cv2.circle(inside, (cx, cy), int(INSIDE * r), 255, -1)
    lines = cv2.HoughLinesP(cv2.bitwise_and(ink, inside), 1, np.pi / 360, threshold=int(0.2 * r),
                            minLineLength=int(DIV_LINE_MIN * r), maxLineGap=int(LINE_GAP * r))
    return [(x1, y1, x2, y2) for x1, y1, x2, y2 in (lines.reshape(-1, 4) if lines is not None else [])
            if _end_dist(x1, y1, x2, y2, cx, cy) < MAX_END_DIST * r]


# Distance from a point to the nearer end of a segment
def _end_dist(x1, y1, x2, y2, x, y):
    """Distance from (x, y) to the nearer end of the segment."""
    return min(math.hypot(x1 - x, y1 - y), math.hypot(x2 - x, y2 - y))


# Find the point where the dividers meet
def _ink_centre(segments):
    """The point closest to all lines (least squares, every line counts the same), or None."""
    rows, values = [], []
    for x1, y1, x2, y2 in segments:
        d = np.array([x2 - x1, y2 - y1], dtype=float)
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        rows.append(n)
        values.append(n @ np.array([x1, y1]))
    if len(rows) < 2 or np.linalg.cond(np.array(rows)) > 1e3:      # 1 line, or all lines parallel
        return None
    x, y = np.linalg.lstsq(np.array(rows), np.array(values))[0]
    return float(x), float(y)


# Find the circle and the ink centre
def _find_centre(gray, ink):
    """Returns (hough circle, ink centre or None), or None."""
    circle = _find_circle(gray)
    if circle is None:
        return None
    cx, cy, r = circle
    used = _find_segments(ink, cx, cy, r)
    centre = _ink_centre(used)
    if centre is not None:
        # second round: only the segments that start near the first estimate (drops text baselines)
        good = [seg for seg in used if _end_dist(*seg, *centre) < MAX_END_DIST_2 * r]
        if _ink_centre(good) is not None:
            centre = _ink_centre(good)
    return circle, centre


# ---------------------------------------------------------------- 4.-7. unroll, histogram, sectors

# Unroll the pie into a rectangle
def _unroll(ink, centre, r):
    """Ink around the centre as a rectangle: column = angle (0..360, clockwise from 3 o'clock),
    row = distance from the centre (0..R px). Returns a uint8 image, 255 = ink."""
    polar = cv2.warpPolar(ink, (r, ANGLES), centre, r, cv2.WARP_POLAR_LINEAR + cv2.INTER_LINEAR)
    return (polar.T > 127).astype(np.uint8) * 255   # transpose: angles go along the x axis


# Find the divider angles: the peaks in the ink histogram
def _peaks(share):
    """Angles (degrees, clockwise from 3 o'clock) of the runs where the share is above PEAK.
    The run may wrap around 360."""
    above = share >= PEAK
    start = int(np.argmin(above))   # start the walk at a low point, so no run is cut at 0 degrees
    runs, run = [], []
    # once around plus one step: the walk ends on the low point again, so the last run is closed too.
    # No ink peak at all, or ink everywhere: no run is closed -> []
    for i in range(start, start + len(above) + 1):
        if above[i % len(above)]:
            run.append(i)
        elif run:
            runs.append(run)
            run = []
    step = 360 / len(above)
    return [(np.mean(run) * step) % 360 for run in runs]


# Angle of a vector
def _angle_of(dx, dy):
    """Angle of the vector (dx, dy) in image coordinates, clockwise from 3 o'clock (degrees).
    The y axis of an image points down, so atan2 goes clockwise."""
    return math.degrees(math.atan2(dy, dx)) % 360


# Turn the divider angles into sectors
def _sectors(angles):
    """Divider angles -> list of (start, end, value %) clockwise.
    The first sector starts at the first divider clockwise from 3 o'clock (the smallest angle)."""
    angles = sorted(angles)
    # size = angle to the next divider.
    # area share = angle / 360 (the same d for every sector)
    return [(a, b, (b - a) % 360 / 360 * 100) for a, b in zip(angles, angles[1:] + angles[:1])]


# ---------------------------------------------------------------- 8.-10. labels: OCR text per sector

# Is an angle inside a sector?
def _in_sector(angle, sector):
    """True if the angle lies between the two dividers of the sector."""
    start, end, _ = sector
    from_start = (angle - start) % 360     # how far clockwise the angle is from the first divider
    width = (end - start) % 360            # size of the sector in degrees
    return from_start < width


# Join the text boxes of one sector in reading order
def _reading_order(boxes):
    """The boxes of one sector -> one text, read like a page: lines from top to bottom,
    boxes in a line from left to right. Every box is a dict with x, y (centre), height and text."""
    lines = []                                         # every line is a list of boxes
    for box in sorted(boxes, key=lambda b: b["y"]):    # top to bottom
        if lines:
            first = lines[-1][0]                       # the first (highest) box of the current line
            # less than half a box height lower than that box -> it is on the same line
            if box["y"] - first["y"] < first["height"] / 2:
                lines[-1].append(box)
                continue
        lines.append([box])                            # otherwise it starts a new line

    words = []
    for line in lines:
        for box in sorted(line, key=lambda b: b["x"]):   # left to right
            words.append(box["text"])
    return " ".join(words)


# Put every text box inside the circle into its sector
def _text_per_sector(results, centre, r, found):
    """One text per sector: all OCR boxes inside the circle, in reading order."""
    boxes = [[] for _ in found]     # boxes[k] = the OCR boxes in sector k
    for corners, text, _ in results:
        corners = np.array(corners, float)
        x, y = corners.mean(axis=0)                     # centre of the box
        dx, dy = x - centre[0], y - centre[1]           # from the ink centre to the box
        if math.hypot(dx, dy) >= r:                     # outside the circle: title, notes -> not used
            continue
        # the sector whose two dividers are around the angle of the box
        angle = _angle_of(dx, dy)
        for i, sector in enumerate(found):
            if _in_sector(angle, sector):
                k = i
                break
        height = corners[:, 1].max() - corners[:, 1].min()
        boxes[k].append({"x": x, "y": y, "height": height, "text": text})

    return [_reading_order(sector_boxes) for sector_boxes in boxes]


# Read the percentages from the sector texts
def _ocr_values(texts):
    """The percentages in every sector text. Returns (values per sector, use them?)."""
    values = []
    for text in texts:
        # every "25%" or "38.78%" in the text. A comma counts as a decimal point ("6,12%")
        values.append([float(v.replace(",", ".")) for v in PERCENT.findall(text)])

    if any(len(vs) != 1 for vs in values):             # a sector with no value or with two values
        return values, False

    total = sum(vs[0] for vs in values)
    # a small rounding error is fine (e.g. 25 + 22 + 20 + 17 + 17 = 101)
    return values, abs(total - 100) <= SUM_TOL


# Detect and extract a pie graph from an image and return an object
def detect_pie_graph(image):
    """Detect a pie graph in a preprocessed BGR image.

    Returns a list of `Sector` objects, clockwise from 3 o'clock (an empty list if no pie
    circle or no dividers are found).

    Sector.value is calculated from the angle between the dividers. Sector.ocr_value is the
    written percentage, but only if every sector has exactly one and they add up to 100;
    otherwise it is None for all sectors.
    """
    return find_pie(image)[0]


# Detect a pie graph and also say where it is
def find_pie(image):
    """Like `detect_pie_graph`, but returns (sectors, (centre x, centre y, radius)), with the
    circle in the input image's pixels, so callers can tell where the pie is.
    Returns ([], None) if no pie is found.
    """
    scale = WORK_SIZE / max(image.shape[:2])
    image = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    # 0. coloured sectors -> line drawing
    image = _colour_to_lines(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1.-3. text removal, circle and ink centre
    results = _read_text(_paddle(), image)
    clean = _remove_text(gray, results)   # 255 = ink
    circle = _find_centre(255 - clean, clean)
    if circle is None or circle[1] is None:
        return [], None
    (_, _, r), centre = circle

    # 4.-7. unroll, histogram, peaks -> dividers -> sectors
    polar = _unroll(clean, centre, r)
    share = polar[int(BAND_IN * r):int(BAND_OUT * r)].mean(axis=0) / 255
    angles = _peaks(share)
    if not angles:
        return [], None
    found = _sectors(angles)

    # 8.-10. one text per sector (same OCR boxes), percentages from it
    texts = _text_per_sector(results, centre, r, found)
    values, ok = _ocr_values(texts)

    sectors = [Sector(k + 1, round(float(start), 1), round(float(end), 1), round(float(value), 1),
                      values[k][0] if ok else None, texts[k] or None)
               for k, (start, end, value) in enumerate(found)]
    return sectors, (float(centre[0]) / scale, float(centre[1]) / scale, float(r) / scale)


# Preprocess a photo and print the sectors of its pie graph
def main():
    if len(sys.argv) < 2:
        print("Usage: python -m chartx.detection.pie_graph photo.jpg")
        exit(1)
    path = sys.argv[1]

    image = cv2.imread(path)
    if image is None:
        print(f"Error: image at {path} not found")
        exit(1)

    from chartx.main import preprocess   # imported here: chartx.main loads the whole pipeline (deskew, matplotlib)
    image = preprocess(image)

    sectors = detect_pie_graph(image)
    if not sectors:
        print("No pie graph found")
    print(f"{'index':>5}  {'start':>6}  {'end':>6}  {'value':>6}  {'ocr':>6}  label")
    for s in sectors:
        ocr = "-" if s.ocr_value is None else f"{s.ocr_value:g}"
        print(f"{s.index:>5}  {s.start_angle:>6}  {s.end_angle:>6}  {s.value:>6}  {ocr:>6}  {s.label or '-'}")


if __name__ == "__main__":
    main()

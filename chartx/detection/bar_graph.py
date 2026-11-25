"""Bar graph detection. Expects a PREPROCESSED image: deskewed, white board, no glare/shadow."""
import warnings
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# Object used for storing data for a bar
@dataclass(frozen=True)
class Bar:
    index: int                      # 1-based, left -> right
    x: int                          # left edge (px)
    y: int                          # top edge (px)
    width: int                      # px
    height: int                     # px
    relative_height: float          # height / tallest bar
    value: Optional[float]          # read from the y-axis labels (None if unreadable)
    colour: str                     # "#rrggbb"
    label: Optional[str] = None     # the text written under the bar (None if there is none)


# Is a number odd?
def _odd(n):
    n = int(n)
    return max(3, n + (n % 2 == 0))


# Create a mask to extract the backgrong from the chart
def _ink_mask(image, ink_threshold=40):
    ink = 255 - image.min(axis=2)          # low min-channel = coloured or dark
    ink = cv2.GaussianBlur(ink, (5, 5), 0)
    return (ink > ink_threshold).astype(np.uint8) * 255


# Fix gaps, holes and thin lines in the mask
def _clean_mask(mask, k_close, k_open):
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
    return filled


# Find all the bars (rectangles) in an image
def detect_bars(image, min_bars=3, min_extent=0.8, baseline_tol=0.02):
    """Find bars. Returns an (n, 4) array of x, y, w, h sorted left -> right.

    A shape is a bar if it is a solid rectangle, not tiny, not touching the image
    edge, on the same baseline as the others, and a similar width. Returns an empty
    array if fewer than `min_bars` qualify (i.e. not a bar graph).
    """
    empty = np.empty((0, 4), int)
    h, w = image.shape[:2]
    side = min(h, w)
    mask = _clean_mask(_ink_mask(image), _odd(side * 0.008), _odd(side * 0.02))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return empty

    boxes = np.array([cv2.boundingRect(c) for c in contours])
    areas = np.array([cv2.contourArea(c) for c in contours])
    x, y, bw, bh = boxes.T
    box_area = bw * bh

    on_border = (x <= 2) | (y <= 2) | (x + bw >= w - 2) | (y + bh >= h - 2)
    keep = ((areas / np.maximum(box_area, 1) >= min_extent)
            & (box_area >= 0.002 * h * w) & ~on_border)
    boxes = boxes[keep]
    if len(boxes) < min_bars:
        return empty

    # Same baseline: keep the biggest group whose bottoms line up
    bottoms = boxes[:, 1] + boxes[:, 3]
    same_base = np.abs(bottoms[:, None] - bottoms[None, :]) <= baseline_tol * h
    boxes = boxes[same_base[same_base.sum(axis=1).argmax()]]

    # Similar width
    median_w = np.median(boxes[:, 2])
    boxes = boxes[np.abs(boxes[:, 2] - median_w) <= 0.5 * median_w]

    if len(boxes) < min_bars:
        return empty
    return boxes[np.argsort(boxes[:, 0])]


# Find the small blobs of ink (the axis labels) in the strip left of the plot
def _label_blobs(strip):
    height = strip.shape[0]
    _, ink = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Delete long thin lines (the axis and its ticks) so they don't touch the digits
    tall = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(height * 0.15))))
    wide = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(height * 0.04)), 1))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, tall) | cv2.morphologyEx(ink, cv2.MORPH_OPEN, wide)
    ink = cv2.subtract(ink, cv2.dilate(lines, np.ones((3, 3), np.uint8)))

    # Join the digits of a number ("1" + "0"), then keep blobs that are label sized
    joined = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    x, y, w, h = stats[1:, :4].T
    return stats[1:, :4][(h >= 0.012 * height) & (h <= 0.05 * height) & (w <= 3 * h)]


# Find the rows of the gridlines / y-axis ticks, from the baseline (value 0) upwards
def _gridlines(image, boxes, dark=150):
    """Return the pixel rows of the gridlines, starting with the baseline.

    Looks at the strip just left of the first bar, where the y-axis ticks, labels
    and gridlines are, and counts the dark pixels on each row (vertical lines like
    the y-axis itself are removed first). Rows with a peak in that count are
    candidate gridlines. Gridlines are roughly evenly spaced up from the baseline,
    so the longest evenly spaced run of peaks is taken as the axis. Stray marks,
    titles and the whiteboard frame don't fit the spacing and are skipped.
    """
    baseline = int(np.median(boxes[:, 1] + boxes[:, 3]))
    first_bar = int(boxes[:, 0].min())
    pad = max(3, int(0.005 * image.shape[0]))
    left = max(0, int(first_bar - 1.5 * np.median(boxes[:, 2])))

    # Dark pixels, minus anything tall and thin (the y-axis)
    ink = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < dark).astype(np.uint8)
    tall = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(0.05 * baseline))))
    ink = cv2.subtract(ink, cv2.morphologyEx(ink, cv2.MORPH_OPEN, tall))

    # Fraction of dark pixels on each row of the strip, smoothed so tilted or dotted lines still show
    profile = ink[:baseline, left:first_bar - pad].mean(axis=1)
    k = max(3, int(0.01 * baseline) | 1)
    profile = np.convolve(profile, np.ones(k) / k, "same")

    # Peaks: highest row within k either side, and clearly above the background level
    floor = np.median(profile)
    peaks = [y for y in range(k, baseline - 2 * k)
             if profile[y] - floor >= 0.1 * (profile.max() - floor)
             and profile[y] == profile[y - k:y + k + 1].max()]

    # Starting from the baseline and one peak, keep stepping up by the average spacing so far,
    # taking the strongest peak near each expected row, until there isn't one
    def run_from(peak):
        rows = [baseline, peak]
        while True:
            spacing = (baseline - rows[-1]) / (len(rows) - 1)
            expected = rows[-1] - spacing
            near = [y for y in peaks if abs(y - expected) <= 0.25 * spacing]
            if not near:
                return rows
            rows.append(max(near, key=lambda y: profile[y]))

    return max((run_from(p) for p in peaks), key=len, default=[baseline])


# Read the value at the top of each bar using the gridlines
def _bar_values(image, boxes, axis_min=0, axis_max=10):
    """Values at the bar tops, assuming the gridlines go evenly from axis_min to axis_max."""
    rows = _gridlines(image, boxes)
    if len(rows) < 3:      # need two gridlines above the baseline to trust the spacing
        warnings.warn("No gridlines found, so bar values can't be read")
        return [None] * len(boxes)

    print(f"Gridline rows (bottom to top): {rows}")

    # Interpolate between the gridlines either side of each bar top
    # (np.interp needs increasing x, so flip the rows to negative)
    values = np.linspace(axis_min, axis_max, len(rows))
    return np.interp(-boxes[:, 1], -np.array(rows), values).round(2).tolist()


# Get the colour of a bar (rectangle)
def _bar_colour(image, box):
    """Typical colour from the middle of the bar, as #rrggbb."""
    x, y, w, h = box
    mid = image[y + h // 4: y + 3 * h // 4, x + w // 4: x + 3 * w // 4].reshape(-1, 3)
    b, g, r = np.percentile(mid, 25, axis=0).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


# Match the text found under the bars to the bar above it
def _assign_labels(results, boxes):
    """Give each bar the text whose centre lies within the bar's width (widened by a quarter
    on each side, as a label can be a little off centre). Text that sits under no bar, such
    as an axis number, is dropped. Several words under one bar are joined left to right.

    results: (4 corner points, text, confidence) for each piece of text, as read by OCR,
        in the same pixel coordinates as `boxes`.
    """
    words = [[] for _ in boxes]
    for poly, text, _ in results:
        centre_x = float(np.mean([point[0] for point in poly]))
        for i, (x, _, w, _) in enumerate(boxes):
            if x - w / 4 <= centre_x <= x + w + w / 4:
                words[i].append((centre_x, text.strip()))
                break
    return [" ".join(text for _, text in sorted(found)) or None for found in words]


# Read the labels written under the bars
def _bar_labels(image, boxes):
    """OCR the strip just below the baseline and match each piece of text to its bar.

    Returns one label (or None) per bar. If PaddleOCR is not installed a warning is
    given and every label is None.
    """
    baseline = int(np.median(boxes[:, 1] + boxes[:, 3]))
    strip_height = max(40, int(0.2 * boxes[:, 3].max()))
    top = baseline + 2                                   # just below the axis line
    strip = image[top: min(image.shape[0], baseline + strip_height)]
    if strip.shape[0] < 5:                               # the bars reach the bottom of the photo
        return [None] * len(boxes)

    try:
        from chartx.detection import ocr                     # loads PaddleOCR on the first call
        found = ocr.read_page(strip)
    except ImportError:
        warnings.warn("PaddleOCR is not installed, so the bar labels can't be read")
        return [None] * len(boxes)

    return _assign_labels(found, boxes)


# Detect and extract a bar graph from an image and return an object
def detect_bar_graph(image, calibration=None, horizontal=False):
    """Detect a bar graph in a preprocessed BGR image.

    Returns a list of `Bar` objects, left to right (an empty list if no bar graph
    is found).

    Each bar's label is read with OCR from the text written under it (vertical bars
    only; for horizontal bars `Bar.label` is None).

    Bar values are read from the horizontal gridlines, assuming they go evenly
    from 0 at the baseline to 10 at the top one. If no gridlines are found a
    warning is given and `Bar.value` is None.

    calibration: optional manual override (y_px_1, value_1, y_px_2, value_2): the
        pixel rows of two axis ticks and their values.
    horizontal: True for horizontal bars. Axis labels are not read automatically
        for these, so pass `calibration` (pixel columns of two ticks) for values.
    """
    if horizontal:
        image = np.rot90(image).copy()      # baseline on the left -> baseline at the bottom

    boxes = detect_bars(image)
    if len(boxes) == 0:
        return []

    x, y, w, h = boxes.T
    relative = h / h.max()

    if calibration is not None:
        y1, v1, y2, v2 = calibration
        values = (v1 + (y - y1) * (v2 - v1) / (y2 - y1)).round(3).tolist()   # value at bar top
    elif horizontal:
        warnings.warn("Axis labels are not read for horizontal bars; pass calibration= for values")
        values = [None] * len(boxes)
    else:
        values = _bar_values(image, boxes)

    labels = [None] * len(boxes) if horizontal else _bar_labels(image, boxes)

    return [Bar(i + 1, int(x[i]), int(y[i]), int(w[i]), int(h[i]),
                round(float(relative[i]), 3), values[i], _bar_colour(image, boxes[i]), labels[i])
            for i in range(len(boxes))]
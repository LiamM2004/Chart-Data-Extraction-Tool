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
def detect_bars(image, min_bars=3, min_extent=0.85, baseline_tol=0.02):
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


# Read the y-axis numbers and work out how many value units one pixel is worth
def _axis_scale(image, boxes, axis_min=0, axis_max=10):
    """
    Calculate value per pixel using OpenCV horizontal gridline detection.
    """

    baseline = np.median(boxes[:, 1] + boxes[:, 3])

    # Restrict image to the chart area
    left = max(0, int(boxes[:, 0].min()) - 45)
    right = min(
        image.shape[1],
        int((boxes[:, 0] + boxes[:, 2]).max()) + 10
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    roi = gray[:int(baseline), left:right]

    edges = cv2.Canny(roi, 30, 100)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=25,
        minLineLength=20,  # Reduced from 60% of chart width
        maxLineGap=20
    )

    if lines is None:
        warnings.warn("No horizontal gridlines detected")
        return None

    horizontal_lines = []

    for x1, y1, x2, y2 in lines.reshape(-1, 4):

        line_length = abs(x2 - x1)

        if (
            abs(y2 - y1) <= 3
            and line_length >= 20
            and y1 < boxes[:, 1].min()
        ):
            horizontal_lines.append((y1 + y2) / 2)

    if not horizontal_lines:
        warnings.warn("No suitable horizontal gridlines found")
        return None

    # Top gridline represents maximum axis value
    axis_top = max(horizontal_lines)

    plot_height = baseline - axis_top

    if plot_height <= 0:
        warnings.warn("Invalid plot height")
        return None

    scale = (axis_max - axis_min) / plot_height

    print(f"Axis top: {axis_top}")
    print(f"Axis baseline: {baseline}")
    print(f"Axis scale: {scale}")

    return scale


# Get the colour of a bar (rectangle)
def _bar_colour(image, box):
    """Typical colour from the middle of the bar, as #rrggbb."""
    x, y, w, h = box
    mid = image[y + h // 4: y + 3 * h // 4, x + w // 4: x + 3 * w // 4].reshape(-1, 3)
    b, g, r = np.percentile(mid, 25, axis=0).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


# Detect and extract a bar graph from an image and return an object
def detect_bar_graph(image, calibration=None, horizontal=False):
    """Detect a bar graph in a preprocessed BGR image.

    Returns a list of `Bar` objects, left to right (an empty list if no bar graph
    is found).

    Bar values are read automatically from the y-axis labels (needs pytesseract),
    assuming the axis starts at 0. If they can't be read a warning is given and
    `Bar.value` is None.

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
        scale = _axis_scale(image, boxes)

        values = (
            [None] * len(boxes)
            if scale is None
            else (h * scale).round(3).tolist()
        )

    return [Bar(i + 1, int(x[i]), int(y[i]), int(w[i]), int(h[i]),
                round(float(relative[i]), 3), values[i], _bar_colour(image, boxes[i]))
            for i in range(len(boxes))]
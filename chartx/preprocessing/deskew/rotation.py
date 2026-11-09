"""
rotation.py - work out how far a photo is tilted, so it can be levelled.

Two ways to measure the tilt, depending on what the chart has:

    tilt_from_chart_axes(find_long_straight_lines(img))   # bar / line charts: the "L" of the two axes
    tilt_from_writing(img, pie)                           # pie charts: no axes, so use the text rows

Tilt is in degrees; positive means the image is tilted counter-clockwise, so
rotate it by minus that to level it.

Requires: opencv-python, numpy
"""

from __future__ import annotations

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Finding the marker ink and its long straight lines
# --------------------------------------------------------------------------- #
def find_ink(img: np.ndarray) -> np.ndarray:
    """
    Binary mask of marker ink (white = ink). Adaptive thresholding copes with
    the uneven lighting and glare that whiteboard photos nearly always have.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    block = max(15, (min(gray.shape) // 30) | 1)     # odd, scales with image size
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, block, 15)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return _without_ink_touching_the_border(mask)


def _without_ink_touching_the_border(mask: np.ndarray) -> np.ndarray:
    """Drop anything touching the image border (board frame, leftover wall, rotation seams). Real chart ink sits inside the board."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    x, y, bw, bh = (stats[:, i] for i in range(4))
    touches = (x <= 1) | (y <= 1) | (x + bw >= w - 1) | (y + bh >= h - 1)
    touches[0] = True                                  # background label
    mask[touches[labels]] = 0
    return mask


def find_long_straight_lines(img: np.ndarray, min_len_frac: float = 0.25, ignoring_pie=None) -> np.ndarray:
    """
    Long straight segments of ink, as an Nx4 array of (x1, y1, x2, y2).
    Pass a pie (from find_pie_ellipse) as `ignoring_pie` to leave its hatching and
    outline out: strokes inside a pie can line up into a convincing but false "L".
    """
    mask = find_ink(img)
    if ignoring_pie is not None:
        _erase_the_pie(mask, ignoring_pie)
    h, w = mask.shape
    min_len = int(min_len_frac * min(h, w))
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 720, threshold=80,
                            minLineLength=min_len,
                            maxLineGap=max(10, min_len // 10))  # bridges wobbly freehand gaps
    return np.empty((0, 4), int) if lines is None else lines.reshape(-1, 4)


# --------------------------------------------------------------------------- #
# Tilt from the chart's axes
# --------------------------------------------------------------------------- #
def tilt_from_chart_axes(lines: np.ndarray, max_tilt: float = 20.0,
                         max_shear: float = 8.0, corner_frac: float = 0.15) -> float | None:
    """
    Estimate the image tilt (degrees) from the chart's axes.

    The axes are the longest near-horizontal and near-vertical segments that
    meet in a corner (an "L"): handwritten labels, titles and legends have long
    straight-ish strokes too, but nothing perpendicular joining them. The two
    axes' deviations from level are averaged, weighted by length, so a freehand
    pair that isn't quite square is levelled evenly. Hough reports one stroke as
    several overlapping segments, so near-identical segments are counted once.

    Returns None when no L is found - e.g. a pie chart - and the caller should
    leave the image as it is (or try tilt_from_writing).
    """
    near_axis = _segments_near_level_or_upright(lines, max_tilt)
    if near_axis is None:
        return None
    segments, length, is_horizontal, deviation = near_axis
    first_of_its_line = _is_first_segment_of_its_line(segments, length)
    pair = _longest_corner_pair(segments, length, is_horizontal, deviation, first_of_its_line,
                                max_shear, corner_frac)
    if pair is None:
        return None
    i, j = pair
    return float((deviation[i] * length[i] + deviation[j] * length[j]) / (length[i] + length[j]))


def _segments_near_level_or_upright(lines: np.ndarray, max_tilt: float):
    """
    Keep the segments within `max_tilt` degrees of horizontal or vertical.
    Returns (segments, lengths, is_horizontal, deviation in degrees from level / upright), or None.
    """
    if len(lines) == 0:
        return None
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    # Image y points down; flip so angles follow the usual convention.
    angle = np.degrees(np.arctan2(-dy, dx))
    angle = (angle + 90) % 180 - 90                    # fold into [-90, 90)

    off_horizontal = angle
    off_vertical = np.where(angle >= 0, angle - 90, angle + 90)
    is_horizontal = np.abs(off_horizontal) <= max_tilt
    is_vertical = np.abs(off_vertical) <= max_tilt
    keep = is_horizontal | is_vertical
    if not keep.any():
        return None
    deviation = np.where(is_horizontal[keep], off_horizontal[keep], off_vertical[keep])
    return lines[keep], np.hypot(dx, dy)[keep], is_horizontal[keep], deviation


def _is_first_segment_of_its_line(lines: np.ndarray, length: np.ndarray,
                                  max_angle_deg: float = 0.5, max_offset_px: float = 8.0) -> np.ndarray:
    """
    True for the longest segment of each distinct line, False for the segments
    that merely repeat it (same direction within `max_angle_deg`, same position
    within `max_offset_px`).
    """
    dx, dy = lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]
    theta = np.arctan2(dy, dx) % np.pi
    offset = -np.sin(theta) * lines[:, 0] + np.cos(theta) * lines[:, 1]
    first = np.zeros(len(lines), bool)
    taken: list[int] = []
    for i in np.argsort(-length):
        for j in taken:
            gap = abs(theta[i] - theta[j])
            if min(gap, np.pi - gap) < np.radians(max_angle_deg) and abs(offset[i] - offset[j]) < max_offset_px:
                break
        else:
            first[i] = True
            taken.append(i)
    return first


def _longest_corner_pair(segments, length, is_horizontal, deviation, first_of_its_line,
                         max_shear: float, corner_frac: float) -> tuple[int, int] | None:
    """The (horizontal, vertical) pair of distinct segments that meet in a corner and are longest together."""
    best, best_total = None, 0.0
    for i in np.flatnonzero(first_of_its_line & is_horizontal):
        for j in np.flatnonzero(first_of_its_line & ~is_horizontal):
            if abs(deviation[i] - deviation[j]) > max_shear:
                continue
            if not _segments_meet_in_a_corner(segments[i], segments[j], corner_frac * min(length[i], length[j])):
                continue
            if length[i] + length[j] > best_total:
                best, best_total = (i, j), length[i] + length[j]
    return best


def _segments_meet_in_a_corner(a: np.ndarray, b: np.ndarray, tolerance_px: float) -> bool:
    """True if segments a and b, extended to lines, cross within `tolerance_px` of both."""
    p, r = a[:2].astype(float), (a[2:] - a[:2]).astype(float)
    q, s = b[:2].astype(float), (b[2:] - b[:2]).astype(float)
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) < 1e-9:
        return False
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / cross
    corner = p + t * r
    return (_distance_from_point_to_segment(corner, a) <= tolerance_px
            and _distance_from_point_to_segment(corner, b) <= tolerance_px)


def _distance_from_point_to_segment(point: np.ndarray, segment: np.ndarray) -> float:
    start, direction = segment[:2].astype(float), (segment[2:] - segment[:2]).astype(float)
    along = np.clip(np.dot(point - start, direction) / np.dot(direction, direction), 0, 1)
    return float(np.linalg.norm(point - (start + along * direction)))


# --------------------------------------------------------------------------- #
# Tilt from the writing (pie charts)
# --------------------------------------------------------------------------- #
def tilt_from_writing(img: np.ndarray, pie, max_tilt: float = 10.0, step: float = 0.25,
                      min_peak: float = 1.05) -> float | None:
    """
    Estimate the image tilt (degrees) from the writing, for a pie chart, which
    has no axes to level on.

    Titles, legends and labels are written in rows. Rotate the ink through
    `max_tilt` either way and the angle at which the rows are sharpest (the row
    profile of the ink has the most variance) is the angle at which the text is
    level. The pie itself is blanked out first: its outline and hatching point
    every which way and would blur the peak.

    Handwriting is noisy and bar or line charts can carry sideways text, so
    this is only meant for a pie with no axes. Returns None unless the peak is
    well inside the search range and clearly above a typical angle's score.
    """
    ink = _ink_without_the_pie(img, pie)
    if ink is None:
        return None
    angles = np.arange(-max_tilt, max_tilt + step / 2, step)
    scores = np.array([_how_row_like_the_ink_is(ink, angle) for angle in angles])
    best = int(scores.argmax())
    if best in (0, len(angles) - 1) or scores[best] < min_peak * np.median(scores):
        return None
    return float(-angles[best])      # rotating the ink by +a levelled it, so it was tilted by -a


def _ink_without_the_pie(img: np.ndarray, pie) -> np.ndarray | None:
    """Ink on a 1000 px copy with the pie blanked out (0-1 floats), or None if there is almost no ink left."""
    scale = 1000 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = find_ink(small)
    _erase_the_pie(mask, pie, scale)
    if cv2.countNonZero(mask) < 0.001 * mask.size:
        return None
    return mask.astype(np.float32) / 255


def _erase_the_pie(mask: np.ndarray, pie, scale: float = 1.0):
    """Blank out the pie (a little wider than its outline) from an ink mask that is `scale` times the photo's size."""
    (cx, cy), (d0, d1), angle = pie
    cv2.ellipse(mask, ((cx * scale, cy * scale), (d0 * scale * 1.15, d1 * scale * 1.15), angle), 0, -1)


def _how_row_like_the_ink_is(ink: np.ndarray, rotate_by_deg: float) -> float:
    """Higher when, after rotating, the ink falls into distinct rows (text lines) rather than a smear."""
    h, w = ink.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate_by_deg, 1.0)
    rows = cv2.warpAffine(ink, M, (w, h)).sum(axis=1)
    return float((rows ** 2).sum())


# --------------------------------------------------------------------------- #
# Applying a rotation
# --------------------------------------------------------------------------- #
def rotation_transform(shape: tuple, angle_deg: float) -> tuple[np.ndarray, tuple[int, int]]:
    """3x3 matrix rotating an image of `shape` by angle_deg (CCW positive) without cropping corners."""
    h, w = shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - w / 2
    M[1, 2] += new_h / 2 - h / 2
    return np.vstack([M, [0, 0, 1]]), (new_w, new_h)

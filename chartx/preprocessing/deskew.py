"""
deskew.py - Step 1 of the chart extraction pipeline: geometric pre-processing.

Implements the two-step approach from the skew research post:

    1. Fix perspective  - find the board's four corners and warp them back to a
                          rectangle. If no board is found (e.g. a close-up of
                          just the chart, or auto-detection isn't confident),
                          fall back to user-supplied corners.
    2. Level the axes   - find the chart's long straight lines with a
                          probabilistic Hough transform, estimate the tilt from
                          the dominant near-horizontal / near-vertical lines,
                          and counter-rotate.

The detected axis lines are returned too, so the next pipeline stage (axis
detection / tick reading) can reuse them instead of redoing the work.

Usage (CLI):
    python -m chartx.preprocessing.deskew photo.jpg -o out.png              # auto; opens the drag window if unsure
    python -m chartx.preprocessing.deskew photo.jpg -o out.png --manual     # always open the drag window
    python -m chartx.preprocessing.deskew photo.jpg -o out.png --no-prompt  # never open a window (batch use)
    python -m chartx.preprocessing.deskew photo.jpg -o out.png --corners 10,20 900,15 920,700 5,690
    python -m chartx.preprocessing.deskew photo.jpg -o out.png --debug debug.png

Requires: opencv-python, numpy
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from chartx.preprocessing.ellipse import circularise_transform, find_pie_ellipse
from chartx.preprocessing.geometry import edge_coverage, hough_lines, intersect, order_corners
from chartx.preprocessing.partial_board import find_partial_board_corners


# Tilts smaller than this (degrees) are not worth a resample.
MIN_ROTATION_DEG = 0.2

# A pie outline this much off round (long/short axis) is stretched back to a circle.
PIE_MIN_STRETCH = 1.03

# Share of a pie's outline that must be visible before we level an axis-less
# chart on its writing (0.8 is required to stretch; one photo's pie had 0.7-0.8).
TEXT_PIE_COVERAGE = 0.65

# A board seen with one side out of shot is only trusted when its three visible
# edges are this well supported (the good cases scored 0.97+, a wall scored 0.56).
PARTIAL_MIN = 0.8

# Auto-detected corners with less edge support than this need a human look.
# Chosen from a handful of photos (good ones scored 0.56-0.79, bad ones 0.11-0.13).
REVIEW_BELOW = 0.4


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class DeskewResult:
    image: np.ndarray                       # final corrected image
    corners: np.ndarray | None              # board corners (TL, TR, BR, BL) in input coords; if
                                            # corner_source is "none", only a suggestion, not applied
    corner_source: str                      # "auto", "partial", "manual", "provided" or "none" (not warped)
    confidence: float                       # auto-detection edge support, 0-1 (1.0 if corners were given)
    needs_review: bool                      # True if a human should check/adjust the corners
    rotation_deg: float                     # rotation applied in step 2
    axis_lines: list = field(default_factory=list)  # Hough segments (x1,y1,x2,y2) in final image
    warnings: list = field(default_factory=list)
    pie_stretch: float = 1.0                # stretch applied to make a pie round (1.0 = none)


# --------------------------------------------------------------------------- #
# Step 1 - perspective: find & straighten the board
# --------------------------------------------------------------------------- #
def _clip_to_quad(line, quad: np.ndarray):
    """Endpoints of the part of `line` (rho, theta) inside the convex `quad`, or None."""
    n = np.array([np.cos(line[1]), np.sin(line[1])])
    d = quad @ n - line[0]                        # signed distance of each corner
    pts = [quad[i] + d[i] / (d[i] - d[(i + 1) % 4]) * (quad[(i + 1) % 4] - quad[i])
           for i in range(4) if d[i] * d[(i + 1) % 4] < 0]
    return (pts[0], pts[1]) if len(pts) == 2 else None


def _count_crossing_lines(quad: np.ndarray, edges: np.ndarray, horiz, vert,
                          sides_h, sides_v, near_px: float) -> int:
    """
    Count long, edge-backed lines that cut across the quad without being one of
    its sides. A real board has none (its frame is the boundary); a quad that
    overshoots the board - say it also takes in the wall below - has the
    board's true edge running straight through it.
    """
    width = (np.linalg.norm(quad[0] - quad[1]) + np.linalg.norm(quad[3] - quad[2])) / 2
    height = (np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] - quad[2])) / 2
    count = 0
    for lines, sides, span in ((horiz, sides_h, width), (vert, sides_v, height)):
        for line in lines:
            if line in sides:
                continue
            # The frame has an inner and outer edge; a line hugging a side isn't a crossing.
            if any(abs(line[0] - r) < near_px and abs(line[1] - t) < np.radians(5) for r, t in sides):
                continue
            seg = _clip_to_quad(line, quad)
            if seg is None or np.linalg.norm(seg[0] - seg[1]) < 0.5 * span:
                continue
            if edge_coverage(edges, *seg) >= 0.7:
                count += 1
    return count


def find_board_corners(img: np.ndarray,
                       min_area_frac: float = 0.08,
                       work_width: int = 1000) -> tuple[np.ndarray | None, float]:
    """
    Look for the board as the quadrilateral whose four sides all sit on strong edges.

    Candidates are built from pairs of near-horizontal and near-vertical Hough
    lines. Each is scored by how much of its perimeter is actually backed by
    edge pixels (the board's frame is, the wall's imaginary outline is not), so
    the board beats the bigger wall around it.

    Returns (corners, confidence). Corners are ordered TL, TR, BR, BL, or None
    if no candidate exists. Confidence is the edge support of the candidate's
    weakest side (0-1), halved for each long edge-backed line that cuts through
    the quad. It is low when a board edge is out of shot or the guess
    overshoots the board; the corners are still returned as a starting point
    for manual adjustment.

    Works on a downscaled copy for speed, then scales the corners back up.
    """
    h, w = img.shape[:2]
    scale = work_width / w if w > work_width else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else img
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    thick = cv2.dilate(edges, np.ones((5, 5), np.uint8))   # tolerate slightly-off lines
    img_area = gray.shape[0] * gray.shape[1]
    near_px = 0.04 * np.hypot(*gray.shape)

    horiz = hough_lines(edges, vertical=False)
    vert = hough_lines(edges, vertical=True)
    best, best_score, best_conf = None, 0.0, 0.0
    for top, bottom in itertools.combinations(horiz, 2):
        for left, right in itertools.combinations(vert, 2):
            pts = [intersect(top, left), intersect(top, right),
                   intersect(bottom, right), intersect(bottom, left)]
            if any(p is None for p in pts):
                continue
            quad = np.array(pts, np.float32)
            area = cv2.contourArea(quad) / img_area
            if not (min_area_frac < area < 0.9) or not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
                continue
            cov = [edge_coverage(thick, quad[i], quad[(i + 1) % 4]) for i in range(4)]
            crossings = _count_crossing_lines(quad, thick, horiz, vert,
                                              (top, bottom), (left, right), near_px)
            conf = min(cov) * 0.5 ** crossings
            score = conf * np.mean(cov)
            if score > best_score:
                best, best_score, best_conf = quad, score, conf

    if best is None:
        return None, 0.0
    return order_corners(best / scale), best_conf


def estimate_aspect_ratio(corners: np.ndarray, image_shape: tuple,
                          focal_frac: float = 0.72) -> float | None:
    """
    True width/height of the rectangle seen at `corners` (TL, TR, BR, BL).

    From Zhang & He (2004), "Whiteboard scanning and image enhancement": a
    rectangle's four corners fix its aspect ratio once the camera is known.
    Their paper also estimates the focal length from the same corners, but
    that goes unstable when a pair of sides is nearly parallel (it failed on a
    near head-on photo here), so we assume a typical phone main camera instead:
    square pixels, optical centre mid-photo, focal length `focal_frac` x the
    photo's long side. The ratio is only mildly sensitive to that guess.

    Returns None if the corners don't form a usable quad.
    """
    h, w = image_shape[:2]
    f = focal_frac * max(h, w)
    tl, tr, br, bl = (np.array([x, y, 1.0]) for x, y in order_corners(corners))
    with np.errstate(all="ignore"):
        k2 = np.dot(np.cross(tl, br), bl) / np.dot(np.cross(tr, br), bl)
        k3 = np.dot(np.cross(tl, br), tr) / np.dot(np.cross(bl, br), tr)
        A_inv = np.linalg.inv(np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]))
        n2, n3 = A_inv @ (k2 * tr - tl), A_inv @ (k3 * bl - tl)
        r2 = np.dot(n2, n2) / np.dot(n3, n3)
    return float(np.sqrt(r2)) if np.isfinite(r2) and r2 > 0 else None


def rectifying_transform(corners: np.ndarray, image_shape: tuple) -> tuple[np.ndarray, tuple[int, int]]:
    """
    3x3 matrix taking the quadrilateral `corners` (TL, TR, BR, BL) to an upright
    rectangle with the board's true proportions (see estimate_aspect_ratio).
    Returns (matrix, output size (w, h)).
    """
    tl, tr, br, bl = order_corners(corners)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    ratio = estimate_aspect_ratio(corners, image_shape)
    if ratio is not None and 0.2 < ratio < 5:
        # Keep roughly the same pixel count, but with the right shape.
        width = np.sqrt(width * height * ratio)
        height = width / ratio
    width, height = int(round(width)), int(round(height))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), dst)
    # Corner detection is never pixel-perfect, so a sliver of wall/frame
    # usually survives along the edges. Trim it - otherwise those dead-straight
    # edge strips look like axes to the Hough step.
    m = max(2, int(0.01 * min(width, height)))
    shift = np.array([[1, 0, -m], [0, 1, -m], [0, 0, 1]], float)
    return shift @ M, (width - 2 * m, height - 2 * m)


def warp_to_rectangle(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the board quadrilateral `corners` (TL, TR, BR, BL) to an upright rectangle."""
    M, size = rectifying_transform(corners, img.shape)
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


class _CornerEditor:
    """
    State and drawing for the drag-the-corners window, kept apart from the
    OpenCV window loop so it can be driven without a display.

    The photo sits on a larger grey canvas so a handle can be dragged outside
    it - for a board whose corner is out of shot. Each side is extended across
    the canvas as a guide line, so you can line it up with a visible board edge
    instead of judging the corner by eye, and a magnifier follows the active
    handle. Coordinates handed in and out are in original image pixels and may
    lie outside the photo.
    """
    PAD = 0.3          # grey margin around the photo, as a fraction of its size
    GRAB = 25          # px radius for picking up a handle
    LOUPE, ZOOM = 150, 4
    HINT = ("drag handles, or 1-4 to pick + w/a/s/d to nudge", "Enter accept | Esc cancel")

    def __init__(self, img: np.ndarray, initial: np.ndarray | None = None, max_size: int = 1000):
        h, w = img.shape[:2]
        total = np.array([w, h]) * (1 + 2 * self.PAD)
        self.scale = min(1.0, max_size / total.max())
        cw, ch = (total * self.scale).round().astype(int)
        self.offset = np.array([w, h]) * self.PAD * self.scale
        self.offset = self.offset.round()
        self.base = np.full((ch, cw, 3), 70, np.uint8)
        photo = cv2.resize(img, None, fx=self.scale, fy=self.scale)
        ox, oy = self.offset.astype(int)
        self.base[oy:oy + photo.shape[0], ox:ox + photo.shape[1]] = photo
        self.photo_box = (ox, oy, ox + photo.shape[1] - 1, oy + photo.shape[0] - 1)
        if initial is None:
            m = 0.1
            initial = np.array([[w * m, h * m], [w * (1 - m), h * m],
                                [w * (1 - m), h * (1 - m)], [w * m, h * (1 - m)]])
        self.pts = [list(order_corners(initial)[i] * self.scale + self.offset) for i in range(4)]
        self.active = 0
        self.dragging = False

    def mouse(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            dists = [np.hypot(x - px, y - py) for px, py in self.pts]
            if min(dists) < self.GRAB:
                self.active, self.dragging = int(np.argmin(dists)), True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self._move(self.active, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def key(self, k: int) -> str | None:
        """Handle a key press; returns "accept", "cancel" or None."""
        if k in (13, 10):
            return "accept"
        if k == 27:
            return "cancel"
        if ord("1") <= k <= ord("4"):
            self.active = k - ord("1")
        elif chr(k) in "wasd":
            dx, dy = {"w": (0, -1), "a": (-1, 0), "s": (0, 1), "d": (1, 0)}[chr(k)]
            x, y = self.pts[self.active]
            self._move(self.active, x + dx, y + dy)
        return None

    def _move(self, i: int, x: float, y: float):
        h, w = self.base.shape[:2]
        self.pts[i] = [min(max(x, 0), w - 1), min(max(y, 0), h - 1)]

    def result(self) -> np.ndarray:
        """Corners (TL, TR, BR, BL) in original image pixels, possibly off-photo."""
        pts = (np.array(self.pts, np.float32) - self.offset) / self.scale
        return order_corners(pts)

    def render(self) -> np.ndarray:
        h, w = self.base.shape[:2]
        frame = self.base.copy()
        x0, y0, x1, y1 = self.photo_box
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1)   # where the photo ends
        pts = np.array(self.pts)
        for i in range(4):                                             # sides, extended as guides
            p, q = pts[i], pts[(i + 1) % 4]
            d = (q - p) / max(np.linalg.norm(q - p), 1e-6)
            ok, a, b = cv2.clipLine((0, 0, w, h), tuple(int(v) for v in p - d * 5000),
                                    tuple(int(v) for v in p + d * 5000))
            if ok:
                cv2.line(frame, a, b, (0, 220, 255), 1)
        cv2.polylines(frame, [pts.astype(np.int32)], True, (0, 200, 0), 2)
        loupe = self._loupe(frame)
        for i, (px, py) in enumerate(pts.astype(int)):
            colour = (0, 255, 255) if i == self.active else (0, 0, 255)
            cv2.circle(frame, (px, py), 8, colour, -1)
            cv2.putText(frame, str(i + 1), (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
        # Park the magnifier in a corner the active handle isn't near.
        ax, ay = pts[self.active]
        lx = w - self.LOUPE - 10 if ax < self.LOUPE + 60 and ay < self.LOUPE + 60 else 10
        frame[10:10 + self.LOUPE, lx:lx + self.LOUPE] = loupe
        cv2.rectangle(frame, (lx, 10), (lx + self.LOUPE, 10 + self.LOUPE), (255, 255, 255), 1)
        for row, text in enumerate(self.HINT):
            for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(frame, text, (10, h - 34 + 20 * row), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colour, thick, cv2.LINE_AA)
        return frame

    def _loupe(self, frame: np.ndarray) -> np.ndarray:
        """Magnified view of `frame` around the active handle, with a crosshair."""
        half = self.LOUPE // (2 * self.ZOOM)
        padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_CONSTANT, value=(70, 70, 70))
        x, y = (int(round(v)) for v in self.pts[self.active])
        crop = padded[y:y + 2 * half, x:x + 2 * half]
        loupe = cv2.resize(crop, (self.LOUPE, self.LOUPE), interpolation=cv2.INTER_NEAREST)
        c = self.LOUPE // 2
        cv2.line(loupe, (c, 0), (c, self.LOUPE), (0, 0, 255), 1)
        cv2.line(loupe, (0, c), (self.LOUPE, c), (0, 0, 255), 1)
        return loupe


def select_corners_manually(img: np.ndarray,
                            initial: np.ndarray | None = None,
                            window: str = "Adjust the 4 corners") -> np.ndarray | None:
    """
    Desktop fallback: show the image with four draggable handles (see
    _CornerEditor). Returns the corners, which may lie outside the photo, or
    None if cancelled.
    (In a mobile/web front end you'd do the same thing in the UI and pass the
    result to `deskew(..., corners=...)`.)
    """
    editor = _CornerEditor(img, initial)
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, lambda event, x, y, flags, _: editor.mouse(event, x, y))
    result = None
    while True:
        cv2.imshow(window, editor.render())
        k = cv2.waitKey(20) & 0xFF
        action = editor.key(k) if k != 255 else None
        if action == "accept":
            result = editor.result()
        if action:
            break
    cv2.destroyWindow(window)
    return result


# --------------------------------------------------------------------------- #
# Step 2 - rotation: level the axes with a Hough transform
# --------------------------------------------------------------------------- #
def ink_mask(img: np.ndarray) -> np.ndarray:
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
    # Drop anything touching the image border (board frame, leftover wall,
    # rotation seams). Real chart ink sits inside the board.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape
    x, y, bw, bh = (stats[:, i] for i in range(4))
    touches = (x <= 1) | (y <= 1) | (x + bw >= w - 1) | (y + bh >= h - 1)
    touches[0] = True                                  # background label
    mask[touches[labels]] = 0
    return mask


def detect_lines(img: np.ndarray, min_len_frac: float = 0.25) -> np.ndarray:
    """Long straight segments via probabilistic Hough. Returns Nx4 (x1,y1,x2,y2)."""
    mask = ink_mask(img)
    h, w = mask.shape
    min_len = int(min_len_frac * min(h, w))
    lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 720, threshold=80,
                            minLineLength=min_len,
                            maxLineGap=max(10, min_len // 10))  # bridges wobbly freehand gaps
    return np.empty((0, 4), int) if lines is None else lines.reshape(-1, 4)


def _first_of_each_line(lines: np.ndarray, length: np.ndarray,
                        max_angle_deg: float = 0.5, max_offset_px: float = 8.0) -> np.ndarray:
    """
    1.0 for the longest segment of each distinct line, 0.0 for the segments that
    merely repeat it (same direction within `max_angle_deg`, same position
    within `max_offset_px`).
    """
    dx, dy = lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]
    theta = np.arctan2(dy, dx) % np.pi
    offset = -np.sin(theta) * lines[:, 0] + np.cos(theta) * lines[:, 1]
    first = np.zeros(len(lines))
    taken: list[int] = []
    for i in np.argsort(-length):
        for j in taken:
            gap = abs(theta[i] - theta[j])
            if min(gap, np.pi - gap) < np.radians(max_angle_deg) and abs(offset[i] - offset[j]) < max_offset_px:
                break
        else:
            first[i] = 1.0
            taken.append(i)
    return first


def _distance_to_segment(p: np.ndarray, seg: np.ndarray) -> float:
    a, v = seg[:2].astype(float), (seg[2:] - seg[:2]).astype(float)
    u = np.clip(np.dot(p - a, v) / np.dot(v, v), 0, 1)
    return float(np.linalg.norm(p - (a + u * v)))


def _meet_in_a_corner(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    """True if segments a and b, extended to lines, cross within `tol` px of both."""
    p, r = a[:2].astype(float), (a[2:] - a[:2]).astype(float)
    q, s = b[:2].astype(float), (b[2:] - b[:2]).astype(float)
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) < 1e-9:
        return False
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / cross
    corner = p + t * r
    return _distance_to_segment(corner, a) <= tol and _distance_to_segment(corner, b) <= tol


def estimate_tilt(lines: np.ndarray, max_tilt: float = 20.0,
                  max_shear: float = 8.0, corner_frac: float = 0.15
                  ) -> tuple[float | None, np.ndarray]:
    """
    Estimate the image tilt (degrees, positive = counter-clockwise in the
    usual maths sense) from the chart's axes.

    The axes are the longest near-horizontal and near-vertical segments that
    meet in a corner (an "L"): handwritten labels, titles and legends have long
    straight-ish strokes too, but nothing perpendicular joining them. The two
    axes' deviations from level are averaged, weighted by length, so a freehand
    pair that isn't quite square is levelled evenly. Hough reports one stroke as
    several overlapping segments, so near-identical segments are counted once.

    Returns (tilt, near-axis segments). Tilt is None when no L is found - e.g.
    a pie chart, which needs no levelling - and the caller should leave the
    image as it is.
    """
    if len(lines) == 0:
        return None, lines
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    length = np.hypot(dx, dy)
    # Image y points down; flip so angles follow the usual convention.
    angle = np.degrees(np.arctan2(-dy, dx))
    angle = (angle + 90) % 180 - 90                    # fold into [-90, 90)

    dev_h = angle                                     # deviation from horizontal
    dev_v = np.where(angle >= 0, angle - 90, angle + 90)  # deviation from vertical
    is_h = np.abs(dev_h) <= max_tilt
    is_v = np.abs(dev_v) <= max_tilt
    keep = is_h | is_v
    if not keep.any():
        return None, lines[:0]

    segs, length = lines[keep], length[keep]
    horizontal = is_h[keep]
    dev = np.where(horizontal, dev_h[keep], dev_v[keep])
    distinct = _first_of_each_line(segs, length) > 0

    best, best_len = None, 0.0
    for i in np.flatnonzero(distinct & horizontal):
        for j in np.flatnonzero(distinct & ~horizontal):
            if abs(dev[i] - dev[j]) > max_shear:
                continue
            if not _meet_in_a_corner(segs[i], segs[j], corner_frac * min(length[i], length[j])):
                continue
            if length[i] + length[j] > best_len:
                best, best_len = (i, j), length[i] + length[j]
    if best is None:
        return None, segs
    i, j = best
    return float((dev[i] * length[i] + dev[j] * length[j]) / (length[i] + length[j])), segs


def estimate_text_tilt(img: np.ndarray, pie, max_tilt: float = 10.0, step: float = 0.25,
                       min_peak: float = 1.05) -> float | None:
    """
    Estimate the image tilt (degrees, same sign convention as `estimate_tilt`)
    from the writing, for a pie chart, which has no axes to level on.

    Titles, legends and labels are written in rows. Rotate the ink through
    `max_tilt` either way and the angle at which the rows are sharpest (the row
    profile of the ink has the most variance) is the angle at which the text is
    level. The pie itself is blanked out first: its outline and hatching point
    every which way and would blur the peak.

    Handwriting is noisy and bar or line charts can carry sideways text, so
    this is only meant for a pie with no axes. Returns None unless the peak is
    well inside the search range and clearly above a typical angle's score.
    """
    scale = 1000 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = ink_mask(small)
    (cx, cy), (d0, d1), ang = pie
    cv2.ellipse(mask, ((cx * scale, cy * scale), (d0 * scale * 1.15, d1 * scale * 1.15), ang), 0, -1)
    if cv2.countNonZero(mask) < 0.001 * mask.size:
        return None

    mask = mask.astype(np.float32) / 255
    h, w = mask.shape
    angles = np.arange(-max_tilt, max_tilt + step / 2, step)
    scores = []
    for a in angles:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rows = cv2.warpAffine(mask, M, (w, h)).sum(axis=1)
        scores.append(float((rows ** 2).sum()))
    scores = np.array(scores)
    best = int(scores.argmax())
    if best in (0, len(angles) - 1) or scores[best] < min_peak * np.median(scores):
        return None
    return float(-angles[best])      # rotating the ink by +a levelled it, so it was tilted by -a


def rotation_transform(shape: tuple, angle_deg: float) -> tuple[np.ndarray, tuple[int, int]]:
    """3x3 matrix rotating an image of `shape` by angle_deg (CCW positive) without cropping corners."""
    h, w = shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return np.vstack([M, [0, 0, 1]]), (nw, nh)


def rotate_keep_all(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate by angle_deg (CCW positive) without cropping corners."""
    M, size = rotation_transform(img.shape, angle_deg)
    # Fill the new corners with the board's typical colour rather than
    # smearing edge pixels, which creates fake streaks.
    fill = tuple(float(v) for v in np.median(img.reshape(-1, img.shape[2] if img.ndim == 3 else 1), axis=0))
    return cv2.warpAffine(img, M[:2], size, flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def level_axes(img: np.ndarray, min_correction: float = MIN_ROTATION_DEG,
               max_tilt: float = 20.0) -> tuple[np.ndarray, float, np.ndarray]:
    """Detect tilt from the chart's own lines and counter-rotate."""
    tilt, _ = estimate_tilt(detect_lines(img), max_tilt)
    if tilt is None or abs(tilt) < min_correction:
        out, applied = img, 0.0
    else:
        out, applied = rotate_keep_all(img, -tilt), -tilt
    # Re-detect on the levelled image so callers get axis lines in final coords.
    _, final_lines = estimate_tilt(detect_lines(out), max_tilt)
    return out, applied, final_lines


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def deskew(img: np.ndarray,
           corners: np.ndarray | None = None) -> DeskewResult:
    """
    Run the pre-processing steps: straighten the board, remove leftover pie
    stretch, level the axes.

    corners       - optional user-supplied board/chart corners (any order).
                    Pass these from your UI when auto-detection fails.

    If the whole board isn't in the photo but three of its edges are, only the
    visible part is straightened (corner_source "partial").

    If auto-detection is not confident, `needs_review` is set, the photo is
    returned untouched and `corners` holds the best guess (or None). A guess we
    don't trust can warp the chart out of frame - a quad over the wall, say - so
    the caller should let the user adjust the corners (see
    select_corners_manually) and call again with `corners=`.

    After the board warp, a pie chart's outline (if there is one) is used to
    remove any stretch left over.
    """
    warnings = []

    # Step 1: perspective
    if corners is not None:
        corners, source, confidence = order_corners(corners), "provided", 1.0
    else:
        corners, confidence = find_board_corners(img)
        source = "auto"
        if corners is None or confidence < REVIEW_BELOW:
            # Maybe a board side is out of shot: try rectifying just the visible part.
            partial, partial_conf = find_partial_board_corners(img)
            if partial is not None and partial_conf >= PARTIAL_MIN:
                corners, confidence, source = partial, partial_conf, "partial"
                warnings.append("Part of the board is out of shot: only the visible part was "
                                "straightened (assumes its vertical edges are parallel).")
    if corners is None or confidence < REVIEW_BELOW:
        warnings.append(f"Board corners not found with confidence ({confidence:.2f}) - "
                        "please check or drag the corners into place. Image left unchanged.")
        return DeskewResult(img, corners, "none", confidence, True, 0.0, [], warnings)

    # Each correction is found on a quick preview and recorded as a 3x3 matrix.
    # The matrices are multiplied and the photo is resampled once at the end, so
    # the final image is not softened by repeated warping.
    M, size = rectifying_transform(corners, img.shape)
    board = cv2.perspectiveTransform(
        np.array([[[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]], np.float32),
        np.linalg.inv(M))[0]                                  # the trimmed board, in photo coordinates
    on_board = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(on_board, np.round(board).astype(np.int32), 255)

    def render(M, size, flags):
        out = cv2.warpPerspective(img, M, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
        valid = cv2.warpPerspective(on_board, M, size, flags=cv2.INTER_NEAREST) > 0
        if not valid.all():                                   # corners opened up by a rotation
            out[~valid] = np.median(out[valid], axis=0)       # fill with the board colour, not the wall
        return out

    preview = render(M, size, cv2.INTER_LINEAR)

    # Step 1b: a pie's outline should now be a circle; stretch away what's left.
    pie_stretch = 1.0
    pie = find_pie_ellipse(preview)
    if pie is not None and max(pie[1]) / min(pie[1]) >= PIE_MIN_STRETCH:
        stretch_M, size, pie_stretch = circularise_transform(pie, preview.shape)
        M = stretch_M @ M
        preview = render(M, size, cv2.INTER_LINEAR)

    # Step 2: rotation
    rot = 0.0
    tilt, _ = estimate_tilt(detect_lines(preview))
    if tilt is None:
        # A pie has no axes: level it on its writing instead. Only done when a pie
        # is there, so sideways text on a chart that merely missed its L can't drag
        # the image round. The pie gate is looser than for stretching (a pie with a
        # gap in its outline still tells us where to ignore); a wrong guess here
        # costs less than a wrong stretch.
        text_pie = find_pie_ellipse(preview, min_coverage=TEXT_PIE_COVERAGE)
        if text_pie is not None:
            tilt = estimate_text_tilt(preview, text_pie)
    if tilt is not None and abs(tilt) >= MIN_ROTATION_DEG:
        rotate_M, size = rotation_transform(preview.shape, -tilt)
        M, rot = rotate_M @ M, -tilt

    levelled = render(M, size, cv2.INTER_CUBIC)
    # Axis lines in final coordinates, for the next stage.
    _, axis_lines = estimate_tilt(detect_lines(levelled))
    if len(axis_lines) == 0:
        warnings.append("No long straight lines found - rotation not corrected.")

    return DeskewResult(levelled, corners, source, confidence, False, rot,
                        [tuple(map(int, l)) for l in axis_lines], warnings, pie_stretch)


def debug_view(original: np.ndarray, result: DeskewResult) -> np.ndarray:
    """Side-by-side: input with detected corners | output with Hough lines."""
    left = original.copy()
    if result.corners is not None:
        cv2.polylines(left, [result.corners.astype(np.int32)], True, (0, 200, 0), 4)
    right = result.image.copy()
    for x1, y1, x2, y2 in result.axis_lines:
        cv2.line(right, (x1, y1), (x2, y2), (0, 0, 255), 2)
    h = 600
    l = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
    r = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
    return np.hstack([l, np.full((h, 10, 3), 255, np.uint8), r])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_corners(vals):
    return np.array([[float(v) for v in p.split(",")] for p in vals], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("image")
    ap.add_argument("-o", "--output", default="deskewed.png")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--manual", action="store_true",
                      help="always open the drag-corners window, even if auto-detection is confident")
    mode.add_argument("--no-prompt", action="store_true",
                      help="never open a window; just warn when corners need review (for batch runs)")
    ap.add_argument("--corners", nargs=4, metavar="X,Y",
                    help="board corners in any order, e.g. 10,20 900,15 920,700 5,690")
    ap.add_argument("--debug", help="save a side-by-side debug image here")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")

    res = deskew(img, corners=_parse_corners(args.corners) if args.corners else None)

    # Ask the user to adjust the corners when asked to, or when auto-detection
    # wasn't confident. The window starts from the best guess, if there is one.
    if (args.manual or res.needs_review) and not args.no_prompt and not args.corners:
        picked = select_corners_manually(img, initial=res.corners)
        if picked is not None:
            res = deskew(img, corners=picked)
            res.corner_source = "manual"
    cv2.imwrite(args.output, res.image)
    print(f"corners: {res.corner_source}  rotation applied: {res.rotation_deg:+.2f} deg  "
          f"pie stretch: {res.pie_stretch:.2f}  axis lines: {len(res.axis_lines)}")
    for w in res.warnings:
        print("warning:", w)
    if args.debug:
        cv2.imwrite(args.debug, debug_view(img, res))


if __name__ == "__main__":
    main()
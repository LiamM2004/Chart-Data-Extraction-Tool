"""
geometry.py - small shared helpers for working with corners, edges and lines.

Requires: opencv-python, numpy
"""

from __future__ import annotations

import cv2
import numpy as np


# The camera we assume: square pixels, lens centred on the photo, and a focal
# length this multiple of the photo's long side - a typical phone's main camera
# (about 24 mm in 35mm terms). A different lens (ultra-wide, zoom) needs `--focal`.
ASSUMED_FOCAL_FRAC = 0.72


def camera_matrix(width: int, height: int, focal_frac: float = ASSUMED_FOCAL_FRAC) -> np.ndarray:
    """The 3x3 camera matrix for a photo of this size, under the assumptions above."""
    f = focal_frac * max(width, height)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as top-left, top-right, bottom-right, bottom-left.

    The points are sorted clockwise by their angle around the centre, which gives four
    different corners however the shape is rotated, and the list starts at the point
    nearest the photo's top-left (smallest x+y).
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    centre = pts.mean(axis=0)
    angle = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])    # y points down, so this runs clockwise
    clockwise = pts[np.argsort(angle)]
    return np.roll(clockwise, -int(np.argmin(clockwise.sum(axis=1))), axis=0)


def is_a_usable_quad(corners: np.ndarray, image_shape: tuple, min_area_frac: float = 0.005) -> bool:
    """True if `corners` are four finite points forming a convex shape that covers a real part of the photo."""
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if not np.all(np.isfinite(pts)):
        return False
    ordered = order_corners(pts)
    return (bool(cv2.isContourConvex(ordered.reshape(-1, 1, 2)))
            and cv2.contourArea(ordered) >= min_area_frac * image_shape[0] * image_shape[1])


def shrink_to_width(img: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """A copy of `img` no wider than `width` px (searching is faster on a small copy), and the shrink factor."""
    scale = width / img.shape[1] if img.shape[1] > width else 1.0
    if scale == 1.0:
        return img, scale
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def find_edge_maps(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where the brightness changes sharply: the thin edge map, and a thickened copy that forgives near misses."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    return edges, cv2.dilate(edges, np.ones((5, 5), np.uint8))


def find_edge_lines(edges: np.ndarray, vertical: bool, k: int = 8,
                    max_dev: float = 35.0) -> list[tuple[float, float]]:
    """Strongest near-vertical (or near-horizontal) lines as (rho, theta), with near-duplicates merged."""
    lines = cv2.HoughLines(edges, 1, np.pi / 360, int(0.25 * min(edges.shape)))
    out: list[tuple[float, float]] = []
    for rho, theta in ([] if lines is None else lines[:, 0]):
        # theta is the normal's angle: ~0 for vertical lines, ~90 deg for horizontal.
        dev = np.degrees(min(theta, np.pi - theta) if vertical else abs(theta - np.pi / 2))
        if dev > max_dev:
            continue
        if any(_is_near_duplicate(rho, theta, r, t) for r, t in out):
            continue                              # near-duplicate of a stronger line
        out.append((float(rho), float(theta)))
        if len(out) >= k:
            break
    return out


def _is_near_duplicate(rho: float, theta: float, other_rho: float, other_theta: float) -> bool:
    """
    True if two (rho, theta) lines are almost the same line. A line is also written as
    (-rho, theta - pi), so a near-vertical edge can show up at theta ~0 and again at theta ~pi.
    """
    for r, t in ((other_rho, other_theta), (-other_rho, other_theta - np.pi), (-other_rho, other_theta + np.pi)):
        if abs(rho - r) < 12 and abs(theta - t) < np.radians(3):
            return True
    return False


def where_lines_cross(line_a, line_b) -> np.ndarray | None:
    """The point where two (rho, theta) lines meet, or None if they are parallel."""
    A = np.array([[np.cos(line_a[1]), np.sin(line_a[1])], [np.cos(line_b[1]), np.sin(line_b[1])]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, [line_a[0], line_b[0]])


def as_homogeneous_line(line) -> np.ndarray:
    """A (rho, theta) line as the 3 numbers (a, b, c) of a*x + b*y + c = 0."""
    return np.array([np.cos(line[1]), np.sin(line[1]), -line[0]])


def line_direction(line) -> np.ndarray:
    """A unit vector along a (rho, theta) line (which way along it is arbitrary)."""
    return np.array([-np.sin(line[1]), np.cos(line[1])])


def edge_support_score(support: list[float], penalty: float = 1.0) -> tuple[float, float]:
    """
    (confidence, score) for a candidate board whose sides have this edge support (each 0-1).
    Confidence is the weakest side, times `penalty`; the score also rewards strong sides all round.
    """
    confidence = min(support) * penalty
    return confidence, confidence * float(np.mean(support))


def best_candidate(candidates):
    """The (score, confidence, corners) with the highest positive score, or None if there is none."""
    best = None
    for candidate in candidates:
        if candidate[0] > (best[0] if best else 0.0):
            best = candidate
    return best


def where_ray_leaves_photo(start: np.ndarray, direction: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Where the ray from `start` along `direction` leaves the photo."""
    far = start + direction * 10 * max(width, height)
    part = part_of_segment_in_photo(start, far, width, height)
    return None if part is None else part[1]


def part_of_segment_in_photo(p: np.ndarray, q: np.ndarray, width: int, height: int):
    """The part of the segment p-q inside the photo, as (start, end), or None. Either end of p-q may be off-photo."""
    if max(np.abs(p).max(), np.abs(q).max()) > 1e8:           # too far out for OpenCV's integer maths
        return None
    ok, a, b = cv2.clipLine((0, 0, width - 1, height - 1), tuple(int(v) for v in p), tuple(int(v) for v in q))
    return (np.array(a, float), np.array(b, float)) if ok else None


def share_of_segment_on_edges(edges: np.ndarray, p: np.ndarray, q: np.ndarray, n: int = 150) -> float:
    """Fraction of the segment p-q that lies on an edge pixel (off-image counts as a miss)."""
    t = np.linspace(0, 1, n)
    xs = (p[0] + (q[0] - p[0]) * t).round().astype(int)
    ys = (p[1] + (q[1] - p[1]) * t).round().astype(int)
    inside = (xs >= 0) & (xs < edges.shape[1]) & (ys >= 0) & (ys < edges.shape[0])
    hit = np.zeros(n, bool)
    hit[inside] = edges[ys[inside], xs[inside]] > 0
    return float(hit.mean())

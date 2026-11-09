"""
board.py - find the whiteboard in a photo and work out how to straighten it.

    corners, confidence = find_board_corners(photo)           # all four sides in shot
    matrix, size = board_to_rectangle_transform(corners, photo.shape)

(When a side of the board is out of shot, see three_edge_board.py.)

Requires: opencv-python, numpy
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np

from chartx.preprocessing.deskew.geometry import (ASSUMED_FOCAL_FRAC, best_candidate, camera_matrix, edge_support_score,
                                           find_edge_lines, find_edge_maps, order_corners, shrink_to_width,
                                           share_of_segment_on_edges, where_lines_cross)

# Corner detection is never pixel-perfect, so a sliver of wall or frame usually
# survives along the board's edges. This much is trimmed off every side (a
# fraction of the board's short side) - otherwise those dead-straight strips
# would look like chart axes.
EDGE_TRIM_FRAC = 0.01


# --------------------------------------------------------------------------- #
# Finding the board
# --------------------------------------------------------------------------- #
def find_board_corners(img: np.ndarray,
                       min_area_frac: float = 0.08,
                       work_width: int = 1000) -> tuple[np.ndarray | None, float]:
    """
    Look for the board as the quadrilateral whose four sides all sit on strong edges.

    Candidates are built from pairs of near-horizontal and near-vertical lines.
    Each is scored by how much of its outline is actually backed by edge pixels
    (the board's frame is, the wall's imaginary outline is not), so the board
    beats the bigger wall around it.

    Returns (corners, confidence). Corners are ordered TL, TR, BR, BL, or None
    if no candidate exists. Confidence is the edge support of the candidate's
    weakest side (0-1), halved for each long edge-backed line that cuts through
    the quad. It is low when a board edge is out of shot or the guess
    overshoots the board; the corners are still returned as a starting point
    for manual adjustment.

    Works on a downscaled copy for speed, then scales the corners back up.
    """
    small, scale = shrink_to_width(img, work_width)
    edges, thick_edges = find_edge_maps(small)
    horizontal = find_edge_lines(edges, vertical=False)
    vertical = find_edge_lines(edges, vertical=True)

    best = best_candidate(_scored_quads(horizontal, vertical, edges.shape, thick_edges, min_area_frac))
    if best is None:
        return None, 0.0
    _, confidence, quad = best
    return order_corners(quad / scale), confidence


def _scored_quads(horizontal, vertical, image_shape, thick_edges, min_area_frac: float):
    """
    Every convex, sensibly-sized four-sided shape made from two horizontal and two
    vertical lines, as (score, confidence, quad).
    """
    image_area = image_shape[0] * image_shape[1]
    near_px = 0.04 * np.hypot(*image_shape)
    for top, bottom in itertools.combinations(horizontal, 2):
        for left, right in itertools.combinations(vertical, 2):
            corners = [where_lines_cross(top, left), where_lines_cross(top, right),
                       where_lines_cross(bottom, right), where_lines_cross(bottom, left)]
            if any(c is None for c in corners):
                continue
            quad = np.array(corners, np.float32)
            area = cv2.contourArea(quad) / image_area
            if not (min_area_frac < area < 0.9) or not cv2.isContourConvex(quad.reshape(-1, 1, 2)):
                continue
            confidence, score = _rate_quad(quad, thick_edges, horizontal, vertical,
                                           (top, bottom), (left, right), near_px)
            yield score, confidence, quad


def _rate_quad(quad, thick_edges, horizontal, vertical, horizontal_sides, vertical_sides, near_px):
    """(confidence, score) for a candidate: how well its sides sit on edges, minus overshoot."""
    side_support = [share_of_segment_on_edges(thick_edges, quad[i], quad[(i + 1) % 4]) for i in range(4)]
    crossings = _count_lines_cutting_through(quad, thick_edges, horizontal, vertical,
                                             horizontal_sides, vertical_sides, near_px)
    return edge_support_score(side_support, penalty=0.5 ** crossings)    # each stray crossing line halves it


def _count_lines_cutting_through(quad: np.ndarray, edges: np.ndarray, horizontal, vertical,
                                 horizontal_sides, vertical_sides, near_px: float) -> int:
    """
    Count long, edge-backed lines that cut across the quad without being one of
    its sides. A real board has none (its frame is the boundary); a quad that
    overshoots the board - say it also takes in the wall below - has the
    board's true edge running straight through it.
    """
    width = (np.linalg.norm(quad[0] - quad[1]) + np.linalg.norm(quad[3] - quad[2])) / 2
    height = (np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] - quad[2])) / 2
    count = 0
    for lines, sides, span in ((horizontal, horizontal_sides, width), (vertical, vertical_sides, height)):
        for line in lines:
            if line in sides:
                continue
            # The frame has an inner and outer edge; a line hugging a side isn't a crossing.
            if any(abs(line[0] - r) < near_px and abs(line[1] - t) < np.radians(5) for r, t in sides):
                continue
            part_inside = _part_of_line_inside_quad(line, quad)
            if part_inside is None or np.linalg.norm(part_inside[0] - part_inside[1]) < 0.5 * span:
                continue
            if share_of_segment_on_edges(edges, *part_inside) >= 0.7:
                count += 1
    return count


def _part_of_line_inside_quad(line, quad: np.ndarray):
    """Endpoints of the part of `line` (rho, theta) inside the convex `quad`, or None."""
    n = np.array([np.cos(line[1]), np.sin(line[1])])
    d = quad @ n - line[0]                        # signed distance of each corner
    pts = [quad[i] + d[i] / (d[i] - d[(i + 1) % 4]) * (quad[(i + 1) % 4] - quad[i])
           for i in range(4) if d[i] * d[(i + 1) % 4] < 0]
    return (pts[0], pts[1]) if len(pts) == 2 else None


# --------------------------------------------------------------------------- #
# Straightening the board
# --------------------------------------------------------------------------- #
def estimate_aspect_ratio(corners: np.ndarray, image_shape: tuple,
                          focal_frac: float = ASSUMED_FOCAL_FRAC) -> float | None:
    """
    True width/height of the rectangle seen at `corners` (TL, TR, BR, BL).

    From Zhang & He (2004), "Whiteboard scanning and image enhancement": a
    rectangle's four corners fix its aspect ratio once the camera is known.
    Their paper also estimates the focal length from the same corners, but
    that goes unstable when a pair of sides is nearly parallel (it failed on a
    near head-on photo here), so we assume a typical phone's main camera instead
    (see ASSUMED_FOCAL_FRAC): `focal_frac` x the photo's long side. The ratio is
    only mildly sensitive to it for a phone's main camera.

    Returns None if the corners don't form a usable quad.
    """
    h, w = image_shape[:2]
    tl, tr, br, bl = (np.array([x, y, 1.0]) for x, y in order_corners(corners))
    with np.errstate(all="ignore"):
        k2 = np.dot(np.cross(tl, br), bl) / np.dot(np.cross(tr, br), bl)
        k3 = np.dot(np.cross(tl, br), tr) / np.dot(np.cross(bl, br), tr)
        A_inv = np.linalg.inv(camera_matrix(w, h, focal_frac))
        n2, n3 = A_inv @ (k2 * tr - tl), A_inv @ (k3 * bl - tl)
        r2 = np.dot(n2, n2) / np.dot(n3, n3)
    return float(np.sqrt(r2)) if np.isfinite(r2) and r2 > 0 else None


def board_to_rectangle_transform(corners: np.ndarray, image_shape: tuple,
                                 focal_frac: float = ASSUMED_FOCAL_FRAC) -> tuple[np.ndarray, tuple[int, int]]:
    """
    3x3 matrix taking the quadrilateral `corners` (TL, TR, BR, BL) to an upright
    rectangle with the board's true proportions (see estimate_aspect_ratio),
    with a thin margin trimmed off every side (see EDGE_TRIM_FRAC).
    Returns (matrix, output size (w, h)).
    """
    tl, tr, br, bl = order_corners(corners)
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    ratio = estimate_aspect_ratio(corners, image_shape, focal_frac)
    if ratio is not None and 0.2 < ratio < 5:
        # Keep roughly the same pixel count, but with the right shape.
        width = np.sqrt(width * height * ratio)
        height = width / ratio
    width, height = int(round(width)), int(round(height))
    upright = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    to_rectangle = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), upright)

    trim = max(2, int(EDGE_TRIM_FRAC * min(width, height)))
    shift = np.array([[1, 0, -trim], [0, 1, -trim], [0, 0, 1]], float)
    return shift @ to_rectangle, (width - 2 * trim, height - 2 * trim)

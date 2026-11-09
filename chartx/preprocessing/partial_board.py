"""
partial_board.py - find the board when one of its sides is out of shot.

The full-board search needs all four edges. Phone photos often cut one off,
but the top, bottom and one vertical edge are still there. From those we can
rectify the part of the board that *is* in the photo:

    - the two horizontal edges and the visible vertical edge are real board edges
    - the missing side is built from the photo's border: the horizontal edges
      are run out to where they leave the photo, and a line parallel to the
      visible vertical edge is dropped through that point

Assuming the vertical edges are parallel is the one guess in here. With a single
vertical edge in view there is nothing to measure convergence from (vanishing-
point methods were tried and were no better on these photos, and unstable), so
a camera tilted far up or down will leave some vertical stretch behind.

Requires: opencv-python, numpy
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np

from chartx.preprocessing.geometry import (as_homogeneous_line, best_candidate, edge_support_score, find_edge_lines,
                                           find_edge_maps, line_direction, order_corners, shrink_to_width,
                                           share_of_segment_on_edges, where_lines_cross, where_ray_leaves_photo)


def find_partial_board_corners(img: np.ndarray,
                               work_width: int = 1000,
                               min_visible_frac: float = 0.25
                               ) -> tuple[np.ndarray | None, float]:
    """
    Look for a board seen with one vertical side out of shot.

    Tries every (top, bottom, vertical) triple of strong lines, on either
    side of the vertical one, and scores it by edge support along the three
    visible edges. Returns (corners, confidence), with corners ordered TL, TR,
    BR, BL in image coordinates and confidence the weakest visible edge's
    support (0-1), or (None, 0.0). The quad only covers the visible part of the
    board; its out-of-shot side lies along the photo's border.
    """
    small, scale = shrink_to_width(img, work_width)
    edges, thick_edges = find_edge_maps(small)
    horizontal = find_edge_lines(edges, vertical=False, k=12)
    vertical = find_edge_lines(edges, vertical=True, k=12)
    height, width = edges.shape

    best = best_candidate(_scored_boards(horizontal, vertical, thick_edges, width, height, min_visible_frac))
    if best is None:
        return None, 0.0
    _, confidence, quad = best
    return order_corners(quad / scale), confidence


def _scored_boards(horizontal, vertical, thick_edges, width: int, height: int, min_visible_frac: float):
    """Every (top, bottom, side) triple, with the board on either side of the side edge, as (score, confidence, quad)."""
    for top, bottom in itertools.combinations(horizontal, 2):
        for side_line in vertical:
            top_corner, bottom_corner = where_lines_cross(top, side_line), where_lines_cross(bottom, side_line)
            if not _is_a_usable_visible_edge(top_corner, bottom_corner, width, height):
                continue
            for board_is_to_the_right in (True, False):
                found = _board_on_one_side(top, bottom, top_corner, bottom_corner, board_is_to_the_right,
                                           thick_edges, width, height, min_visible_frac)
                if found is not None:
                    yield found


def _is_a_usable_visible_edge(top_corner, bottom_corner, width: int, height: int) -> bool:
    """The visible vertical edge must meet both horizontals inside the photo and be reasonably tall."""
    if top_corner is None or bottom_corner is None:
        return False
    if not (_is_in_photo(top_corner, width, height) and _is_in_photo(bottom_corner, width, height)):
        return False
    return np.linalg.norm(top_corner - bottom_corner) >= 0.2 * height


def _board_on_one_side(top, bottom, top_corner, bottom_corner, board_is_to_the_right: bool,
                       thick_edges, width: int, height: int, min_visible_frac: float):
    """
    Try the board lying to one side of the visible vertical edge.
    Returns (score, confidence, quad), or None if that side doesn't work out.
    """
    side = 1 if board_is_to_the_right else -1
    exits = [where_ray_leaves_photo(corner, _direction_towards(line, side), width, height)
             for line, corner in ((top, top_corner), (bottom, bottom_corner))]
    if any(e is None for e in exits):
        return None
    top_exit, bottom_exit = exits
    visible = min(np.linalg.norm(top_exit - top_corner), np.linalg.norm(bottom_exit - bottom_corner))
    if visible < min_visible_frac * width:
        return None

    support = [share_of_segment_on_edges(thick_edges, top_corner, top_exit),
               share_of_segment_on_edges(thick_edges, bottom_corner, bottom_exit),
               share_of_segment_on_edges(thick_edges, top_corner, bottom_corner)]
    up = (bottom_corner - top_corner) / np.linalg.norm(bottom_corner - top_corner)
    quad = _close_the_missing_side(top, bottom, top_corner, bottom_corner, exits, up, width, height)
    if quad is None:
        return None
    confidence, score = edge_support_score(support)
    return score, confidence, quad


def _close_the_missing_side(top, bottom, top_corner, bottom_corner, exits, up, width: int, height: int):
    """
    Build the missing side: parallel to the visible vertical edge, through
    whichever exit point keeps all four corners in the photo (larger quad wins).
    """
    options = []
    for exit_point in exits:
        parallel_point = exit_point + up
        ends = [_where_line_meets_line_through(line, exit_point, parallel_point) for line in (top, bottom)]
        if any(end is None for end in ends):
            continue
        quad = np.array([top_corner, ends[0], ends[1], bottom_corner], np.float32)
        all_in_photo = all(_is_in_photo(corner, width, height) for corner in quad)
        options.append((all_in_photo, cv2.contourArea(quad), quad))
    if not options:
        return None
    preferred = [option for option in options if option[0]] or options
    return max(preferred, key=lambda option: option[1])[2]


def _is_in_photo(point, width: int, height: int, margin: float = 0.03) -> bool:
    return -margin * width <= point[0] <= (1 + margin) * width and -margin * height <= point[1] <= (1 + margin) * height


def _direction_towards(line, side: int) -> np.ndarray:
    """Unit vector along a (rho, theta) line, pointing right (side=1) or left (side=-1)."""
    d = line_direction(line)
    return d if d[0] * side > 0 else -d


def _where_line_meets_line_through(line, p: np.ndarray, q: np.ndarray) -> np.ndarray | None:
    """Where a (rho, theta) line meets the line through points p and q, or None if they are parallel."""
    through_pq = np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])
    meeting = np.cross(as_homogeneous_line(line), through_pq)
    return meeting[:2] / meeting[2] if abs(meeting[2]) > 1e-9 else None

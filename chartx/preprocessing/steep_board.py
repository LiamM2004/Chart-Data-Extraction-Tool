"""
steep_board.py - find the board when it is seen at a steep angle, with its corners out of shot.

The other finders need corners they can see (or a side that runs off the photo
cleanly). A board photographed from the side often has neither: the edges are
all in view but the corners are not. Three edges are enough, though, if we know
the camera (we assume a typical phone's; see ASSUMED_FOCAL_FRAC):

    - the board's top and bottom edges are parallel in real life, so in the photo
      they converge on a point. That point is the direction "along the board".
    - the side edge lies in a plane through the camera, and the board's "up"
      direction must be square to "along" - which pins "up" down.
    - with both directions known, we know how the board is tilted, and can turn
      it face-on.

The result is handed back as four corners (some will be off the photo), the same
as the other finders, so the rest of the pipeline doesn't change.

Because it leans on the camera guess and on a steep, strongly converging view,
the other finders are tried first; this is the last resort before asking a
person for the corners.

Requires: opencv-python, numpy
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np

from chartx.preprocessing.geometry import (ASSUMED_FOCAL_FRAC, as_homogeneous_line, best_candidate, camera_matrix,
                                           edge_support_score, find_edge_lines, find_edge_maps, line_direction,
                                           order_corners, part_of_segment_in_photo, share_of_segment_on_edges,
                                           shrink_to_width, where_lines_cross)

MAX_BOARD_TILT_DEG = 75.0     # a board turned further away than this is too edge-on to trust


def find_steep_board_corners(img: np.ndarray,
                             focal_frac: float = ASSUMED_FOCAL_FRAC,
                             work_width: int = 1000,
                             min_convergence_deg: float = 10.0,
                             min_visible_frac: float = 0.08) -> tuple[np.ndarray | None, float]:
    """
    Look for a board from three of its edges: two that converge (top and bottom) and one side.

    Every combination is tried, with the board lying to either side of the side
    edge, and scored by edge support along the visible part of each of the three
    edges. Returns (corners, confidence), with corners ordered TL, TR, BR, BL in
    image coordinates (possibly off the photo) and confidence the weakest edge's
    support (0-1), or (None, 0.0).

    The top and bottom edges must converge by at least `min_convergence_deg`;
    nearly parallel edges say too little about the board's tilt. The far side of
    the board is taken to be as far as the photo shows.
    """
    small, scale = shrink_to_width(img, work_width)
    edges, thick_edges = find_edge_maps(small)
    height, width = edges.shape
    camera = camera_matrix(width, height, focal_frac)
    horizontals = find_edge_lines(edges, vertical=False, k=10, max_dev=60)     # steep views tilt these a lot
    verticals = find_edge_lines(edges, vertical=True, k=10, max_dev=25)

    best = best_candidate(_scored_boards(camera, horizontals, verticals, thick_edges, width, height,
                                         min_convergence_deg, min_visible_frac))
    if best is None:
        return None, 0.0
    _, confidence, corners = best
    return order_corners(corners / scale), confidence


def _scored_boards(camera, horizontals, verticals, thick_edges, width: int, height: int,
                   min_convergence_deg: float, min_visible_frac: float):
    """Every (converging pair, side edge) triple, with the board on either side of the side edge, as (score, confidence, corners)."""
    for line_a, line_b in itertools.combinations(horizontals, 2):
        if _angle_between(line_a, line_b) < min_convergence_deg:
            continue
        for side_line in verticals:
            to_face_on = _face_on_homography(camera, line_a, line_b, side_line)
            if to_face_on is None:
                continue
            for board_is_to_the_right in (True, False):
                found = _board_on_one_side(to_face_on, line_a, line_b, side_line, board_is_to_the_right,
                                           thick_edges, width, height, min_visible_frac)
                if found is not None:
                    yield found


# --------------------------------------------------------------------------- #
# Turning the board face-on
# --------------------------------------------------------------------------- #
def _face_on_homography(camera: np.ndarray, line_a, line_b, side_line) -> np.ndarray | None:
    """
    The 3x3 matrix that would show the board face-on, worked out from its two
    converging edges and its side edge. None if the three don't make sense.
    """
    to_directions = np.linalg.inv(camera)
    along = to_directions @ np.cross(as_homogeneous_line(line_a), as_homogeneous_line(line_b))   # where the edges converge
    if np.linalg.norm(along) < 1e-9:
        return None
    along = _pointing(along / np.linalg.norm(along), axis=0)                             # towards the right

    side_plane_normal = camera.T @ as_homogeneous_line(side_line)       # the plane through the camera and the side edge
    up_down = np.cross(side_plane_normal, along)                    # in that plane, and square to "along"
    if np.linalg.norm(up_down) < 1e-9:
        return None
    up_down = _pointing(up_down / np.linalg.norm(up_down), axis=1)                       # towards the bottom

    facing = np.cross(along, up_down)                               # the board's normal
    if facing[2] < np.cos(np.radians(MAX_BOARD_TILT_DEG)):          # turned away too far (or mirrored)
        return None
    return camera @ np.vstack([along, up_down, facing]) @ to_directions


def _pointing(direction: np.ndarray, axis: int) -> np.ndarray:
    """`direction` or its opposite, whichever points the positive way along `axis`."""
    return direction if direction[axis] >= 0 else -direction


def _angle_between(line_a, line_b) -> float:
    return float(np.degrees(abs(line_a[1] - line_b[1])))


def _apply(matrix: np.ndarray, point) -> np.ndarray | None:
    """Move a point by a 3x3 matrix, or None if it ends up at infinity."""
    x, y, w = matrix @ np.array([point[0], point[1], 1.0])
    return None if abs(w) < 1e-9 else np.array([x / w, y / w])


# --------------------------------------------------------------------------- #
# Trying the board on one side of the side edge
# --------------------------------------------------------------------------- #
def _board_on_one_side(to_face_on, line_a, line_b, side_line, board_is_to_the_right: bool,
                       thick_edges, width: int, height: int, min_visible_frac: float):
    """
    Try the board lying to one side of the side edge.
    Returns (score, confidence, corners), or None if that side doesn't work out.
    """
    corner_a, corner_b = where_lines_cross(line_a, side_line), where_lines_cross(line_b, side_line)
    if corner_a is None or corner_b is None:
        return None
    flat_a, flat_b = _apply(to_face_on, corner_a), _apply(to_face_on, corner_b)
    if flat_a is None or flat_b is None:
        return None

    support = _support_along_the_three_edges(to_face_on, (line_a, corner_a, flat_a), (line_b, corner_b, flat_b),
                                             board_is_to_the_right, thick_edges, width, height, min_visible_frac)
    if support is None:
        return None

    corners = _board_corners(to_face_on, flat_a, flat_b, board_is_to_the_right, width, height)
    if corners is None:
        return None
    confidence, score = edge_support_score(support)
    return score, confidence, corners


def _support_along_the_three_edges(to_face_on, edge_a, edge_b, board_is_to_the_right: bool,
                                   thick_edges, width: int, height: int, min_visible_frac: float):
    """
    How much of each visible edge (the two converging ones, running towards the
    board, and the side edge between them) sits on real image edges. None if
    any of them is too short to judge.
    """
    segments = []
    for line, corner, flat_corner in (edge_a, edge_b):
        along = _direction_towards_the_board(to_face_on, line, corner, flat_corner, board_is_to_the_right)
        if along is None:
            return None
        segments.append(part_of_segment_in_photo(corner, corner + along * 10 * max(width, height), width, height))
    segments.append(part_of_segment_in_photo(edge_a[1], edge_b[1], width, height))

    if any(segment is None for segment in segments):
        return None
    shortest_allowed = min_visible_frac * np.hypot(width, height)
    if any(np.linalg.norm(end - start) < shortest_allowed for start, end in segments):
        return None
    return [share_of_segment_on_edges(thick_edges, start, end) for start, end in segments]


def _direction_towards_the_board(to_face_on, line, corner, flat_corner, board_is_to_the_right: bool):
    """Unit vector along `line` that heads towards the board once it is face-on, or None."""
    direction = line_direction(line)
    stepped = _apply(to_face_on, corner + 20 * direction)
    if stepped is None:
        return None
    moves_right = stepped[0] > flat_corner[0]
    return direction if moves_right == board_is_to_the_right else -direction


def _board_corners(to_face_on, flat_a, flat_b, board_is_to_the_right: bool, width: int, height: int):
    """
    The board as four image corners: between the two converging edges, from the
    side edge as far across as the photo shows. None if that isn't a believable board.
    """
    top, bottom = sorted((flat_a[1], flat_b[1]))
    side_x = (flat_a[0] + flat_b[0]) / 2
    reach = _how_far_the_photo_reaches(to_face_on, width, height, top, bottom)
    if reach is None:
        return None
    far_x = reach[1] if board_is_to_the_right else reach[0]
    left, right = sorted((side_x, far_x))
    if bottom - top <= 0 or not 0.3 < (right - left) / (bottom - top) < 6:
        return None

    face_on_corners = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], np.float32)
    corners = cv2.perspectiveTransform(face_on_corners.reshape(-1, 1, 2), np.linalg.inv(to_face_on)).reshape(4, 2)
    if not np.all(np.isfinite(corners)) or np.abs(corners).max() > 20 * max(width, height):
        return None
    return corners


def _how_far_the_photo_reaches(to_face_on, width: int, height: int, top: float, bottom: float):
    """
    Once face-on, the leftmost and rightmost points of the photo between the
    `top` and `bottom` rows, as (left, right). None if the photo reaches the
    horizon of the board's plane (the view is too extreme to use).
    """
    photo = [(0, 0), (width, 0), (width, height), (0, height)]
    mapped = []
    for x, y in photo:
        flat_x, flat_y, w = to_face_on @ np.array([x, y, 1.0])
        if w <= 1e-9:
            return None
        mapped.append((flat_x / w, flat_y / w))

    xs = []
    for (x1, y1), (x2, y2) in zip(mapped, mapped[1:] + mapped[:1]):
        for row in (top, bottom):
            if (y1 - row) * (y2 - row) < 0:                       # this side of the photo crosses the row
                xs.append(x1 + (row - y1) / (y2 - y1) * (x2 - x1))
        if top <= y1 <= bottom:                                   # a photo corner inside the strip
            xs.append(x1)
    return (min(xs), max(xs)) if xs else None

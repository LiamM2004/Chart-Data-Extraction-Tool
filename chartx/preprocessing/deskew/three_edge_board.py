"""Find the board from three of its edges, when its corners aren't all in shot.

The top edge, bottom edge and one side edge are enough to straighten the visible part of the
board. Two methods build the missing side, chosen by how strongly the top and bottom edges
converge: "partial" (under 10 degrees; the sides are taken as parallel) and "steep" (10 degrees
or more; uses the assumed camera). See the README, "How straightening works".

    corners, confidence, source = find_three_edge_board(photo)    # source: "partial" or "steep"
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import cv2
import numpy as np

from chartx.preprocessing.deskew.geometry import (
    ASSUMED_FOCAL_FRAC, as_homogeneous_line, best_candidate, camera_matrix, edge_support_score,
    find_edge_lines, find_edge_maps, line_direction, order_corners, part_of_segment_in_photo,
    share_of_segment_on_edges, shrink_to_width, where_lines_cross, where_ray_leaves_photo)

# The top and bottom edges must converge by this much (degrees) for the camera-based method;
# nearly parallel edges say too little about the board's tilt.
MIN_CONVERGENCE_DEG = 10.0

# How well the three visible edges must sit on real image edges (0-1) for each method to be trusted.
# The parallel-sides guess is the less certain method, so it asks for more.
MIN_CONFIDENCE = {"partial": 0.8, "steep": 0.7}

PARTIAL_MIN_VISIBLE_FRAC = 0.25    # each horizontal edge must show at least this share of the photo's width
STEEP_MIN_VISIBLE_FRAC = 0.08      # each of the three edges must show at least this share of the photo's diagonal
MAX_BOARD_TILT_DEG = 75.0          # a board turned further away than this is too edge-on to trust


@dataclass(frozen=True)
class _Scene:
    """What every candidate board is judged against."""
    thick_edges: np.ndarray
    width: int
    height: int
    camera: np.ndarray


def find_three_edge_board(img: np.ndarray,
                          focal_frac: float = ASSUMED_FOCAL_FRAC,
                          work_width: int = 1000) -> tuple[np.ndarray | None, float, str | None]:
    """Find a board from two roughly horizontal edges (top and bottom) and one side edge.

    Every combination is tried, with the board on either side of the side edge, and scored by the
    edge support along the visible part of each edge. Only candidates that are trustworthy for
    their method (see MIN_CONFIDENCE) count.

    Returns:
        (corners, confidence, source): corners ordered TL, TR, BR, BL (possibly off the photo), the
        weakest edge's support (0-1), and "partial" or "steep". (None, 0.0, None) if nothing
        trustworthy was found.
    """
    small, scale = shrink_to_width(img, work_width)
    edges, thick_edges = find_edge_maps(small)
    height, width = edges.shape
    scene = _Scene(thick_edges, width, height, camera_matrix(width, height, focal_frac))
    horizontals = find_edge_lines(edges, vertical=False, k=12, max_dev=55)    # steep views tilt these a lot
    verticals = find_edge_lines(edges, vertical=True, k=12, max_dev=35)

    best = best_candidate(_trusted_candidates(horizontals, verticals, scene))
    if best is None:
        return None, 0.0, None
    _, confidence, corners, source = best
    return order_corners(corners / scale), confidence, source


def _trusted_candidates(horizontals, verticals, scene: _Scene):
    """Candidate boards from every (two horizontals, side edge) triple, on either side of the side edge.

    Yields (score, confidence, corners, source), keeping only those trustworthy for their method.
    """
    for line_a, line_b in itertools.combinations(horizontals, 2):
        source = "steep" if _angle_between(line_a, line_b) >= MIN_CONVERGENCE_DEG else "partial"
        try_board = _steep_candidate if source == "steep" else _partial_candidate
        for side_line in verticals:
            for board_is_to_the_right in (True, False):
                found = try_board(line_a, line_b, side_line, board_is_to_the_right, scene)
                if found is not None and found[1] >= MIN_CONFIDENCE[source]:
                    yield (*found, source)


def _angle_between(line_a, line_b) -> float:
    return float(np.degrees(abs(line_a[1] - line_b[1])))


def _partial_candidate(top, bottom, side_line, board_is_to_the_right: bool, scene: _Scene):
    """
    Try the board lying to one side of the visible side edge, with its far side built parallel to it.
    Returns (score, confidence, quad), or None if that doesn't work out.
    """
    top_corner, bottom_corner = where_lines_cross(top, side_line), where_lines_cross(bottom, side_line)
    if not _is_a_usable_visible_edge(top_corner, bottom_corner, scene):
        return None

    side = 1 if board_is_to_the_right else -1
    exits = [where_ray_leaves_photo(corner, _direction_towards(line, side), scene.width, scene.height)
             for line, corner in ((top, top_corner), (bottom, bottom_corner))]
    if any(e is None for e in exits):
        return None
    top_exit, bottom_exit = exits
    visible = min(np.linalg.norm(top_exit - top_corner), np.linalg.norm(bottom_exit - bottom_corner))
    if visible < PARTIAL_MIN_VISIBLE_FRAC * scene.width:
        return None

    support = [share_of_segment_on_edges(scene.thick_edges, top_corner, top_exit),
               share_of_segment_on_edges(scene.thick_edges, bottom_corner, bottom_exit),
               share_of_segment_on_edges(scene.thick_edges, top_corner, bottom_corner)]
    up = (bottom_corner - top_corner) / np.linalg.norm(bottom_corner - top_corner)
    quad = _close_the_missing_side(top, bottom, top_corner, bottom_corner, exits, up, scene)
    if quad is None:
        return None
    confidence, score = edge_support_score(support)
    return score, confidence, quad


def _is_a_usable_visible_edge(top_corner, bottom_corner, scene: _Scene) -> bool:
    """The visible side edge must meet both horizontals inside the photo and be reasonably tall."""
    if top_corner is None or bottom_corner is None:
        return False
    if not (_is_in_photo(top_corner, scene) and _is_in_photo(bottom_corner, scene)):
        return False
    return np.linalg.norm(top_corner - bottom_corner) >= 0.2 * scene.height


def _close_the_missing_side(top, bottom, top_corner, bottom_corner, exits, up, scene: _Scene):
    """
    Build the missing side: parallel to the visible side edge, through whichever
    exit point keeps all four corners in the photo (larger quad wins).
    """
    options = []
    for exit_point in exits:
        parallel_point = exit_point + up
        ends = [_where_line_meets_line_through(line, exit_point, parallel_point) for line in (top, bottom)]
        if any(end is None for end in ends):
            continue
        quad = np.array([top_corner, ends[0], ends[1], bottom_corner], np.float32)
        all_in_photo = all(_is_in_photo(corner, scene) for corner in quad)
        options.append((all_in_photo, cv2.contourArea(quad), quad))
    if not options:
        return None
    preferred = [option for option in options if option[0]] or options
    return max(preferred, key=lambda option: option[1])[2]


def _is_in_photo(point, scene: _Scene, margin: float = 0.03) -> bool:
    return (-margin * scene.width <= point[0] <= (1 + margin) * scene.width
            and -margin * scene.height <= point[1] <= (1 + margin) * scene.height)


def _direction_towards(line, side: int) -> np.ndarray:
    """Unit vector along a (rho, theta) line, pointing right (side=1) or left (side=-1)."""
    d = line_direction(line)
    return d if d[0] * side > 0 else -d


def _where_line_meets_line_through(line, p: np.ndarray, q: np.ndarray) -> np.ndarray | None:
    """Where a (rho, theta) line meets the line through points p and q, or None if they are parallel."""
    through_pq = np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0])
    meeting = np.cross(as_homogeneous_line(line), through_pq)
    return meeting[:2] / meeting[2] if abs(meeting[2]) > 1e-9 else None


def _steep_candidate(line_a, line_b, side_line, board_is_to_the_right: bool, scene: _Scene):
    """
    Try the board lying to one side of the side edge, turned face-on to find its corners.
    Returns (score, confidence, corners), or None if that doesn't work out.
    """
    to_face_on = _face_on_homography(scene.camera, line_a, line_b, side_line)
    if to_face_on is None:
        return None
    corner_a, corner_b = where_lines_cross(line_a, side_line), where_lines_cross(line_b, side_line)
    if corner_a is None or corner_b is None:
        return None
    flat_a, flat_b = _apply(to_face_on, corner_a), _apply(to_face_on, corner_b)
    if flat_a is None or flat_b is None:
        return None

    support = _support_along_the_three_edges(to_face_on, (line_a, corner_a, flat_a), (line_b, corner_b, flat_b),
                                             board_is_to_the_right, scene)
    if support is None:
        return None
    corners = _board_corners(to_face_on, flat_a, flat_b, board_is_to_the_right, scene)
    if corners is None:
        return None
    confidence, score = edge_support_score(support)
    return score, confidence, corners


def _face_on_homography(camera: np.ndarray, line_a, line_b, side_line) -> np.ndarray | None:
    """
    The 3x3 matrix that would show the board face-on, worked out from its two
    converging edges and its side edge. None if the three don't make sense.
    """
    to_directions = np.linalg.inv(camera)
    converge_at = np.cross(as_homogeneous_line(line_a), as_homogeneous_line(line_b))
    along = to_directions @ converge_at
    if np.linalg.norm(along) < 1e-9:
        return None
    along = _pointing(along / np.linalg.norm(along), axis=0)                  # towards the right

    side_plane_normal = camera.T @ as_homogeneous_line(side_line)
    up_down = np.cross(side_plane_normal, along)                              # in that plane, and square to "along"
    if np.linalg.norm(up_down) < 1e-9:
        return None
    up_down = _pointing(up_down / np.linalg.norm(up_down), axis=1)            # towards the bottom

    board_normal = np.cross(along, up_down)
    if board_normal[2] < np.cos(np.radians(MAX_BOARD_TILT_DEG)):                    # turned away too far (or mirrored)
        return None
    return camera @ np.vstack([along, up_down, board_normal]) @ to_directions


def _pointing(direction: np.ndarray, axis: int) -> np.ndarray:
    """`direction` or its opposite, whichever points the positive way along `axis`."""
    return direction if direction[axis] >= 0 else -direction


def _apply(matrix: np.ndarray, point) -> np.ndarray | None:
    """Move a point by a 3x3 matrix, or None if it ends up at infinity."""
    x, y, w = matrix @ np.array([point[0], point[1], 1.0])
    return None if abs(w) < 1e-9 else np.array([x / w, y / w])


def _support_along_the_three_edges(to_face_on, edge_a, edge_b, board_is_to_the_right: bool, scene: _Scene):
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
        far = corner + along * 10 * max(scene.width, scene.height)
        segments.append(part_of_segment_in_photo(corner, far, scene.width, scene.height))
    segments.append(part_of_segment_in_photo(edge_a[1], edge_b[1], scene.width, scene.height))

    if any(segment is None for segment in segments):
        return None
    shortest_allowed = STEEP_MIN_VISIBLE_FRAC * np.hypot(scene.width, scene.height)
    if any(np.linalg.norm(end - start) < shortest_allowed for start, end in segments):
        return None
    return [share_of_segment_on_edges(scene.thick_edges, start, end) for start, end in segments]


def _direction_towards_the_board(to_face_on, line, corner, flat_corner, board_is_to_the_right: bool):
    """Unit vector along `line` that heads towards the board once it is face-on, or None."""
    direction = line_direction(line)
    stepped = _apply(to_face_on, corner + 20 * direction)
    if stepped is None:
        return None
    moves_right = stepped[0] > flat_corner[0]
    return direction if moves_right == board_is_to_the_right else -direction


def _board_corners(to_face_on, flat_a, flat_b, board_is_to_the_right: bool, scene: _Scene):
    """
    The board as four image corners: between the two converging edges, from the
    side edge as far across as the photo shows. None if that isn't a believable board.
    """
    top, bottom = sorted((flat_a[1], flat_b[1]))
    side_x = (flat_a[0] + flat_b[0]) / 2
    reach = _how_far_the_photo_reaches(to_face_on, scene, top, bottom)
    if reach is None:
        return None
    far_x = reach[1] if board_is_to_the_right else reach[0]
    left, right = sorted((side_x, far_x))
    if bottom - top <= 0 or not 0.3 < (right - left) / (bottom - top) < 6:
        return None

    face_on_corners = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], np.float32)
    corners = cv2.perspectiveTransform(face_on_corners.reshape(-1, 1, 2), np.linalg.inv(to_face_on)).reshape(4, 2)
    if not np.all(np.isfinite(corners)) or np.abs(corners).max() > 20 * max(scene.width, scene.height):
        return None
    return corners


def _how_far_the_photo_reaches(to_face_on, scene: _Scene, top: float, bottom: float):
    """
    Once face-on, the leftmost and rightmost points of the photo between the
    `top` and `bottom` rows, as (left, right). None if the photo reaches the
    horizon of the board's plane (the view is too extreme to use).
    """
    photo = [(0, 0), (scene.width, 0), (scene.width, scene.height), (0, scene.height)]
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

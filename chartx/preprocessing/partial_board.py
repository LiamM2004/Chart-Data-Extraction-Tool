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

from chartx.preprocessing.geometry import edge_coverage, hough_lines, intersect, order_corners


def _exit_point(start: np.ndarray, direction: np.ndarray, w: int, h: int) -> np.ndarray | None:
    """Where the ray from `start` along `direction` leaves the w x h photo."""
    far = start + direction * 10 * max(w, h)
    ok, _, end = cv2.clipLine((0, 0, w - 1, h - 1), tuple(int(v) for v in start),
                              tuple(int(v) for v in far))
    return np.array(end, float) if ok else None


def find_partial_board_corners(img: np.ndarray,
                               work_width: int = 1000,
                               min_visible_frac: float = 0.25
                               ) -> tuple[np.ndarray | None, float]:
    """
    Look for a board seen with one vertical side out of shot.

    Tries every (top, bottom, vertical) triple of strong Hough lines, on either
    side of the vertical one, and scores it by edge support along the three
    visible edges. Returns (corners, confidence), with corners ordered TL, TR,
    BR, BL in image coordinates and confidence the weakest visible edge's
    support (0-1), or (None, 0.0). The quad only covers the visible part of the
    board; its out-of-shot side lies along the photo's border.
    """
    h, w = img.shape[:2]
    scale = work_width / w if w > work_width else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else img
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    thick = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    H, W = edges.shape

    def in_photo(p, margin: float = 0.03) -> bool:
        return -margin * W <= p[0] <= (1 + margin) * W and -margin * H <= p[1] <= (1 + margin) * H

    best, best_score, best_conf = None, 0.0, 0.0
    horiz = hough_lines(edges, vertical=False, k=12)
    vert = hough_lines(edges, vertical=True, k=12)
    for top, bottom in itertools.combinations(horiz, 2):
        for side_line in vert:
            a, b = intersect(top, side_line), intersect(bottom, side_line)
            if a is None or b is None or not (in_photo(a) and in_photo(b)):
                continue
            if np.linalg.norm(a - b) < 0.2 * H:
                continue
            up = (b - a) / np.linalg.norm(b - a)              # direction of the visible vertical edge
            for side in (1, -1):                              # is the board to the right or the left?
                exits = []
                for line, start in ((top, a), (bottom, b)):
                    d = np.array([-np.sin(line[1]), np.cos(line[1])])
                    exits.append(_exit_point(start, d if d[0] * side > 0 else -d, W, H))
                if any(e is None for e in exits):
                    continue
                if min(np.linalg.norm(exits[0] - a), np.linalg.norm(exits[1] - b)) < min_visible_frac * W:
                    continue
                cov = [edge_coverage(thick, a, exits[0]), edge_coverage(thick, b, exits[1]),
                       edge_coverage(thick, a, b)]
                score = min(cov) * np.mean(cov)
                if score <= best_score:
                    continue
                # The missing side: parallel to the visible one, through whichever
                # exit point keeps all four corners in the photo (larger quad wins).
                quads = []
                for e in exits:
                    far = e + up
                    ends = []
                    for line in (top, bottom):
                        p = np.cross([e[0], e[1], 1.0], [far[0], far[1], 1.0])
                        q = np.cross(np.array([np.cos(line[1]), np.sin(line[1]), -line[0]]), p)
                        ends.append(q[:2] / q[2] if abs(q[2]) > 1e-9 else None)
                    if any(p is None for p in ends):
                        continue
                    quad = np.array([a, ends[0], ends[1], b], np.float32)
                    quads.append((all(in_photo(p) for p in quad), cv2.contourArea(quad), quad))
                if not quads:
                    continue
                inside = [q for q in quads if q[0]] or quads
                best = max(inside, key=lambda q: q[1])[2]
                best_score, best_conf = score, min(cov)

    if best is None:
        return None, 0.0
    return order_corners(best / scale), best_conf

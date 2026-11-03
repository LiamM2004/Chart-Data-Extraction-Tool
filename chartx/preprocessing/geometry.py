"""
geometry.py - small line/corner helpers shared by the board-finding modules.

Requires: opencv-python, numpy
"""

from __future__ import annotations

import cv2
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()          # y - x
    return np.array([
        pts[np.argmin(s)],                    # TL: smallest x+y
        pts[np.argmin(d)],                    # TR: smallest y-x
        pts[np.argmax(s)],                    # BR: largest x+y
        pts[np.argmax(d)],                    # BL: largest y-x
    ], dtype=np.float32)



def hough_lines(edges: np.ndarray, vertical: bool, k: int = 8,
                 max_dev: float = 35.0) -> list[tuple[float, float]]:
    """Strongest near-vertical (or near-horizontal) lines as (rho, theta), merged."""
    lines = cv2.HoughLines(edges, 1, np.pi / 360, int(0.25 * min(edges.shape)))
    out: list[tuple[float, float]] = []
    for rho, theta in ([] if lines is None else lines[:, 0]):
        # theta is the normal's angle: ~0 for vertical lines, ~90 deg for horizontal.
        dev = np.degrees(min(theta, np.pi - theta) if vertical else abs(theta - np.pi / 2))
        if dev > max_dev:
            continue
        if any(abs(rho - r) < 12 and abs(theta - t) < np.radians(3) for r, t in out):
            continue                              # near-duplicate of a stronger line
        out.append((float(rho), float(theta)))
        if len(out) >= k:
            break
    return out



def intersect(l1, l2) -> np.ndarray | None:
    A = np.array([[np.cos(l1[1]), np.sin(l1[1])], [np.cos(l2[1]), np.sin(l2[1])]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, [l1[0], l2[0]])



def edge_coverage(edges: np.ndarray, p: np.ndarray, q: np.ndarray, n: int = 150) -> float:
    """Fraction of the segment p-q that lies on an edge pixel (off-image counts as a miss)."""
    t = np.linspace(0, 1, n)
    xs = (p[0] + (q[0] - p[0]) * t).round().astype(int)
    ys = (p[1] + (q[1] - p[1]) * t).round().astype(int)
    inside = (xs >= 0) & (xs < edges.shape[1]) & (ys >= 0) & (ys < edges.shape[0])
    hit = np.zeros(n, bool)
    hit[inside] = edges[ys[inside], xs[inside]] > 0
    return float(hit.mean())

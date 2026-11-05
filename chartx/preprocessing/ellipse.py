"""
ellipse.py - refine perspective using a pie chart's outline.

Under perspective a circle becomes an ellipse, so a pie's outline says how
much stretch is left in the image. This runs *after* the board warp, when the
pie is already close to round, so we can be strict about what counts as one:
anything that isn't a nearly-complete, nearly-round outline is ignored.

    ellipse = find_pie_ellipse(img)       # None if no convincing pie
    img = circularise(img, ellipse)       # stretch the short axis to match

Requires: opencv-python, numpy
"""

from __future__ import annotations

import cv2
import numpy as np

WORK_WIDTH = 1200        # long side of the working copy, px
TOL = 2.5                # px (at working size) for a point to lie on the outline


def black_ink_contours(img: np.ndarray) -> tuple[list[np.ndarray], float]:
    """
    Contours of dark, unsaturated marker ink (the outline and text, not the
    coloured fills), on a copy scaled so its long side is WORK_WIDTH.
    Returns (contours, scale used).
    """
    scale = WORK_WIDTH / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    local = cv2.blur(gray, (51, 51))
    mask = ((gray.astype(int) < local.astype(int) - 35) & (sat < 110)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours, scale


def _distance(pts: np.ndarray, e) -> np.ndarray:
    """Approximate distance in px from each point to the ellipse `e` (cv2 format)."""
    (cx, cy), (d0, d1), ang = e
    a, b = d0 / 2, d1 / 2
    t = np.radians(ang)
    d = pts - [cx, cy]
    x = d[:, 0] * np.cos(t) + d[:, 1] * np.sin(t)
    y = -d[:, 0] * np.sin(t) + d[:, 1] * np.cos(t)
    return np.abs(np.hypot(x / a, y / b) - 1) * min(a, b)


def _coverage(pts: np.ndarray, e, tol: float, bins: int = 36) -> float:
    """Fraction of the outline's angular range with at least one supporting point."""
    near = pts[_distance(pts, e) < tol]
    if len(near) == 0:
        return 0.0
    ang = np.arctan2(near[:, 1] - e[0][1], near[:, 0] - e[0][0])
    return len(set(((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins)) / bins


def find_pie_ellipse(img: np.ndarray,
                     min_axis_ratio: float = 0.8,
                     min_coverage: float = 0.8,
                     min_radius_frac: float = 0.03,
                     max_radius_frac: float = 0.3,
                     iterations: int = 250):
    """
    Look for a pie's outline: a near-round, near-complete dark ellipse.

    Each large dark contour gets a RANSAC ellipse fit; the fit is then grown
    using supporting points from every contour (gaps split an outline into
    pieces) and refitted. Candidates must be nearly round (`min_axis_ratio`,
    short/long axis), cover `min_coverage` of the perimeter and be a plausible
    size, so a frame, a digit or a partial arc doesn't pass.

    Returns the ellipse in OpenCV's ((cx, cy), (d0, d1), angle) form, in the
    coordinates of `img`, or None.
    """
    contours, scale = black_ink_contours(img)
    long_side = WORK_WIDTH
    big = [c.reshape(-1, 2).astype(np.float32) for c in contours if len(c) >= 30]
    if not big:
        return None
    everything = np.vstack(big)
    rng = np.random.default_rng(0)

    def plausible(e) -> bool:
        a, b = max(e[1]) / 2, min(e[1]) / 2
        return (b >= min_radius_frac * long_side and a <= max_radius_frac * long_side
                and b / a >= min_axis_ratio)

    best, best_score = None, 0.0
    with np.errstate(all="ignore"):
        for pts in big:
            if len(pts) < 120 or max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) < 0.08 * long_side:
                continue
            seed, seed_n = None, 0
            for _ in range(iterations):
                try:
                    e = cv2.fitEllipse(pts[rng.choice(len(pts), 6, replace=False)])
                except cv2.error:
                    continue
                if not plausible(e):
                    continue
                n = int((_distance(pts, e) < TOL).sum())
                if n > seed_n:
                    seed, seed_n = e, n
            if seed is None:
                continue
            e = seed
            for _ in range(4):                       # grow using every contour
                near = everything[_distance(everything, e) < TOL * 1.4]
                if len(near) < 20:
                    break
                e = cv2.fitEllipse(near)
            if not plausible(e):
                continue
            cov = _coverage(everything, e, TOL * 1.4)
            support = int((_distance(everything, e) < TOL * 1.4).sum())
            score = cov * np.sqrt(support)
            if cov >= min_coverage and score > best_score:
                best, best_score = e, score

    if best is None:
        return None
    (cx, cy), (d0, d1), ang = best
    return ((cx / scale, cy / scale), (d0 / scale, d1 / scale), ang)


def circularise_transform(ellipse, shape: tuple) -> tuple[np.ndarray, tuple[int, int], float]:
    """
    3x3 matrix that stretches an image of `shape` along the ellipse's short axis
    so the ellipse becomes a circle (the long axis is kept, so nothing is
    downsampled). Returns (matrix, output size (w, h), stretch factor = long
    axis / short axis). The matrix shifts the result so nothing is cut off.
    """
    (cx, cy), (d0, d1), ang = ellipse
    target = max(d0, d1)
    t = np.radians(ang)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    L = R @ np.diag([target / d0, target / d1]) @ R.T    # linear part
    h, w = shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) @ L.T
    lo = corners.min(axis=0)
    M = np.eye(3)
    M[:2, :2], M[:2, 2] = L, -lo
    size = tuple(int(np.ceil(v)) for v in (corners.max(axis=0) - lo))
    return M, size, float(target / min(d0, d1))


def circularise(img: np.ndarray, ellipse) -> tuple[np.ndarray, float]:
    """Apply circularise_transform to `img`. Returns (image, stretch factor)."""
    M, size, stretch = circularise_transform(ellipse, img.shape)
    out = cv2.warpAffine(img, M[:2], size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return out, stretch

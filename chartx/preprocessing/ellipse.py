"""
ellipse.py - refine perspective using a pie chart's outline.

Under perspective a circle becomes an ellipse, so a pie's outline says how
much stretch is left in the image. This runs *after* the board warp, when the
pie is already close to round, so we can be strict about what counts as one:
anything that isn't a nearly-complete, nearly-round outline is ignored.

    pie = find_pie_ellipse(img)                                   # None if no convincing pie
    matrix, size, stretch = pie_to_circle_transform(pie, img.shape)   # stretch the short axis to match

Requires: opencv-python, numpy
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_WIDTH = 1200        # long side of the working copy, px
TOL = 2.5                # px (at working size) for a point to lie on the outline
GROW_TOL = TOL * 1.4     # looser, for gathering the rest of the outline once we have a good guess


# --------------------------------------------------------------------------- #
# Finding the pie
# --------------------------------------------------------------------------- #
def find_pie_ellipse(img: np.ndarray,
                     min_axis_ratio: float = 0.8,
                     min_coverage: float = 0.7,
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
    outlines = [c.reshape(-1, 2).astype(np.float32) for c in contours if len(c) >= 30]
    if not outlines:
        return None
    all_ink = np.vstack(outlines)
    rules = _PieRules(min_axis_ratio, min_radius_frac * WORK_WIDTH, max_radius_frac * WORK_WIDTH)
    rng = np.random.default_rng(0)

    best, best_score = None, 0.0
    with np.errstate(all="ignore"):
        for outline in outlines:
            if not _is_big_enough_to_be_a_pie(outline):
                continue
            first_guess = _best_ellipse_through_random_points(outline, rules, rng, iterations)
            if first_guess is None:
                continue
            ellipse = _refit_using_all_ink(first_guess, all_ink)
            if not rules.accepts(ellipse):
                continue
            coverage = _share_of_outline_covered(all_ink, ellipse, GROW_TOL)
            support = int((_distance_to_ellipse(all_ink, ellipse) < GROW_TOL).sum())
            score = coverage * np.sqrt(support)
            if coverage >= min_coverage and score > best_score:
                best, best_score = ellipse, score

    if best is None:
        return None
    (cx, cy), (d0, d1), angle = best
    return ((cx / scale, cy / scale), (d0 / scale, d1 / scale), angle)


def black_ink_contours(img: np.ndarray) -> tuple[list[np.ndarray], float]:
    """
    Contours of dark, unsaturated marker ink (the outline and text, not the
    coloured fills), on a copy scaled so its long side is WORK_WIDTH.
    Returns (contours, scale used).
    """
    scale = WORK_WIDTH / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    saturation = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 1]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    local_brightness = cv2.blur(gray, (51, 51))
    is_dark_ink = (gray.astype(int) < local_brightness.astype(int) - 35) & (saturation < 110)
    contours, _ = cv2.findContours(is_dark_ink.astype(np.uint8) * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours, scale


@dataclass
class _PieRules:
    """What size and shape of ellipse could plausibly be a pie (sizes in working-copy px)."""
    min_axis_ratio: float
    min_radius: float
    max_radius: float

    def accepts(self, ellipse) -> bool:
        long_radius, short_radius = max(ellipse[1]) / 2, min(ellipse[1]) / 2
        return (short_radius >= self.min_radius and long_radius <= self.max_radius
                and short_radius / long_radius >= self.min_axis_ratio)


def _is_big_enough_to_be_a_pie(outline: np.ndarray) -> bool:
    """Skip specks and short strokes: a pie's outline is long and spans a good part of the photo."""
    return len(outline) >= 120 and max(np.ptp(outline[:, 0]), np.ptp(outline[:, 1])) >= 0.08 * WORK_WIDTH


def _best_ellipse_through_random_points(outline: np.ndarray, rules: _PieRules, rng, iterations: int):
    """RANSAC: fit ellipses to random handfuls of points and keep the one most of the outline lies on."""
    best, best_support = None, 0
    for _ in range(iterations):
        try:
            ellipse = cv2.fitEllipse(outline[rng.choice(len(outline), 6, replace=False)])
        except cv2.error:
            continue
        if not rules.accepts(ellipse):
            continue
        support = int((_distance_to_ellipse(outline, ellipse) < TOL).sum())
        if support > best_support:
            best, best_support = ellipse, support
    return best


def _refit_using_all_ink(ellipse, all_ink: np.ndarray, rounds: int = 4):
    """Gather every ink point near the ellipse (from any contour) and refit, so a gappy outline still counts as one."""
    for _ in range(rounds):
        near = all_ink[_distance_to_ellipse(all_ink, ellipse) < GROW_TOL]
        if len(near) < 20:
            break
        ellipse = cv2.fitEllipse(near)
    return ellipse


def _distance_to_ellipse(pts: np.ndarray, ellipse) -> np.ndarray:
    """Approximate distance in px from each point to the ellipse (OpenCV format)."""
    (cx, cy), (d0, d1), angle = ellipse
    a, b = d0 / 2, d1 / 2
    t = np.radians(angle)
    d = pts - [cx, cy]
    x = d[:, 0] * np.cos(t) + d[:, 1] * np.sin(t)
    y = -d[:, 0] * np.sin(t) + d[:, 1] * np.cos(t)
    return np.abs(np.hypot(x / a, y / b) - 1) * min(a, b)


def _share_of_outline_covered(pts: np.ndarray, ellipse, tol: float, bins: int = 36) -> float:
    """Fraction of the way round the outline that has at least one supporting point."""
    near = pts[_distance_to_ellipse(pts, ellipse) < tol]
    if len(near) == 0:
        return 0.0
    angle = np.arctan2(near[:, 1] - ellipse[0][1], near[:, 0] - ellipse[0][0])
    return len(set(((angle + np.pi) / (2 * np.pi) * bins).astype(int) % bins)) / bins


# --------------------------------------------------------------------------- #
# Making the pie round
# --------------------------------------------------------------------------- #
def pie_to_circle_transform(ellipse, shape: tuple) -> tuple[np.ndarray, tuple[int, int], float]:
    """
    3x3 matrix that stretches an image of `shape` along the ellipse's short axis
    so the ellipse becomes a circle (the long axis is kept, so nothing is
    downsampled). Returns (matrix, output size (w, h), stretch factor = long
    axis / short axis). The matrix shifts the result so nothing is cut off.
    """
    (cx, cy), (d0, d1), angle = ellipse
    target = max(d0, d1)
    t = np.radians(angle)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    L = R @ np.diag([target / d0, target / d1]) @ R.T    # linear part
    h, w = shape[:2]
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float) @ L.T
    lowest = corners.min(axis=0)
    M = np.eye(3)
    M[:2, :2], M[:2, 2] = L, -lowest
    size = tuple(int(np.ceil(v)) for v in (corners.max(axis=0) - lowest))
    return M, size, float(target / min(d0, d1))

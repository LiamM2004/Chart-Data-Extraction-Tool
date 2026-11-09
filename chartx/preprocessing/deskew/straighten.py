"""
straighten.py - straighten a photo of a hand-drawn whiteboard chart.

It works in three moves:

    1. Fix perspective  - find the board's four corners and warp them back to a
                          rectangle (if only part of the board is in shot, the
                          visible part is straightened; a board seen at a steep
                          angle is straightened from three of its edges). If the
                          board can't be found with confidence, the photo is left
                          alone and the caller should ask for corners (see
                          corner_editor.py).
    2. Round off pies   - a pie's outline should be a circle; stretch away any
                          squash the board warp left behind.
    3. Level the chart  - rotate so the chart's axes are horizontal and vertical
                          (a pie has no axes, so its writing is used instead).

Each move is worked out on a quick preview and recorded as a matrix. The
matrices are combined and the photo is resampled once, which keeps it sharper
than warping it three times.

    result = deskew(photo)            # result.image is the straightened photo

Requires: opencv-python, numpy
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from chartx.preprocessing.deskew.board import board_to_rectangle_transform, find_board_corners
from chartx.preprocessing.deskew.ellipse import find_pie_ellipse, pie_to_circle_transform
from chartx.preprocessing.deskew.geometry import ASSUMED_FOCAL_FRAC, is_a_usable_quad, order_corners
from chartx.preprocessing.deskew.rotation import (find_long_straight_lines, rotation_transform,
                                                  tilt_from_chart_axes, tilt_from_writing)
from chartx.preprocessing.deskew.three_edge_board import find_three_edge_board

# Tilts smaller than this (degrees) are not worth a resample.
MIN_ROTATION_DEG = 0.2

# A pie outline this much off round (long/short axis) is stretched back to a circle.
PIE_MIN_STRETCH = 1.03

# Auto-detected corners with less edge support than this need a human look.
# Chosen from a handful of photos (good ones scored 0.56-0.79, bad ones 0.11-0.13).
REVIEW_BELOW = 0.4


@dataclass
class DeskewResult:
    image: np.ndarray                       # final corrected image
    corners: np.ndarray | None              # board corners (TL, TR, BR, BL) in input coords; if
                                            # corner_source is "none", only a suggestion, not applied
    corner_source: str                      # "auto", "partial", "steep", "manual", "provided" or "none" (not warped)
    confidence: float                       # auto-detection edge support, 0-1 (1.0 if corners were given)
    needs_review: bool                      # True if a human should check/adjust the corners
    rotation_deg: float                     # rotation applied when levelling the chart
    warnings: list = field(default_factory=list)
    pie_stretch: float = 1.0                # stretch applied to make a pie round (1.0 = none)


# --------------------------------------------------------------------------- #
# The whole pipeline
# --------------------------------------------------------------------------- #
def deskew(img: np.ndarray, corners: np.ndarray | None = None,
           focal_frac: float = ASSUMED_FOCAL_FRAC) -> DeskewResult:
    """
    Straighten a photo of a whiteboard chart.

    corners - optional user-supplied board/chart corners (any order). Pass these
              from your UI when auto-detection fails.
    focal_frac - the lens's focal length as a multiple of the photo's long side. The default
              suits a phone's main camera; an ultra-wide or zoom lens needs its own value.

    If the whole board isn't in the photo but three of its edges are, only the
    visible part is straightened (corner_source "partial"), or, when the board is
    seen at a steep angle with its corners out of shot, it is straightened from
    those three edges (corner_source "steep").

    If auto-detection is not confident, `needs_review` is set, the photo is
    returned untouched and `corners` holds the best guess (or None). A guess we
    don't trust can warp the chart out of frame - a quad over the wall, say - so
    the caller should let the user adjust the corners (see
    select_corners_manually) and call again with `corners=`.
    """
    img = _as_bgr_photo(img)
    corners, source, confidence, warnings = _locate_board(img, corners, focal_frac)
    if corners is None or confidence < REVIEW_BELOW:
        warnings.append(f"Board corners not found with confidence ({confidence:.2f}) - "
                        "please check or drag the corners into place. Image left unchanged.")
        return DeskewResult(img, corners, "none", confidence, True, 0.0, warnings)

    correction = _PhotoCorrection(img, corners, focal_frac)
    pie_stretch = correction.round_off_the_pie()
    rotation = correction.level_the_chart()
    if rotation is None:
        warnings.append("Found neither chart axes nor a pie to level the image on - rotation not corrected.")

    return DeskewResult(correction.final_image(), corners, source, confidence, False,
                        0.0 if rotation is None else rotation, warnings, pie_stretch)


def _as_bgr_photo(img) -> np.ndarray:
    """
    The photo as an 8-bit, 3-channel (BGR) array, the way cv2.imread returns it. Grayscale is
    expanded and an alpha channel dropped; anything that can't be a photo is rejected clearly.
    """
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim not in (2, 3) or img.size == 0:
        found = f"{img.dtype} array of shape {img.shape}" if isinstance(img, np.ndarray) else type(img).__name__
        raise ValueError(f"deskew() needs a photo as a non-empty 8-bit numpy array (as cv2.imread returns), got {found}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.shape[2] != 3:
        raise ValueError(f"deskew() needs 1, 3 or 4 colour channels, got {img.shape[2]}")
    return img


def _locate_board(img: np.ndarray, given_corners: np.ndarray | None, focal_frac: float):
    """
    Decide where the board is. Returns (corners, source, confidence, warnings).
    Tries, in order: the corners we were given, a whole board found by its edges,
    then a board with its far side or corners out of shot, found from three of its
    edges. The corners returned may be a weak guess (or None); the caller checks
    the confidence.
    """
    if given_corners is not None:
        if not is_a_usable_quad(given_corners, img.shape):
            warning = ("The corners given don't make a usable four-sided shape: they need to be four different "
                       "points forming a convex shape that covers a reasonable part of the photo.")
            return None, "provided", 0.0, [warning]
        return order_corners(given_corners), "provided", 1.0, []

    corners, confidence = find_board_corners(img)
    if corners is not None and confidence >= REVIEW_BELOW:
        return corners, "auto", confidence, []

    edge_corners, edge_confidence, edge_source = find_three_edge_board(img, focal_frac)
    if edge_corners is not None:
        return edge_corners, edge_source, edge_confidence, [_three_edge_warning(edge_source, focal_frac)]
    return corners, "auto", confidence, []


def _three_edge_warning(source: str, focal_frac: float) -> str:
    """What the user should know about a board straightened from three of its edges."""
    if source == "partial":
        return ("Part of the board is out of shot: only the visible part was "
                "straightened (assumes its vertical edges are parallel).")
    camera = "a typical phone camera" if focal_frac == ASSUMED_FOCAL_FRAC else "the focal length you gave"
    return (f"Board corners are out of shot: straightened from three of its edges using {camera}, "
            "so the angle is an estimate.")


def _the_pie_after(stretch: np.ndarray, pie):
    """Where the pie ends up after the round-off stretch: a circle as wide as its long axis, at the moved centre."""
    (cx, cy), (d0, d1), _ = pie
    x, y, _ = stretch @ np.array([cx, cy, 1.0])
    diameter = max(d0, d1)
    return (x, y), (diameter, diameter), 0.0


class _PhotoCorrection:
    """
    A photo plus the single combined transform that straightens it, built up one
    step at a time. Each step is measured on a quick preview of the result so
    far; the full-quality image is only produced once, at the end.
    """

    def __init__(self, img: np.ndarray, board_corners: np.ndarray, focal_frac: float):
        self.img = img
        self.matrix, self.size = board_to_rectangle_transform(board_corners, img.shape, focal_frac)
        self.board_mask = self._mask_of_the_board()
        self._preview = None
        self._pie = None
        self._pie_searched = False

    # ---- the steps ------------------------------------------------------ #
    def round_off_the_pie(self) -> float:
        """If there is a pie that is still squashed, stretch it back to a circle. Returns the stretch (1.0 = none)."""
        pie = self._the_pie()
        if pie is None or max(pie[1]) / min(pie[1]) < PIE_MIN_STRETCH:
            return 1.0
        matrix, size, stretch = pie_to_circle_transform(pie, self.preview.shape)
        self._add_step(matrix, size)
        self._pie = _the_pie_after(matrix, pie)             # the stretch turned it into a circle, and moved it
        return stretch

    def level_the_chart(self) -> float | None:
        """
        Rotate so the chart is level. Returns the rotation applied in degrees
        (0.0 if already level), or None if there was nothing to level it on.
        """
        pie = self._the_pie()
        tilt = tilt_from_chart_axes(find_long_straight_lines(self.preview, ignoring_pie=pie))
        if tilt is None and pie is not None:
            # A pie has no axes, so level it on its writing instead. Only done when a pie is there,
            # so sideways text on a chart that merely missed its L can't drag the image round.
            tilt = tilt_from_writing(self.preview, pie)
        if tilt is None:
            return None
        if abs(tilt) < MIN_ROTATION_DEG:
            return 0.0
        matrix, size = rotation_transform(self.preview.shape, -tilt)
        self._add_step(matrix, size)
        return -tilt

    def final_image(self) -> np.ndarray:
        """The corrected photo, resampled once at full quality."""
        return self._render(cv2.INTER_CUBIC)

    # ---- helpers -------------------------------------------------------- #
    @property
    def preview(self) -> np.ndarray:
        """A quick, lower-quality render of the corrections so far (used to measure the next step)."""
        if self._preview is None:
            self._preview = self._render(cv2.INTER_LINEAR)
        return self._preview

    def _the_pie(self):
        """
        The pie, if there is one (in the preview's coordinates). It is looked for once and then
        carried through the steps, because detection is borderline on some photos: finding it for
        one step but not the next would leave a later step unable to ignore it.
        """
        if not self._pie_searched:
            self._pie, self._pie_searched = find_pie_ellipse(self.preview), True
        return self._pie

    def _add_step(self, matrix: np.ndarray, size: tuple[int, int]):
        self.matrix, self.size = matrix @ self.matrix, size
        self._preview = None

    def _mask_of_the_board(self) -> np.ndarray:
        """Which photo pixels are the (trimmed) board, as opposed to the wall around it."""
        w, h = self.size
        board = cv2.perspectiveTransform(
            np.array([[[0, 0], [w, 0], [w, h], [0, h]]], np.float32),
            np.linalg.inv(self.matrix))[0]
        mask = np.zeros(self.img.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, np.round(board).astype(np.int32), 255)
        return mask

    def _render(self, interpolation: int) -> np.ndarray:
        out = cv2.warpPerspective(self.img, self.matrix, self.size, flags=interpolation,
                                  borderMode=cv2.BORDER_REPLICATE)
        on_board = cv2.warpPerspective(self.board_mask, self.matrix, self.size, flags=cv2.INTER_NEAREST) > 0
        if not on_board.all():                                # corners opened up by a rotation
            out[~on_board] = np.median(out[on_board], axis=0)  # fill with the board colour, not the wall
        return out

"""
deskew.py - Step 1 of the chart extraction pipeline: straighten the photo.

Straightens a photo of a hand-drawn whiteboard chart, in three moves:

    1. Fix perspective  - find the board's four corners and warp them back to a
                          rectangle (if only part of the board is in shot, the
                          visible part is straightened; a board seen at a steep
                          angle is straightened from three of its edges). If the
                          board can't be found with confidence, the perspective is
                          left alone (the photo is only levelled, if it shows a
                          tilt) and the caller should ask for corners (see
                          corner_editor.py).
    2. Round off pies   - a pie's outline should be a circle; stretch away any
                          squash the board warp left behind.
    3. Level the chart  - rotate so the chart's axes are horizontal and vertical
                          (a pie has no axes, so its writing is used instead).

Each move is worked out on a quick preview and recorded as a matrix. The
matrices are combined and the photo is resampled once, which keeps it sharper
than warping it three times.

Usage (CLI):
    python -m chartx.preprocessing.deskew photo.jpg -o out.png              # auto; opens the drag window if unsure
    python -m chartx.preprocessing.deskew photo.jpg -o out.png --manual     # always open the drag window
    python -m chartx.preprocessing.deskew photo.jpg -o out.png --no-prompt  # never open a window (batch use)
    python -m chartx.preprocessing.deskew photo.jpg -o out.png --corners 10,20 900,15 920,700 5,690
    python -m chartx.preprocessing.deskew photo.jpg -o out.png --focal 0.5   # lens focal length as a multiple of the long side (default 0.72: a phone's main camera)
    python -m chartx.preprocessing.deskew photo.jpg -o out.png --debug debug.png

Requires: opencv-python, numpy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import cv2
import numpy as np

from chartx.preprocessing.board import board_to_rectangle_transform, find_board_corners
from chartx.preprocessing.corner_editor import select_corners_manually
from chartx.preprocessing.ellipse import find_pie_ellipse, pie_to_circle_transform
from chartx.preprocessing.geometry import ASSUMED_FOCAL_FRAC, order_corners
from chartx.preprocessing.partial_board import find_partial_board_corners
from chartx.preprocessing.rotation import (find_long_straight_lines, rotation_transform, tilt_from_chart_axes,
                                           tilt_from_writing)
from chartx.preprocessing.steep_board import find_steep_board_corners

# Tilts smaller than this (degrees) are not worth a resample.
MIN_ROTATION_DEG = 0.2

# A pie outline this much off round (long/short axis) is stretched back to a circle.
PIE_MIN_STRETCH = 1.03

# A board seen with one side out of shot is only trusted when its three visible
# edges are this well supported (the good cases scored 0.97+, a wall scored 0.56).
PARTIAL_MIN = 0.8

# A board straightened from three edges (steep angle, corners out of shot) needs this much
# support. Wrong guesses on the sample photos scored 0.43-0.55, right ones 0.78 and up.
STEEP_MIN = 0.7

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

    If auto-detection is not confident, `needs_review` is set and `corners` holds
    the best guess (or None). The perspective is not touched - a guess we don't
    trust can warp the chart out of frame, a quad over the wall, say - but the
    photo is levelled if its axes show a tilt. The caller should let the user
    adjust the corners (see select_corners_manually) and call again with `corners=`.
    """
    corners, source, confidence, warnings = _locate_board(img, corners, focal_frac)
    if corners is None or confidence < REVIEW_BELOW:
        return _without_a_board(img, corners, confidence, warnings, focal_frac)

    correction = _PhotoCorrection(img, corners, focal_frac)
    pie_stretch = correction.round_off_the_pie()
    rotation = correction.level_the_chart()
    if rotation is None:
        warnings.append("Found neither chart axes nor a pie to level the image on - rotation not corrected.")

    return DeskewResult(correction.final_image(), corners, source, confidence, False,
                        0.0 if rotation is None else rotation, warnings, pie_stretch)


def _without_a_board(img: np.ndarray, best_guess: np.ndarray | None, confidence: float,
                     warnings: list, focal_frac: float) -> DeskewResult:
    """
    No board we trust. We can't fix the perspective, but if the chart's axes (or a
    pie's writing) show the photo is tilted we can still level it - a close-up of
    just the chart is the usual case. Either way it is flagged for a person to check.
    """
    correction = _PhotoCorrection(img, None, focal_frac)
    rotation = correction.level_the_chart()
    levelled = bool(rotation)                          # None (nothing to go on) and 0.0 (already level) both mean no change
    what_happened = "Image only levelled (perspective not corrected)." if levelled else "Image left unchanged."
    warnings.append(f"Board corners not found with confidence ({confidence:.2f}) - "
                    f"please check or drag the corners into place. {what_happened}")
    return DeskewResult(correction.final_image() if levelled else img, best_guess, "none", confidence, True,
                        rotation if levelled else 0.0, warnings)


def _locate_board(img: np.ndarray, given_corners: np.ndarray | None, focal_frac: float):
    """
    Decide where the board is. Returns (corners, source, confidence, warnings).
    Tries, in order: the corners we were given, a whole board found by its edges,
    a board with one side out of shot, then a board at a steep angle with its
    corners out of shot. The corners returned may be a weak guess (or None); the
    caller checks the confidence.
    """
    if given_corners is not None:
        return order_corners(given_corners), "provided", 1.0, []

    corners, confidence = find_board_corners(img)
    if corners is not None and confidence >= REVIEW_BELOW:
        return corners, "auto", confidence, []

    partial_corners, partial_confidence = find_partial_board_corners(img)
    if partial_corners is not None and partial_confidence >= PARTIAL_MIN:
        warning = ("Part of the board is out of shot: only the visible part was "
                   "straightened (assumes its vertical edges are parallel).")
        return partial_corners, "partial", partial_confidence, [warning]

    steep_corners, steep_confidence = find_steep_board_corners(img, focal_frac)
    if steep_corners is not None and steep_confidence >= STEEP_MIN:
        camera = "a typical phone camera" if focal_frac == ASSUMED_FOCAL_FRAC else "the focal length you gave"
        warning = (f"Board corners are out of shot: straightened from three of its edges using {camera}, "
                   "so the angle is an estimate.")
        return steep_corners, "steep", steep_confidence, [warning]
    return corners, "auto", confidence, []


class _PhotoCorrection:
    """
    A photo plus the single combined transform that straightens it, built up one
    step at a time. Each step is measured on a quick preview of the result so
    far; the full-quality image is only produced once, at the end.
    """

    def __init__(self, img: np.ndarray, board_corners: np.ndarray | None, focal_frac: float):
        self.img = img
        if board_corners is None:                      # no board: the whole photo is all we have
            height, width = img.shape[:2]
            self.matrix, self.size = np.eye(3), (width, height)
            self.board_mask = np.full((height, width), 255, np.uint8)
        else:
            self.matrix, self.size = board_to_rectangle_transform(board_corners, img.shape, focal_frac)
            self.board_mask = self._mask_of_the_board()
        self._preview = None

    # ---- the steps ------------------------------------------------------ #
    def round_off_the_pie(self) -> float:
        """If there is a pie that is still squashed, stretch it back to a circle. Returns the stretch (1.0 = none)."""
        pie = find_pie_ellipse(self.preview)
        if pie is None or max(pie[1]) / min(pie[1]) < PIE_MIN_STRETCH:
            return 1.0
        matrix, size, stretch = pie_to_circle_transform(pie, self.preview.shape)
        self._add_step(matrix, size)
        return stretch

    def level_the_chart(self) -> float | None:
        """
        Rotate so the chart is level. Returns the rotation applied in degrees
        (0.0 if already level), or None if there was nothing to level it on.
        """
        tilt = tilt_from_chart_axes(find_long_straight_lines(self.preview))
        if tilt is None:
            tilt = self._tilt_of_a_pie_charts_writing()
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

    def _add_step(self, matrix: np.ndarray, size: tuple[int, int]):
        self.matrix, self.size = matrix @ self.matrix, size
        self._preview = None

    def _tilt_of_a_pie_charts_writing(self) -> float | None:
        """
        A pie has no axes, so level it on its writing instead. Only done when a pie
        is there, so sideways text on a chart that merely missed its L can't drag
        the image round.
        """
        pie = find_pie_ellipse(self.preview)
        return None if pie is None else tilt_from_writing(self.preview, pie)

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


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def debug_view(original: np.ndarray, result: DeskewResult) -> np.ndarray:
    """Side-by-side: input with the board corners drawn on | the straightened output."""
    before = original.copy()
    if result.corners is not None:
        cv2.polylines(before, [result.corners.astype(np.int32)], True, (0, 200, 0), 4)
    height = 600
    panels = [cv2.resize(p, (int(p.shape[1] * height / p.shape[0]), height)) for p in (before, result.image)]
    return np.hstack([panels[0], np.full((height, 10, 3), 255, np.uint8), panels[1]])


def main():
    args = _build_argument_parser().parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")

    focal_frac = args.focal or ASSUMED_FOCAL_FRAC
    result = deskew(img, corners=_corners_from_text(args.corners) if args.corners else None, focal_frac=focal_frac)

    # Ask the user to adjust the corners when asked to, or when auto-detection
    # wasn't confident. The window starts from the best guess, if there is one.
    if (args.manual or result.needs_review) and not args.no_prompt and not args.corners:
        picked = select_corners_manually(img, initial=result.corners)
        if picked is not None:
            result = deskew(img, corners=picked, focal_frac=focal_frac)
            result.corner_source = "manual"

    cv2.imwrite(args.output, result.image)
    print(f"corners: {result.corner_source}  rotation applied: {result.rotation_deg:+.2f} deg  "
          f"pie stretch: {result.pie_stretch:.2f}")
    for warning in result.warnings:
        print("warning:", warning)
    if args.debug:
        cv2.imwrite(args.debug, debug_view(img, result))


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Straighten a photo of a whiteboard chart.")
    parser.add_argument("image")
    parser.add_argument("-o", "--output", default="deskewed.png")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--manual", action="store_true",
                      help="always open the drag-corners window, even if auto-detection is confident")
    mode.add_argument("--no-prompt", action="store_true",
                      help="never open a window; just warn when corners need review (for batch runs)")
    parser.add_argument("--corners", nargs=4, metavar="X,Y",
                        help="board corners in any order, e.g. 10,20 900,15 920,700 5,690")
    parser.add_argument("--focal", type=float, metavar="FRAC",
                        help=f"lens focal length as a multiple of the photo's long side (default {ASSUMED_FOCAL_FRAC}, "
                             "a phone's main camera; an ultra-wide lens is about 0.4, a 2x zoom about 1.4)")
    parser.add_argument("--debug", help="save a side-by-side debug image here")
    return parser


def _corners_from_text(pairs) -> np.ndarray:
    """Turn ["10,20", "900,15", ...] from the command line into an array of points."""
    return np.array([[float(v) for v in pair.split(",")] for pair in pairs], dtype=np.float32)


if __name__ == "__main__":
    main()

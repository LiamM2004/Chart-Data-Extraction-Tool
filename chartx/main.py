"""
main.py - the command line: straighten a photo of a chart and clean up its glare.

    python -m chartx photo.jpg -o out/photo.png              # auto; opens the drag window if unsure
    python -m chartx photo.jpg -o out/photo.png --manual     # always open the drag window
    python -m chartx photo.jpg -o out/photo.png --no-prompt  # never open a window (batch use)
    python -m chartx photo.jpg -o out/photo.png --corners 10,20 900,15 920,700 5,690
    python -m chartx photo.jpg -o out/photo.png --focal 0.5  # lens focal length as a multiple of the long side
    python -m chartx photo.jpg -o out/photo.png --no-glare   # stop after straightening
    python -m chartx photo.jpg -o out/photo.png --debug out/debug.png

Requires: opencv-python, numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from chartx.pipeline import preprocess
from chartx.preprocessing.deskew import DeskewResult, select_corners_manually
from chartx.preprocessing.deskew.geometry import ASSUMED_FOCAL_FRAC


def main():
    parser = _build_argument_parser()
    args = parser.parse_args()
    if args.manual and args.corners:
        parser.error("--manual and --corners can't be used together (--corners already says where the board is)")
    photo = cv2.imread(args.image)
    if photo is None:
        raise SystemExit(f"Could not read {args.image}")

    ask = None if args.no_prompt else _ask_for_corners_unless_confident(always=args.manual)
    result = preprocess(photo,
                        corners=np.array(args.corners, np.float32) if args.corners else None,
                        focal_frac=args.focal if args.focal is not None else ASSUMED_FOCAL_FRAC,
                        ask_for_corners=ask,
                        clean_glare=not args.no_glare)

    _save_image(args.output, result.image)
    straightened = result.straightened
    print(f"corners: {straightened.corner_source}  rotation applied: {straightened.rotation_deg:+.2f} deg  "
          f"pie stretch: {straightened.pie_stretch:.2f}")
    for warning in straightened.warnings:
        print("warning:", warning)
    if args.debug:
        _save_image(args.debug, _debug_view(photo, straightened, result.image))


def _save_image(path: str, image: np.ndarray):
    """Write the image, creating its folder if needed, and stop with a clear message if that fails."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        saved = cv2.imwrite(path, image)
    except cv2.error:                                  # e.g. an extension OpenCV can't write
        saved = False
    if not saved:
        raise SystemExit(f"Could not save {path} (use a .png or .jpg file name, in a folder you can write to)")


def _ask_for_corners_unless_confident(always: bool):
    """A hook for the pipeline: open the drag window when asked to, or when the board wasn't found with confidence."""
    def ask(photo: np.ndarray, straightened: DeskewResult):
        if always or straightened.needs_review:
            # the window starts from our best guess, if there is one
            return select_corners_manually(photo, initial=straightened.corners)
        return None
    return ask


def _debug_view(original: np.ndarray, straightened: DeskewResult, final: np.ndarray) -> np.ndarray:
    """Side-by-side: the photo with the board corners drawn on | the final result."""
    before = original.copy()
    if straightened.corners is not None:
        cv2.polylines(before, [straightened.corners.astype(np.int32)], True, (0, 200, 0), 4)
    height = 600
    panels = [cv2.resize(p, (int(p.shape[1] * height / p.shape[0]), height)) for p in (before, final)]
    return np.hstack([panels[0], np.full((height, 10, 3), 255, np.uint8), panels[1]])


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Straighten a photo of a whiteboard chart and clean up its glare.")
    parser.add_argument("image")
    parser.add_argument("-o", "--output", default="out.png")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--manual", action="store_true",
                      help="always open the drag-corners window, even if auto-detection is confident")
    mode.add_argument("--no-prompt", action="store_true",
                      help="never open a window; just warn when corners need review (for batch runs)")
    parser.add_argument("--corners", nargs=4, metavar="X,Y", type=_point,
                        help="board corners in any order, e.g. 10,20 900,15 920,700 5,690")
    parser.add_argument("--focal", type=_positive_number, metavar="FRAC",
                        help=f"lens focal length as a multiple of the photo's long side (default {ASSUMED_FOCAL_FRAC}, "
                             "a phone's main camera; an ultra-wide lens is about 0.4, a 2x zoom about 1.4)")
    parser.add_argument("--no-glare", action="store_true", help="stop after straightening (skip the glare clean-up)")
    parser.add_argument("--debug", help="save a side-by-side debug image here")
    return parser


def _point(text: str) -> tuple[float, float]:
    """A corner given on the command line as "x,y"."""
    try:
        x, y = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a point; write it as x,y, for example 120,85")
    return x, y


def _positive_number(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        value = 0.0
    if not value > 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive number")
    return value


if __name__ == "__main__":
    main()

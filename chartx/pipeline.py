"""
pipeline.py - the processing steps, in order. Later stages chain on here.

    result = preprocess(photo)
    result.image          # straightened, then glare-cleaned
    result.straightened   # how the straightening went (corners, rotation, warnings, ...)

Requires: opencv-python, numpy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from chartx.preprocessing import glare
from chartx.preprocessing.deskew import DeskewResult, deskew
from chartx.preprocessing.deskew.geometry import ASSUMED_FOCAL_FRAC

# Called with (the photo, how the first straightening attempt went); returns the board's four
# corners picked by a person, or None to carry on with what we have.
AskForCorners = Callable[[np.ndarray, DeskewResult], "np.ndarray | None"]


@dataclass
class PreprocessResult:
    image: np.ndarray               # the straightened, glare-cleaned photo
    straightened: DeskewResult      # details of the straightening step


def preprocess(photo: np.ndarray,
               corners: np.ndarray | None = None,
               focal_frac: float = ASSUMED_FOCAL_FRAC,
               ask_for_corners: AskForCorners | None = None,
               clean_glare: bool = True) -> PreprocessResult:
    """
    Straighten the photo, then clean the glare off it.

    corners         - board corners if you already know them (skips finding them).
    focal_frac      - the lens's focal length as a multiple of the photo's long side
                      (see deskew); the default suits a phone's main camera.
    ask_for_corners - optional hook for a person (or a UI) to place the corners. It is
                      called after the first straightening attempt; if it returns
                      corners, straightening is redone with them. Without it, a photo
                      the tool can't place is flagged and left as it is.
    clean_glare     - set False to stop after straightening.
    """
    straightened = deskew(photo, corners=corners, focal_frac=focal_frac)
    if corners is None and ask_for_corners is not None:
        picked = ask_for_corners(photo, straightened)
        if picked is not None:
            straightened = deskew(photo, corners=picked, focal_frac=focal_frac)
            straightened.corner_source = "manual"

    image = glare.clean_whiteboard(straightened.image) if clean_glare else straightened.image
    return PreprocessResult(image, straightened)

"""The processing steps in order: straighten the photo, then clean up its glare.

    result = preprocess(photo)
    result.image          # the straightened, glare-cleaned photo
    result.straightened   # how the straightening went (corners, rotation, warnings)
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
    """Straighten the photo, then clean the glare off it.

    Args:
        photo: BGR photo, as returned by cv2.imread.
        corners: Board corners, if already known; skips looking for them.
        focal_frac: Lens focal length as a multiple of the photo's long side. The default
            suits a phone's main camera.
        ask_for_corners: Optional hook that lets a person or a UI place the corners. It is
            called after the first attempt with (photo, result); corners it returns trigger a
            second attempt. Without it, a photo that can't be placed is flagged and left as it is.
        clean_glare: Set False to stop after straightening.

    Returns:
        The final image, plus the details of the straightening step.
    """
    straightened = deskew(photo, corners=corners, focal_frac=focal_frac)
    if corners is None and ask_for_corners is not None:
        picked = ask_for_corners(photo, straightened)
        if picked is not None:
            straightened = deskew(photo, corners=picked, focal_frac=focal_frac)
            if straightened.corner_source == "provided":            # not "none": the corners may have been rejected
                straightened.corner_source = "manual"

    image = glare.clean_whiteboard(straightened.image) if clean_glare else straightened.image
    return PreprocessResult(image, straightened)

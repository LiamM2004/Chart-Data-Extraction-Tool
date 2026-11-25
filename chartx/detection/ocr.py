"""Reading the handwritten text on a chart, with PaddleOCR.

One call per label crop, never one for a whole strip of them: given a strip, PaddleOCR returns
the row as a single box ("1990 1995 2000 2005 ...") that cannot be split back into one text
per label.
"""
import functools

import cv2
import numpy as np


# Build PaddleOCR once: loading the models takes several seconds
@functools.lru_cache(maxsize=None)
def _engine():
    """PaddleOCR with the page-level correction switched off: orientation classifying and
    unwarping are meant for photographed documents, and turning a few characters only loses
    the reading. enable_mkldnn=False because paddlepaddle 3.3.1 crashes in its oneDNN CPU
    path on Linux x86 ("ConvertPirAttribute2RuntimeAttribute not support").
    """
    from paddleocr import PaddleOCR    # imported here, not at module load: it is slow to import
    return PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                     use_textline_orientation=False, enable_mkldnn=False)


# Read the text in one small crop
def read(crop: np.ndarray) -> str:
    """What one label crop says, with the spaces taken out, or "" if nothing could be read.

    A crop occasionally yields more than one box (a stray tick mark read as punctuation, say),
    so the most confident reading wins.
    """
    if crop.size == 0:
        return ""
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    page = _engine().predict(crop)[0]
    if not page["rec_texts"]:
        return ""
    text, _ = max(zip(page["rec_texts"], page["rec_scores"]), key=lambda found: found[1])
    return "".join(text.split())

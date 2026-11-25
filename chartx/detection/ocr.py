"""Reading the handwritten text on a chart, with PaddleOCR.

A whole image at a time keeps where each piece of text sits, which is how a label is matched
to the bar or sector it belongs to. One crop at a time is for a label that has already been
cut out: asked for a strip of several, PaddleOCR returns the row as a single box
("1990 1995 2000 ...") that cannot be split back up.
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


def read_page(image: np.ndarray) -> list[tuple[list, str, float]]:
    """Every piece of text in the image, as (its four corner points, what it says, confidence)."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    page = _engine().predict(image)[0]
    return [(poly.tolist(), text, score)
            for poly, text, score in zip(page["rec_polys"], page["rec_texts"], page["rec_scores"])]


def read(crop: np.ndarray) -> str:
    """What one label crop says, with the spaces taken out, or "" if nothing could be read.

    A crop occasionally yields more than one box (a stray tick mark read as punctuation, say),
    so the most confident reading wins.
    """
    if crop.size == 0:
        return ""
    found = read_page(crop)
    if not found:
        return ""
    _, text, _ = max(found, key=lambda one: one[2])
    return "".join(text.split())

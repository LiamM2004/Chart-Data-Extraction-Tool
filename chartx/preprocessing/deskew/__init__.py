"""Straighten a photo of a hand-drawn whiteboard chart.

    from chartx.preprocessing.deskew import deskew
    result = deskew(photo)    # result.image is the straightened photo

Modules: straighten (the steps; start here), board, three_edge_board, ellipse, rotation,
corner_editor (the drag-the-corners window) and geometry (shared helpers).
See the README, "How straightening works".
"""

from chartx.preprocessing.deskew.corner_editor import select_corners_manually
from chartx.preprocessing.deskew.straighten import DeskewResult, deskew

__all__ = ["DeskewResult", "deskew", "select_corners_manually"]

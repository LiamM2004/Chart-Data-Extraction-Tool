"""
deskew - straighten a photo of a hand-drawn whiteboard chart.

    from chartx.preprocessing.deskew import deskew
    result = deskew(photo)          # result.image is the straightened photo

How it is organised:

    straighten.py     the three steps and how they are combined (start here)
    board.py          find a whole board and work out how to straighten it
    three_edge_board.py  find a board with its far side or corners out of shot, from three of its edges
    ellipse.py        find a pie and round it off
    rotation.py       measure how tilted the chart is
    corner_editor.py  the drag-the-corners window, for when the board can't be found
    geometry.py       small shared helpers
"""

from chartx.preprocessing.deskew.corner_editor import select_corners_manually
from chartx.preprocessing.deskew.straighten import DeskewResult, deskew

__all__ = ["DeskewResult", "deskew", "select_corners_manually"]

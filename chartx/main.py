"""Command line: straighten a photo of a chart, clean up its glare and read its data.

    python -m chartx.main photo.jpg
    python -m chartx.main photo.jpg --manual                      # drag the board corners first
    python -m chartx.main photo.jpg --corners 10,20 900,15 920,700 5,690

Everything depends on the photo being straightened first, and that can fail; when it does the
reason is printed, and --manual or --corners let you place the board yourself. A window only
ever opens when asked for, so a scripted run never stops on one.
"""

import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from chartx.detection import bar_graph, line_graph
from chartx.preprocessing import glare
from chartx.preprocessing.deskew import deskew, select_corners_manually


from dataclasses import fields


# Display data for a graph or chart
def print_table(rows):
    """Print a list of dataclass objects as an aligned table."""
    if not rows:
        print("(no rows)")
        return

    headers = [f.name for f in fields(rows[0])]
    cells = [["-" if getattr(r, h) is None else str(getattr(r, h)) for h in headers] for r in rows]
    widths = [max(len(h), *(len(row[i]) for row in cells)) for i, h in enumerate(headers)]

    def line(items):
        return "  ".join(item.ljust(w) for item, w in zip(items, widths))

    print(line(headers))
    print(line(["-" * w for w in widths]))
    for row in cells:
        print(line(row))


def preprocess(photo, corners=None, ask_for_corners=None, note=None):
    """Straighten a BGR photo of a chart, clean the glare off it, and return the result.

    Args:
        photo: the BGR photo.
        corners: the board's four corners, if they are already known.
        ask_for_corners: called with (photo, deskew's best guess) to get corners from a
            person; return None to keep deskew's. Called whenever given, not only when deskew
            asks for review, since deskew is sometimes confident and wrong.
        note: called with each warning deskew raised, to pass on to the user.

    Returns:
        The preprocessed image.
    """
    straightened = deskew(photo) if corners is None else deskew(photo, corners=corners)
    if ask_for_corners is not None:
        chosen = ask_for_corners(photo, straightened.corners)
        if chosen is not None:
            straightened = deskew(photo, corners=chosen)
    if note is not None:
        for warning in straightened.warnings:
            note(warning)
    return glare.clean_whiteboard(straightened.image)


def extract(image):
    """Read the data out of a preprocessed chart image.

    Bars are looked for first, because a bar chart's axes would otherwise be followed as if
    the bars were a line. Returns (what the chart is, the rows read off it); the chart is None
    and the rows empty if neither kind was found.
    """
    bars = bar_graph.detect_bar_graph(image)
    if bars:
        return "bar chart", bars

    points = line_graph.detect_line_graph(image)
    if points:
        return "line chart", points

    return None, []


def _drag_the_corners(photo, best_guess):
    """Open the corner window so the board can be placed by hand. None if it was cancelled."""
    print("  Drag the board's four corners into place, then press Enter (Esc to keep what we have).")
    return select_corners_manually(photo, initial=best_guess)


def _corners_from(words):
    """The four "x,y" corners given on the command line, as an array. Raises ValueError if
    they aren't four readable points, rather than quietly carrying on without them."""
    points = []
    for word in words:
        across, _, down = word.partition(",")
        points.append((float(across), float(down)))        # ValueError if it isn't a number
    if len(points) != 4:
        raise ValueError(f"--corners wants four x,y points, got {len(points)}")
    return np.array(points, np.float32)


def main():
    # Get the image path from the command line
    arguments = [word for word in sys.argv[1:] if not word.startswith("-")]
    flags = [word for word in sys.argv[1:] if word.startswith("-")]
    if not arguments:
        print("Usage: python -m chartx.main image.jpg [--manual] [--corners x,y x,y x,y x,y]")
        exit(1)
    unknown = [flag for flag in flags if flag not in ("--manual", "--corners")]
    if unknown:
        print(f"Unknown option(s): {' '.join(unknown)}")
        exit(1)
    path = arguments[0]
    try:
        corners = _corners_from(arguments[1:]) if "--corners" in flags else None
    except ValueError as wrong:
        print(f"Error: {wrong}")
        exit(1)

    # Read in an image
    image = cv2.imread(path)
    if image is None:
        print(f"Error: image at {path} not found")
        exit(1)

    # Straighten the image, then clean up its glare. The corner window is only ever opened
    # when it is asked for, so a scripted run never stops on one.
    ask = _drag_the_corners if "--manual" in flags else None
    result = preprocess(image, corners=corners, ask_for_corners=ask, note=lambda warning: print(f"  note: {warning}"))

    # Work out what kind of chart it is and read its data off
    kind, rows = extract(result)

    if kind is None:
        print("No bar chart or line chart found")
        if ask is None and corners is None:
            print("  If the board was not straightened, try again with --manual to set its corners.")
    else:
        print(f"Found a {kind}")
    print_table(rows)

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()

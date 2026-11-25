"""Read the data from a photo of a chart.

    python -m chartx bar photo.jpg     # a bar chart
    python -m chartx line photo.jpg    # a line chart
    python -m chartx pie photo.jpg     # a pie chart

    python -m chartx line photo.jpg --manual     # place the board's corners by hand first
    python -m chartx line photo.jpg --corners 10,20 900,15 920,700 5,690

The photo is straightened and its glare cleaned up, then the chosen detector reads the chart.
The data is printed as a table and the cleaned photo is shown (with the bars boxed for bar
charts). Everything depends on the straightening, which can fail; when it does the reason is
printed, and --manual or --corners let you place the board yourself. A window only ever opens
when asked for, so a scripted run never stops on one.
"""

import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from chartx.detection import bar_graph, line_graph, pie_graph
from chartx.preprocessing import glare
from chartx.preprocessing.deskew import deskew, select_corners_manually


from dataclasses import fields


# The chart types the user can choose from, and the detector that reads each one
DETECTORS = {
    "bar": lambda image: bar_graph.detect_bar_graph(image),
    "line": lambda image: line_graph.detect_line_graph(image),
    "pie": lambda image: pie_graph.detect_pie_graph(image),
}


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


def draw_bar_boxes(ax, bars):
    """Outline each detected bar on the plot and label it with its index from the table."""
    for bar in bars:
        ax.add_patch(Rectangle((bar.x, bar.y), bar.width, bar.height,
                               fill=False, edgecolor="lime", linewidth=2))
        ax.text(bar.x + bar.width / 2, bar.y - 5, str(bar.index), color="lime",
                fontsize=12, fontweight="bold", ha="center", va="bottom")


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


def _drag_the_corners(photo, best_guess):
    """Open the corner window so the board can be placed by hand. None if it was cancelled."""
    print("  Drag the board\'s four corners into place, then press Enter (Esc to keep what we have).")
    return select_corners_manually(photo, initial=best_guess)


def _corners_from(words):
    """The four "x,y" corners given on the command line, as an array. Raises ValueError if
    they aren\'t four readable points, rather than quietly carrying on without them."""
    points = []
    for word in words:
        across, _, down = word.partition(",")
        points.append((float(across), float(down)))        # ValueError if it isn\'t a number
    if len(points) != 4:
        raise ValueError(f"--corners wants four x,y points, got {len(points)}")
    return np.array(points, np.float32)


def main():
    # Get the chart type, image path and options from the command line
    arguments = [word for word in sys.argv[1:] if not word.startswith("-")]
    flags = [word for word in sys.argv[1:] if word.startswith("-")]
    if len(arguments) < 2 or arguments[0] not in DETECTORS:
        print(f"Usage: python -m chartx {{{'|'.join(DETECTORS)}}} image.jpg "
              f"[--manual] [--corners x,y x,y x,y x,y]")
        exit(1)
    unknown = [flag for flag in flags if flag not in ("--manual", "--corners")]
    if unknown:
        print(f"Unknown option(s): {' '.join(unknown)}")
        exit(1)
    chart_type, path = arguments[0], arguments[1]
    try:
        corners = _corners_from(arguments[2:]) if "--corners" in flags else None
    except ValueError as wrong:
        print(f"Error: {wrong}")
        exit(1)

    # Read in an image
    image = cv2.imread(path)
    if image is None:
        print(f"Error: image at {path} not found")
        exit(1)

    # Straighten the image, then clean up its glare
    ask = _drag_the_corners if "--manual" in flags else None
    result = preprocess(image, corners=corners, ask_for_corners=ask,
                        note=lambda warning: print(f"  note: {warning}"))

    # Find the chosen type of chart and get its data object list
    chart_data = DETECTORS[chart_type](result)

    if not chart_data:
        print(f"No {chart_type} chart found")
        if ask is None and corners is None:
            print("  If the board was not straightened, try again with --manual to set its corners.")

    # Display the chart data
    print_table(chart_data)

    # Show the cleaned photo, with a box around each detected bar for bar charts
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    if chart_type == "bar":
        draw_bar_boxes(plt.gca(), chart_data)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()

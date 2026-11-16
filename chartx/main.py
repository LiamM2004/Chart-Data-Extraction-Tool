"""Command line: straighten a photo of a chart and clean up its glare.

    python -m chartx photo.jpg -o out/photo.png              # opens the drag window if unsure
    python -m chartx photo.jpg -o out/photo.png --manual     # always open the drag window
    python -m chartx photo.jpg -o out/photo.png --no-prompt  # never open a window (batch use)
    python -m chartx photo.jpg -o out/photo.png --corners 10,20 900,15 920,700 5,690
    python -m chartx photo.jpg -o out/photo.png --focal 0.5  # lens focal length / photo's long side
    python -m chartx photo.jpg -o out/photo.png --no-glare   # stop after straightening
    python -m chartx photo.jpg -o out/photo.png --debug out/debug.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from chartx.detection import bar_graph
from chartx.preprocessing import glare
from chartx.preprocessing.deskew import deskew


path = "sample-images/new_3.jpg"


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


def preprocess(photo):
    """Straighten a BGR photo of a chart, clean the glare off it, and return the result."""
    straightened = deskew(photo).image
    return glare.clean_whiteboard(straightened)


def main():

    # Read in an image
    image = cv2.imread(path)
    if image is None:
        print(f"Error: image at {path} not found")
        exit(1)

    # Straighten the image, then clean up its glare
    result = preprocess(image)

    # Check if this is a bar chart and get its data object list
    bar_graph_data = bar_graph.detect_bar_graph(result)

    if bar_graph_data == None:
        print("No bar graph found")

    # Display the bar chart data
    print_table(bar_graph_data)

    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


if __name__ == "__main__":
    main()

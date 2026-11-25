"""Read the data from a photo of a chart.

    python -m chartx photo.jpg        # work out which kind of chart it is
    python -m chartx bar photo.jpg    # a bar chart
    python -m chartx pie photo.jpg    # a pie chart

The photo is straightened and its glare cleaned up, then the chart is read. The data is printed
as a table and the cleaned photo is shown (with the bars boxed for bar charts). If no chart is
found, a message says so and the photo is still shown, so you can see what was looked at.
"""

import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from chartx.detection import bar_graph, pie_graph
from chartx.detection.auto import detect_chart
from chartx.preprocessing import glare
from chartx.preprocessing.deskew import deskew


from dataclasses import fields


# The chart types the user can choose from, and the detector that reads each one
DETECTORS = {
    "bar": lambda image: bar_graph.detect_bar_graph(image),
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


def preprocess(photo):
    """Straighten a BGR photo of a chart, clean the glare off it, and return the result."""
    straightened = deskew(photo).image
    return glare.clean_whiteboard(straightened)


# Read the chart type (optional) and image path from the command line
def parse_arguments(arguments):
    """Returns (chart type, image path), or None if the arguments don't make sense.
    The chart type is "auto" when none is given."""
    choices = ["auto", *DETECTORS]
    if len(arguments) == 1 and arguments[0] not in choices:
        return "auto", arguments[0]
    if len(arguments) == 2 and arguments[0] in choices:
        return arguments[0], arguments[1]
    return None


# Find the chart: the chosen kind, or whichever kind is in the photo
def find_charts(chart_type, image):
    """Returns [(chart type, data), ...], the chart to show first; [] if no chart is found."""
    if chart_type == "auto":
        return detect_chart(image)
    data = DETECTORS[chart_type](image)
    return [(chart_type, data)] if data else []


def main():
    # Get the chart type and image path from the command line
    arguments = parse_arguments(sys.argv[1:])
    if arguments is None:
        print(f"Usage: python -m chartx [auto|{'|'.join(DETECTORS)}] image.jpg")
        exit(1)
    chart_type, path = arguments

    # Read in an image
    image = cv2.imread(path)
    if image is None:
        print(f"Error: image at {path} not found")
        exit(1)

    # Straighten the image, then clean up its glare
    result = preprocess(image)

    # Find the chart and get its data object list
    found = find_charts(chart_type, result)

    if not found:
        kind = "chart" if chart_type == "auto" else f"{chart_type} chart"
        print(f"No {kind} found in {path}.")
        print(f"The supported charts are: {', '.join(DETECTORS)}. "
              "The cleaned photo is shown so you can check what was looked at.")
        found_type, data = None, []
    else:
        found_type, data = found[0]
        if chart_type == "auto":
            print(f"Found a {found_type} chart")
        print_table(data)
        for other_type, _ in found[1:]:
            print(f"A {other_type} chart was also found. To read it: python -m chartx {other_type} {path}")

    # Show the cleaned photo, with a box around each detected bar for bar charts
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    if found_type == "bar":
        draw_bar_boxes(plt.gca(), data)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()

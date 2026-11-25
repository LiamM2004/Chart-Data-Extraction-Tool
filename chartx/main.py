"""Read the data from a photo of a chart.

    python -m chartx bar photo.jpg    # a bar chart
    python -m chartx pie photo.jpg    # a pie chart

The photo is straightened and its glare cleaned up, then the chosen detector reads the chart.
The data is printed as a table and the cleaned photo is shown (with the bars boxed for bar charts).
"""

import sys

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from chartx.detection import bar_graph, pie_graph
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


def main():
    # Get the chart type and image path from the command line
    if len(sys.argv) < 3 or sys.argv[1] not in DETECTORS:
        print(f"Usage: python -m chartx {{{'|'.join(DETECTORS)}}} image.jpg")
        exit(1)
    chart_type, path = sys.argv[1], sys.argv[2]

    # Read in an image
    image = cv2.imread(path)
    if image is None:
        print(f"Error: image at {path} not found")
        exit(1)

    # Straighten the image, then clean up its glare
    result = preprocess(image)

    # Find the chosen type of chart and get its data object list
    chart_data = DETECTORS[chart_type](result)

    if not chart_data:
        print(f"No {chart_type} chart found")

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

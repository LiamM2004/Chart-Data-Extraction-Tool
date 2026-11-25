"""Work out which kind of chart is in a photo by running every detector and checking what they find.

    found = detect_chart(image)   # [(chart type, data), ...], most likely first; [] if no chart
"""

import math

from chartx.detection import bar_graph, line_graph, pie_graph
from chartx.detection.axes import find_axes


# Is the point (x, y) inside the pie's circle?
def _in_circle(x, y, circle):
    cx, cy, r = circle
    return math.hypot(x - cx, y - cy) < r


# Area of the rectangle that holds all the bars
def _bar_area(bars):
    left, right = min(b.x for b in bars), max(b.x + b.width for b in bars)
    top, bottom = min(b.y for b in bars), max(b.y + b.height for b in bars)
    return (right - left) * (bottom - top)


# Run every detector and keep the charts that make sense
def detect_chart(image):
    """Find the chart(s) in a preprocessed BGR image.

    Returns a list of (chart type, data) pairs, with the biggest chart first:
    "bar" with a list of `Bar`s, "line" with a list of `LinePoint`s, or "pie" with a list of
    `Sector`s. Returns [] if no chart is found.

    Each detector can find a chart that isn't there, so their answers are checked against each
    other. The pie detector is the one that gets fooled: on a bar chart it can fit a circle
    around the bars and read the gaps between them as sectors, and on a line chart it can take
    the corner where the axes meet for a pie's centre. A pie chart never has bars or a line's
    points inside it, so a pie with either inside its circle is dropped. When a board has two
    real charts side by side, both are kept.
    """
    bars = bar_graph.detect_bar_graph(image)
    axes = find_axes(image)                         # found once, shared with the line detector
    points = line_graph.detect_line_graph(image, axes) if axes is not None else []
    sectors, circle = pie_graph.find_pie(image)

    if sectors:
        bar_centres = [(b.x + b.width / 2, b.y + b.height / 2) for b in bars]
        line_points = [(p.x, p.y) for p in points if p.y is not None]
        if any(_in_circle(x, y, circle) for x, y in bar_centres + line_points):
            sectors = []

    found = []
    if bars:
        found.append(("bar", bars, _bar_area(bars)))
    if points:
        _, _, width, height = axes.plot_area
        found.append(("line", points, width * height))
    if sectors:
        found.append(("pie", sectors, math.pi * circle[2] ** 2))
    found.sort(key=lambda chart: chart[2], reverse=True)
    return [(chart_type, data) for chart_type, data, _ in found]

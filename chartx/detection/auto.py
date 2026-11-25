"""Work out which kind of chart is in a photo by running every detector and checking what they find.

    found = detect_chart(image)   # [(chart type, data), ...], most likely first; [] if no chart
"""

import math

from chartx.detection import bar_graph, pie_graph


# Is the middle of this bar inside the pie's circle?
def _inside(bar, circle):
    cx, cy, r = circle
    return math.hypot(bar.x + bar.width / 2 - cx, bar.y + bar.height / 2 - cy) < r


# Area of the rectangle that holds all the bars
def _bar_area(bars):
    left, right = min(b.x for b in bars), max(b.x + b.width for b in bars)
    top, bottom = min(b.y for b in bars), max(b.y + b.height for b in bars)
    return (right - left) * (bottom - top)


# Run both detectors and keep the charts that make sense
def detect_chart(image):
    """Find the chart(s) in a preprocessed BGR image.

    Returns a list of (chart type, data) pairs, "bar" with a list of `Bar` objects or "pie" with
    a list of `Sector` objects, with the biggest chart first. Returns [] if no chart is found.

    Each detector can find a chart that isn't there, so their answers are checked against each
    other: on a bar chart the pie detector can fit a circle around the bars and read the gaps
    between them as sectors. A pie chart never has bars inside it, so if bars were found inside
    the pie's circle, the pie is dropped. When a board has a bar chart and a pie chart side by
    side, both are kept.
    """
    bars = bar_graph.detect_bar_graph(image)
    sectors, circle = pie_graph.find_pie(image)

    if bars and sectors and any(_inside(bar, circle) for bar in bars):
        sectors = []

    found = []
    if bars:
        found.append(("bar", bars, _bar_area(bars)))
    if sectors:
        found.append(("pie", sectors, math.pi * circle[2] ** 2))
    found.sort(key=lambda chart: chart[2], reverse=True)
    return [(chart_type, data) for chart_type, data, _ in found]

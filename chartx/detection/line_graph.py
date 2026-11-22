"""Line graph detection. Expects a PREPROCESSED image: deskewed, white board, no glare/shadow.

    points = detect_line_graph(image)

The axes come from chartx.detection.axes, which also reads the y-axis labels, so the values
below are real axis values rather than fractions of the plot's height.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from chartx.detection.axes import Axes, find_axes
from chartx.preprocessing.deskew.rotation import find_ink

# How far a line may stray from where it was heading, as a fraction of the plot's height,
# before the ink there is taken to belong to something else.
WANDER = 0.04

# How far back a line looks to work out which way it is heading, as a fraction of the plot's
# width. It has to be long enough to reach back past a crossing, where both lines briefly run
# through the same ink and would otherwise forget which way they were going.
HEADING_RUN_UP = 0.06

# How long a stretch with no ink on it a line may be carried across, as a fraction of the plot's
# width. Past that the line has ended, and any ink further on belongs to something else -- a
# stray title or legend inside the plot area, say.
LONGEST_BREAK = 0.08

# How much colour a line needs before it is called anything but black (0-255).
FAINTEST_COLOUR = 25

# How far down the page a line must reach, as a fraction of the plot's width, to be a line at
# all rather than a stray mark.
SHORTEST_LINE = 0.3


@dataclass(frozen=True)
class LinePoint:
    """One data point read off a line: where it is, and what it is worth."""
    line: int                       # 1-based, lines ordered left to right by where they start
    category: str                   # the x-axis label it sits under
    x: int                          # px
    y: int | None                   # px (None where this line doesn't reach this category)
    value: float | None             # read from the y-axis labels (None if they were unreadable)
    colour: str                     # "#rrggbb"


def detect_line_graph(image: np.ndarray, axes: Axes | None = None) -> list[LinePoint]:
    """Read the data points off a line graph in a preprocessed BGR image.

    Returns a `LinePoint` for every line at every category, ordered by line and then left to
    right. A point whose line doesn't reach that category has `y` and `value` of None, so each
    line always has one point per category. Returns an empty list if no axes or no lines were
    found.

    axes: the already-found axes, if the caller has them; otherwise they are found here.
    """
    axes = axes if axes is not None else find_axes(image)
    if axes is None or len(axes.categories) < 2:
        return []                                   # one labelled point is not a line

    ink = _plot_ink(image, axes)
    traces = _separate_lines(ink, axes)
    if not traces:
        return []

    traces.sort(key=_first_drawn_column)
    return [point
            for number, trace in enumerate(traces, start=1)
            for point in _points_along(trace, _colour_along(image, ink, trace), number, axes)]


# ---------------------------------------------------------------- the ink the lines are drawn in

def _plot_ink(image: np.ndarray, axes: Axes) -> np.ndarray:
    """The ink inside the plot area, with the axis strokes and the glare specks rubbed out.

    Glare on a whiteboard leaves a fine pepper of dark-looking pixels that are in fact white.
    They are dropped by also asking that the ink really is darker or more coloured than the
    board, which matters because the lines are told apart by colour further down.

    The labels need no rubbing out of their own: they are written left of the y-axis and below
    the x-axis, and the plot area starts right of the one and ends above the other, so cropping
    to it has already dropped them.
    """
    left, top, width, height = axes.plot_area
    ink = cv2.bitwise_and(find_ink(image), _darker_than_the_board(image))
    plot = np.zeros_like(ink)
    plot[top:top + height, left:left + width] = ink[top:top + height, left:left + width]

    reach = max(3, int(0.01 * max(width, height)))
    cv2.line(plot, axes.y_axis[:2], axes.y_axis[2:], 0, 2 * reach)
    cv2.line(plot, axes.x_axis[:2], axes.x_axis[2:], 0, 2 * reach)

    return _without_specks(plot, min_pixels=max(6, int(0.00002 * width * height)))


def _darker_than_the_board(image: np.ndarray, ink_threshold: int = 40) -> np.ndarray:
    """Mask of the pixels dark or colourful enough to be pen rather than board."""
    return np.uint8(255 - image.min(axis=2) > ink_threshold) * 255


def _without_specks(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """The mask with every blob smaller than `min_pixels` rubbed out (glare leaves a lot of those)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, 4] >= min_pixels
    return keep[labels].astype(np.uint8) * 255


# ---------------------------------------------------------------- following a line

def _separate_lines(ink: np.ndarray, axes: Axes) -> list[np.ndarray]:
    """Follow every line drawn in this ink: one row per column, NaN where the line is missing.

    In each column the ink falls into runs, usually one per line. The lines are followed from
    one end to the other, each one claiming the run nearest to where it was heading. Keeping
    track of where a line was heading, rather than only where it was, is what carries it
    straight through a crossing: at the point where two lines meet, both read the same run, and
    beyond it each one picks up again on the side its own direction points to.
    """
    runs = [_run_centres(ink[:, column]) for column in range(ink.shape[1])]
    drawn = [column for column, centres in enumerate(runs) if centres]
    wander = max(3.0, WANDER * axes.plot_area[3])
    run_up = max(5, int(HEADING_RUN_UP * axes.plot_area[2]))
    longest_break = max(5, int(LONGEST_BREAK * axes.plot_area[2]))
    traces = _follow(runs, drawn, _how_many_lines(runs, drawn), wander, run_up, longest_break)
    reach = SHORTEST_LINE * axes.plot_area[2]
    return [_bridged(trace) for trace in traces if _drawn_width(trace) >= reach]


def _run_centres(column: np.ndarray) -> list[float]:
    """The middle of each unbroken stretch of ink in one column of the mask."""
    rows = np.nonzero(column)[0]
    if len(rows) == 0:
        return []
    breaks = np.nonzero(np.diff(rows) > 1)[0] + 1
    return [float(stretch.mean()) for stretch in np.split(rows, breaks)]


def _how_many_lines(runs: list[list[float]], drawn: list[int], most: int = 3) -> int:
    """How many lines this ink holds: the number of runs most columns are cut into.

    Columns at a crossing hold fewer runs and columns through a marker sometimes hold more,
    but neither is the common case, so the commonest count is the number of lines.
    """
    counts = np.bincount([len(runs[column]) for column in drawn], minlength=most + 1)
    return int(np.argmax(counts[1:most + 1])) + 1


def _follow(runs: list[list[float]], drawn: list[int], lines: int,
            wander: float, run_up: int, longest_break: int) -> list[np.ndarray]:
    """Follow `lines` lines across the ink, starting where they are all clearly apart."""
    start = next((place for place, column in enumerate(drawn)
                  if len(runs[column]) == lines), None)
    if start is None:
        return []

    traces = [np.full(len(runs), np.nan) for _ in range(lines)]
    for line, centre in enumerate(sorted(runs[drawn[start]])):
        traces[line][drawn[start]] = centre
    for step in (1, -1):
        _follow_one_way(runs, drawn, start, step, traces, wander, run_up, longest_break)
    return traces


def _follow_one_way(runs, drawn, start, step, traces, wander, run_up, longest_break):
    """Carry every line from the starting column to one end of the ink, a column at a time.

    Where two lines land on the same run they are crossing, and that run says nothing about
    which way either of them is going. Such a column is recorded on both lines but kept out of
    their run-ups, so each keeps heading the way it was until it is clear of the other.

    A line that finds no ink of its own for a long stretch has ended, and is dropped rather
    than carried on to whatever is drawn next.
    """
    recent = [[(drawn[start], trace[drawn[start]])] for trace in traces]
    going = [True] * len(traces)

    for column in drawn[start + step::step] if step > 0 else drawn[start - 1::-1]:
        for line, seen in enumerate(recent):
            if going[line] and abs(column - seen[-1][0]) > longest_break:
                going[line] = False                 # this line ended a while back
        chosen = [_nearest_run(runs[column], seen, wander, column) if still else None
                  for seen, still in zip(recent, going)]
        for line, centre in enumerate(chosen):
            if centre is None:
                continue
            traces[line][column] = centre
            if chosen.count(centre) == 1:           # ink this line has to itself
                recent[line].append((column, centre))
                del recent[line][:-run_up]          # only the recent run-up says where it is going


def _nearest_run(centres: list[float], seen: list[tuple[int, float]],
                 wander: float, column: int) -> float | None:
    """The run this line carries on into, or None if none of them is where it was heading."""
    last_column, last_row = seen[-1]
    expected = last_row + _heading(seen) * (column - last_column)
    centre = min(centres, key=lambda candidate: abs(candidate - expected))
    if abs(centre - expected) > wander * max(1.0, abs(column - last_column)):
        return None                                 # that ink belongs to something else
    return centre


def _heading(seen: list[tuple[int, float]]) -> float:
    """How many rows the line moves per column, measured over the run-up behind it."""
    (first_column, first_row), (last_column, last_row) = seen[0], seen[-1]
    if first_column == last_column:
        return 0.0
    return (last_row - first_row) / (last_column - first_column)


def _bridged(trace: np.ndarray) -> np.ndarray:
    """Fill the breaks in a followed line, where the pen lifted or another line crossed over it."""
    drawn = np.nonzero(~np.isnan(trace))[0]
    if len(drawn) < 2:
        return trace
    inside = np.arange(drawn[0], drawn[-1] + 1)
    trace[inside] = np.interp(inside, drawn, trace[drawn])
    return trace


def _drawn_width(trace: np.ndarray) -> int:
    """How far across the page the line was actually followed."""
    drawn = np.nonzero(~np.isnan(trace))[0]
    return int(drawn[-1] - drawn[0]) if len(drawn) else 0


def _first_drawn_column(trace: np.ndarray) -> int:
    drawn = np.nonzero(~np.isnan(trace))[0]
    return int(drawn[0]) if len(drawn) else 0


# ---------------------------------------------------------------- reading the values off

def _colour_along(image: np.ndarray, ink: np.ndarray, trace: np.ndarray, reach: int = 2) -> str:
    """The colour of the pen this line was drawn with, as #rrggbb.

    Taken from the ink the line itself runs through, so two lines in one chart are named by
    their own pens. It is the most strongly coloured pixels that are gone on, not the darkest:
    cleaning the glare off a photo drives the middle of a stroke to pure black whichever pen
    drew it, and what colour survives is in the paler pixels along its edges. Subtracting the
    weakest channel then leaves the pen's own colour, and nothing at all for a black pen.
    """
    pixels = _ink_along(image, ink, trace, reach)
    if len(pixels) == 0:
        return "#000000"

    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    strongest = np.median(pixels[chroma >= np.percentile(chroma, 75)], axis=0)
    blue, green, red = (strongest - strongest.min()).astype(int)
    if max(blue, green, red) < FAINTEST_COLOUR:
        return "#000000"
    return f"#{red:02x}{green:02x}{blue:02x}"


def _ink_along(image: np.ndarray, ink: np.ndarray, trace: np.ndarray, reach: int) -> np.ndarray:
    """The colours of the ink pixels within `reach` rows of the line, as an (n, 3) BGR array."""
    rows, columns = [], []
    for column in np.nonzero(~np.isnan(trace))[0]:
        for row in range(int(trace[column]) - reach, int(trace[column]) + reach + 1):
            if 0 <= row < ink.shape[0] and ink[row, column]:
                rows.append(row)
                columns.append(column)
    return image[rows, columns].astype(float) if rows else np.empty((0, 3))


def _points_along(trace: np.ndarray, colour: str, number: int, axes: Axes) -> list[LinePoint]:
    """Where the line sits above each category, and what that is worth on the y-axis."""
    columns = [category.x for category in axes.categories]
    # half the spacing: a corner is a reading of whichever category it sits nearest to
    snap_within = 0.5 * np.median(np.diff(columns)) if len(columns) > 1 else 0.0
    corners = _corners(trace)

    points = []
    for category in axes.categories:
        row = _row_at(trace, category.x, corners, snap_within)
        value = None if row is None or axes.scale is None else round(axes.scale.value(row), 2)
        points.append(LinePoint(number, category.text, int(category.x),
                                None if row is None else int(round(row)), value, colour))
    return points


def _row_at(trace: np.ndarray, column: int, corners: np.ndarray, snap_within: float) -> float | None:
    """The row the line runs along above this column, or None if the line doesn't reach it.

    A point is read at the nearest corner of the line when there is one close by, because that
    is where the pen actually marked the reading. Sampling at the label's centre instead would
    read partway along a segment whenever the writing is slanted, which on the sample photos it
    usually is, and would miss the first and last readings altogether when the label sits a few
    pixels beyond the end of the line.
    """
    near = corners[np.abs(corners - column) <= snap_within] if len(corners) else corners
    if len(near):
        column = int(near[np.abs(near - column).argmin()])
    if not 0 <= column < len(trace) or np.isnan(trace[column]):
        return None
    return float(trace[column])


def _corners(trace: np.ndarray, straightness: float = 0.02) -> np.ndarray:
    """The columns where the line changes direction, plus the two columns where it starts and ends.

    These are the points the chart was drawn from: a hand-drawn line is straight between its
    readings and bends at them.
    """
    drawn = np.nonzero(~np.isnan(trace))[0]
    if len(drawn) < 2:
        return np.empty(0, int)
    path = np.stack([drawn, trace[drawn]], axis=1).astype(np.int32)
    simplified = cv2.approxPolyDP(path, straightness * np.ptp(path[:, 1]) + 1.0, False)
    return np.unique(simplified[:, 0, 0])

"""Find a chart's axes and read their labels. Expects a PREPROCESSED image: deskewed,
white board, no glare/shadow.

    axes = find_axes(image)
    axes.scale.value(row)      # the axis value at a pixel row
    axes.categories            # the x-axis labels and where they sit

Reading the labels is done with PaddleOCR (chartx.detection.ocr), both the category text
and the y-axis numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations

import cv2
import numpy as np

from chartx.detection import ocr
from chartx.preprocessing.deskew.rotation import find_ink, find_long_straight_lines

# Runs of these are what category labels nearly always are. Where some come back as nonsense,
# the ones that do read say which sequence it is and where it starts, and the rest follow.
LABEL_SEQUENCES = (
    ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"),
    ("January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"),
    ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
    ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
    ("Q1", "Q2", "Q3", "Q4"),
)

# How alike two words have to be to count as the same one read badly, and how many labels have
# to be that alike before a sequence is trusted to correct the others.
CLEARLY_THE_SAME = 0.8
ENOUGH_TO_GO_ON = 2

# Grey level below which a pixel on the cleaned-up board counts as pen rather than paper.
INK_IS_DARKER_THAN = 150


@dataclass(frozen=True)
class Category:
    """An x-axis label: what it says and the pixel column it is centred on."""
    text: str
    x: int


@dataclass(frozen=True)
class ValueScale:
    """Turns a pixel row into an axis value: value = per_row * row + at_row_zero.

    `per_row` is negative, because values grow upwards while rows grow downwards.
    """
    per_row: float
    at_row_zero: float

    def value(self, row: float) -> float:
        return self.per_row * row + self.at_row_zero

    def row(self, value: float) -> float:
        return (value - self.at_row_zero) / self.per_row


@dataclass(frozen=True)
class Axes:
    """Where a chart's axes are, what its y-axis measures and what its categories are called."""
    origin: tuple[int, int]                 # where the two axes meet
    y_axis: tuple[int, int, int, int]       # x1, y1, x2, y2, top -> bottom
    x_axis: tuple[int, int, int, int]       # x1, y1, x2, y2, left -> right
    scale: ValueScale | None                # None if too few y-axis labels could be read
    categories: list[Category]

    @property
    def plot_area(self) -> tuple[int, int, int, int]:
        """The rectangle the data is drawn in (x, y, w, h): right of the y-axis, above the baseline."""
        left = max(self.y_axis[0], self.y_axis[2])
        top = min(self.y_axis[1], self.y_axis[3])
        right = max(self.x_axis[0], self.x_axis[2])
        bottom = min(self.x_axis[1], self.x_axis[3])
        return left, top, max(0, right - left), max(0, bottom - top)


def find_axes(image: np.ndarray) -> Axes | None:
    """Find the axes of a bar or line chart.

    Returns the `Axes`, or None if no pair of axis strokes meeting in a corner was found
    (for example on a pie chart).
    """
    strokes = _axis_strokes(image)
    if strokes is None:
        return None
    y_axis, x_axis, origin = strokes

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = find_ink(image)
    axis_length = _length(y_axis)

    value_boxes = _label_boxes(ink, _y_label_strip(y_axis, origin, image.shape), axis_length)
    scale = _fit_value_scale([_read_number(gray, box) for box in value_boxes],
                             _levels_of(gray, y_axis, value_boxes),
                             _median_gap(value_boxes, axis=1))

    category_boxes = _label_boxes(ink, _x_label_strip(x_axis, origin, image.shape), axis_length)
    texts = [_read_text(gray, box) for box in category_boxes]
    texts = _named_in_sequence(texts) or _numbered_in_steps(texts) or texts
    categories = [Category(text, int(box[0] + box[2] / 2))
                  for text, box in zip(texts, category_boxes)]

    return Axes(origin, y_axis, x_axis, scale, categories)


# ---------------------------------------------------------------- the axis strokes

def _axis_strokes(image: np.ndarray, min_len_frac: float = 0.12, max_tilt_deg: float = 8.0,
                  corner_frac: float = 0.15, margin_frac: float = 0.02):
    """The y-axis and x-axis strokes and the origin where they meet, or None.

    The axes are the longest upright and level strokes that meet in an "L": the origin has to
    sit at the bottom of the upright one, at the left end of the level one, and well inside the
    board. Those rules are what tell the axes apart from the board's own edges, which are longer
    but meet out in a corner of the photo. A few degrees of tilt are allowed, since deskewing
    doesn't always leave the axes perfectly square.
    """
    lines = find_long_straight_lines(image, min_len_frac)
    if len(lines) == 0:
        return None

    angle = np.degrees(np.arctan2(-(lines[:, 3] - lines[:, 1]), lines[:, 2] - lines[:, 0]))
    angle = (angle + 90) % 180 - 90                      # fold into [-90, 90)
    is_level = np.abs(angle) <= max_tilt_deg
    is_upright = np.abs(np.where(angle >= 0, angle - 90, angle + 90)) <= max_tilt_deg

    uprights = _merge_collinear(lines[is_upright], upright=True)
    levels = _merge_collinear(lines[is_level], upright=False)

    best, best_length = None, 0.0
    for y_axis in uprights:
        for x_axis in levels:
            origin = _crossing_point(y_axis, x_axis)
            if origin is None:
                continue
            if (np.hypot(*(origin - y_axis[2:])) > corner_frac * _length(y_axis)
                    or np.hypot(*(origin - x_axis[:2])) > corner_frac * _length(x_axis)):
                continue                                 # not an "L": they cross somewhere else
            if not _is_inside(origin, image.shape, margin_frac):
                continue                                 # a chart's origin is never at the board's edge
            total = _length(y_axis) + _length(x_axis)
            if total > best_length:
                best, best_length = (y_axis, x_axis, origin), total

    if best is None:
        return None
    y_axis, x_axis, origin = best
    return tuple(y_axis.astype(int)), tuple(x_axis.astype(int)), (int(origin[0]), int(origin[1]))


def _is_inside(point, shape, margin_frac: float) -> bool:
    """True if the point sits well inside the image rather than out on its edge."""
    margin = margin_frac * min(shape[:2])
    return (margin <= point[0] <= shape[1] - margin) and (margin <= point[1] <= shape[0] - margin)


def _merge_collinear(lines: np.ndarray, upright: bool, max_offset_px: float = 12.0) -> np.ndarray:
    """Join segments that lie on the same stroke into one, so each stroke is its full length.

    Upright strokes come back running top -> bottom, level ones left -> right.
    """
    merged: list[list[float]] = []
    for line in lines:
        x1, y1, x2, y2 = line.astype(float)
        if (y1 > y2) if upright else (x1 > x2):
            x1, y1, x2, y2 = x2, y2, x1, y1
        across = (x1 + x2) / 2 if upright else (y1 + y2) / 2
        for stroke in merged:
            if abs(across - ((stroke[0] + stroke[2]) / 2 if upright else (stroke[1] + stroke[3]) / 2)) > max_offset_px:
                continue
            if (y1 < stroke[1]) if upright else (x1 < stroke[0]):
                stroke[0], stroke[1] = x1, y1
            if (y2 > stroke[3]) if upright else (x2 > stroke[2]):
                stroke[2], stroke[3] = x2, y2
            break
        else:
            merged.append([x1, y1, x2, y2])
    return np.array(merged, float).reshape(-1, 4)


def _crossing_point(one: np.ndarray, other: np.ndarray) -> np.ndarray | None:
    """Where the two strokes cross once extended, or None if they are parallel."""
    one_start, one_run = one[:2], one[2:] - one[:2]
    other_start, other_run = other[:2], other[2:] - other[:2]
    cross = one_run[0] * other_run[1] - one_run[1] * other_run[0]
    if abs(cross) < 1e-9:
        return None
    gap = other_start - one_start
    along_one = (gap[0] * other_run[1] - gap[1] * other_run[0]) / cross
    return one_start + one_run * along_one


def _length(line) -> float:
    return float(np.hypot(line[2] - line[0], line[3] - line[1]))


# ---------------------------------------------------------------- finding the labels

def _y_label_strip(y_axis, origin, shape) -> tuple[int, int, int, int]:
    """The band left of the y-axis where its labels are written (x, y, w, h)."""
    length = _length(y_axis)
    right = int(min(y_axis[0], y_axis[2]) - 0.012 * length)      # clear of the tick marks
    left = max(0, int(right - 0.32 * length))
    top = max(0, int(min(y_axis[1], y_axis[3])))
    bottom = min(shape[0], int(origin[1] + 0.06 * length))
    return left, top, max(0, right - left), max(0, bottom - top)


def _x_label_strip(x_axis, origin, shape) -> tuple[int, int, int, int]:
    """The band below the x-axis where the category labels are written (x, y, w, h)."""
    height = int(0.22 * _length(x_axis))
    top = int(max(x_axis[1], x_axis[3]) + 0.004 * _length(x_axis))
    left = max(0, origin[0])                                    # clear of the y-axis's own labels
    right = min(shape[1], int(max(x_axis[0], x_axis[2])))
    return left, top, max(0, right - left), max(0, min(shape[0], top + height) - top)


# An axis title sits beyond the tick labels: further out than a y-axis's numbers, below an
# x-axis's categories. Between the two there is a band of blank board at least this much of the
# strip across. The labels themselves are never parted by that much, so a band this wide with
# writing on both sides of it is the gap between the labels and a title.
TITLE_BAND = 0.08


def _label_side_of_the_title(patch: np.ndarray, across: int) -> tuple[int, int]:
    """Where to cut `patch` to drop an axis title, as the (start, stop) of the part to keep.

    The labels are written against the axis and the title beyond them, so the strip holds two
    blocks of writing with blank board between. The widest band with ink on both sides is the
    one between them; a chart with no title has one block and is left whole.

    `across` is which way they are separated: columns (1) for a y-strip, whose title is left of
    its numbers, rows (0) for an x-strip, whose title is below its categories.
    """
    length = patch.shape[across]                     # patch is (rows, columns)
    inked = (patch > 0).any(axis=1 - across)         # ink anywhere along each row/column
    if not inked.any():
        return 0, length

    first, last = int(np.argmax(inked)), length - int(np.argmax(inked[::-1]))
    least = max(3, int(TITLE_BAND * length))
    widest, band = 0, None
    run_start = None
    for place in range(first, last):                 # only bands with writing on both sides
        if not inked[place]:
            if run_start is None:
                run_start = place
            continue
        if run_start is not None:
            if place - run_start > widest:
                widest, band = place - run_start, (run_start, place)
            run_start = None

    if band is None or widest < least:
        return first, last                           # one block of writing: no title to drop
    return (band[1], last) if across == 1 else (first, band[0])


def _label_boxes(ink: np.ndarray, strip: tuple[int, int, int, int], axis_length: float) -> np.ndarray:
    """Boxes round the labels written in `strip`, in reading order, as (n, 4) x, y, w, h.

    Specks of glare are dropped, nearby letters are joined into words, words that sit much
    closer together than the labels on an axis ever do are joined again, and a box that came
    out a whole multiple of the usual width is cut back up.

    The two later passes are what cope with slanted handwriting: no single reach joins "1990"
    into one label without also welding "2015" to "2020". Instead the gaps say where most of
    the cuts go, and the even width of the rest says where the remaining ones do.
    """
    left, top, width, height = strip
    if width <= 0 or height <= 0:
        return np.empty((0, 4), int)
    patch = _without_specks(ink[top:top + height, left:left + width],
                            min_height=max(3, int(0.015 * axis_length)))

    # drop the axis title, if the strip holds one as well as the labels
    across = 1 if height > width else 0              # 1 = columns (y-strip), 0 = rows (x-strip)
    start, stop = _label_side_of_the_title(patch, across)
    if across == 1:
        patch, left = patch[:, start:stop], left + start
    else:
        patch, top = patch[start:stop], top + start
    if patch.size == 0:
        return np.empty((0, 4), int)

    letter_heights = cv2.connectedComponentsWithStats(patch)[2][1:, 3]
    if len(letter_heights) == 0:
        return np.empty((0, 4), int)

    along = 1 if height > width else 0          # labels run down a column, or across a row
    boxes = _words(patch, reach=max(3, int(0.45 * np.median(letter_heights))))
    boxes = _split_double_labels(_joined_into_labels(boxes, along), along)
    boxes = _without_scraps(boxes, along)
    if len(boxes) == 0:
        return np.empty((0, 4), int)

    boxes[:, 0] += left
    boxes[:, 1] += top
    return boxes[np.argsort(boxes[:, along])]


def _words(patch: np.ndarray, reach: int) -> np.ndarray:
    """Group the ink into words by closing gaps up to `reach` wide, as (n, 4) x, y, w, h boxes.

    Anything much shorter than the word it was joined to (a tick mark, the dot of an "i") is
    trimmed off the box again, so each box holds the writing and nothing else.
    """
    joined = cv2.dilate(patch, cv2.getStructuringElement(cv2.MORPH_RECT, (reach, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)

    boxes = []
    for blob in range(1, count):
        blob_left, blob_top, blob_width, blob_height = stats[blob, :4]
        rows = slice(blob_top, blob_top + blob_height)
        columns = slice(blob_left, blob_left + blob_width)
        word = patch[rows, columns].copy()
        word[labels[rows, columns] != blob] = 0
        letters = cv2.connectedComponentsWithStats(word)[2][1:]
        tall = letters[letters[:, 3] >= 0.45 * letters[:, 3].max()]
        if len(tall) == 0:
            continue
        left, top = tall[:, 0].min(), tall[:, 1].min()
        boxes.append((blob_left + left, blob_top + top,
                      (tall[:, 0] + tall[:, 2]).max() - left, (tall[:, 1] + tall[:, 3]).max() - top))
    return np.array(boxes, int) if boxes else np.empty((0, 4), int)


def _joined_into_labels(words: np.ndarray, along: int) -> np.ndarray:
    """Join neighbouring word fragments back into whole labels, as (n, 4) x, y, w, h boxes.

    The usual gap between labels is taken as the middle of the wider half of the gaps, which
    ignores the small gaps inside a label that has broken into pieces. Anything less than half
    of that is a gap inside a label, so the boxes either side of it belong together.
    """
    if len(words) < 3:
        return words
    words = words[np.argsort(words[:, along])]
    starts, sizes = words[:, along], words[:, along + 2]
    gaps = starts[1:] - (starts[:-1] + sizes[:-1])
    limit = 0.5 * np.median(gaps[gaps >= np.median(gaps)])

    labels, group = [], [words[0]]
    for gap, word in zip(gaps, words[1:]):
        if gap >= limit:
            labels.append(_bounding_box(np.array(group)))
            group = []
        group.append(word)
    labels.append(_bounding_box(np.array(group)))
    return np.array(labels, int)


def _split_double_labels(labels: np.ndarray, along: int, min_ratio: float = 1.7) -> np.ndarray:
    """Cut up any label box that is a whole multiple of the usual width.

    Two labels written so close that no gap separates them (on the sample photos, "2010" and
    "2015") come out as one box twice the width of its neighbours. Since labels are evenly
    spaced and about as wide as each other, such a box is split into equal parts.
    """
    usual = np.median(labels[:, along + 2])
    if usual <= 0:
        return labels

    split = []
    for label in labels:
        parts = int(round(label[along + 2] / usual))
        if parts < 2 or label[along + 2] / usual < min_ratio:
            split.append(label)
            continue
        for part in range(parts):
            piece = list(label)
            piece[along + 2] = label[along + 2] // parts
            piece[along] = label[along] + part * piece[along + 2]
            split.append(piece)
    return np.array(split, int)


def _without_scraps(labels: np.ndarray, along: int, least_of_usual: float = 0.25) -> np.ndarray:
    """Drop any box far too small along the axis to be one of its labels.

    Labels on an axis are about as big as each other, which is the same thing
    _split_double_labels() leans on to cut a box that came out a whole multiple too big. A
    speck of glare that survived as its own box is the other side of that: it sits among
    labels many times its size, and would otherwise be read as a label of its own and throw
    the count of them out.
    """
    if len(labels) < 3:
        return labels
    usual = np.median(labels[:, along + 2])
    return labels[labels[:, along + 2] >= least_of_usual * usual]


def _bounding_box(boxes: np.ndarray) -> tuple[int, int, int, int]:
    """The one box that holds all of them."""
    left, top = boxes[:, 0].min(), boxes[:, 1].min()
    return left, top, (boxes[:, 0] + boxes[:, 2]).max() - left, (boxes[:, 1] + boxes[:, 3]).max() - top


# ---------------------------------------------------------------- what row a label names

# How far out from the axis to look for a tick, and how much ink a row needs to be one. The
# reach is in label heights so it holds at any size the chart is drawn; anything from 0.15 to
# 0.35 reads the sample photos equally well, so these sit mid-range rather than on a best fit.
TICK_REACH = 0.3
TICK_INK = 0.25


def _levels_of(gray: np.ndarray, y_axis, boxes: np.ndarray) -> np.ndarray:
    """The pixel row each y-axis label names: its tick mark, or its own middle if it has none.

    The middle of a handwritten "140" is a poor guess at the level it labels, sitting up to
    15 rows off on the sample photos and tilting the fitted scale by two or three per cent.
    The tick drawn against the axis is a measurement of that level rather than a guess at it.
    """
    middles = np.array([box[1] + box[3] / 2 for box in boxes], float)
    if len(boxes) == 0:
        return middles

    label_height = float(np.median(boxes[:, 3]))
    ticks = _tick_rows(gray, y_axis, label_height)
    if not ticks:
        return middles

    levels = middles.copy()
    for place, (box, middle) in enumerate(zip(boxes, middles)):
        nearest = min(ticks, key=lambda row: abs(row - middle))
        if abs(nearest - middle) <= 0.9 * box[3]:      # a tick further off belongs to another label
            levels[place] = nearest
    return levels


def _tick_rows(gray: np.ndarray, y_axis, label_height: float) -> list[int]:
    """The rows of the tick marks: short spurs of ink sticking out of the y-axis, away from
    the plot. Taken as the rows where that narrow strip is unusually inky."""
    column = int((y_axis[0] + y_axis[2]) / 2)
    top, bottom = sorted((int(y_axis[1]), int(y_axis[3])))
    reach = max(3, int(TICK_REACH * label_height))
    strip = (gray[top:bottom, max(0, column - reach - 2):max(1, column - 2)] < INK_IS_DARKER_THAN)
    if strip.size == 0:
        return []

    inkiness = np.convolve(strip.mean(axis=1), np.ones(3) / 3, "same")
    background = np.median(inkiness)
    rows: list[int] = []
    for row in range(2, len(inkiness) - 2):
        if inkiness[row] - background < TICK_INK:
            continue
        if inkiness[row] < inkiness[max(0, row - 4):row + 5].max():
            continue                                   # not the inkiest row of its own tick
        if not rows or top + row - rows[-1] > 0.4 * label_height:
            rows.append(top + row)
    return rows


def _without_specks(mask: np.ndarray, min_height: int) -> np.ndarray:
    """The mask with every blob shorter than `min_height` rubbed out (glare leaves a lot of those)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, 3] >= min_height
    return keep[labels].astype(np.uint8) * 255


def _median_gap(boxes: np.ndarray, axis: int) -> float:
    """The usual distance between neighbouring label centres, used as the fitting tolerance."""
    if len(boxes) < 2:
        return 0.0
    centres = np.sort(boxes[:, axis] + boxes[:, axis + 2] / 2)
    return float(np.median(np.diff(centres)))


# ---------------------------------------------------------------- reading the labels

def _crop(gray: np.ndarray, box, pad_frac: float = 0.35) -> np.ndarray:
    left, top, width, height = box
    pad = max(3, int(height * pad_frac))
    return gray[max(0, top - pad):top + height + pad, max(0, left - pad):left + width + pad]


def _read_number(gray: np.ndarray, box) -> list[float]:
    """What this y-axis label says, as a list holding one number, or empty if it isn't one.

    A list because _fit_value_scale() weighs up every candidate reading of a label. Only one
    is offered now, but pooling several is the repair to reach for if a number ever misreads.
    """
    text = ocr.read(_crop(gray, box))
    return [float(text)] if text.isdigit() else []


def _read_text(gray: np.ndarray, box) -> str:
    """What a category label says, or "" if it can't be read. Some come back as nonsense,
    which the sequence fitting below is there to repair."""
    return ocr.read(_crop(gray, box))


def _named_in_sequence(texts: list[str]) -> list[str] | None:
    """The run of a familiar sequence these readings are, or None if they are not one.

    Every sequence is tried at every starting point, and the run that the readings look most
    like wins. A couple of labels read clearly is enough to pin down which run it is, and the
    labels that came back as nonsense are then named from their place in it rather than from
    what the OCR made of them.
    """
    best, best_score = None, 0.0
    for sequence in LABEL_SEQUENCES:
        for start in range(len(sequence) - len(texts) + 1):
            run = sequence[start:start + len(texts)]
            alike = [_how_alike(text, word) for text, word in zip(texts, run)]
            if sum(score >= CLEARLY_THE_SAME for score in alike) < ENOUGH_TO_GO_ON:
                continue                                     # too few read clearly to go on
            if sum(alike) > best_score:
                best, best_score = list(run), sum(alike)
    return best


def _numbered_in_steps(texts: list[str]) -> list[str] | None:
    """The run of evenly spaced numbers these readings look most like, or None if they are not one.

    The same idea as _named_in_sequence() for the axes that count rather than name: 1, 2, 3, 4
    or 1990, 1995, 2000. Each pair of readings that parse as numbers proposes a first value and
    a step, and the run those two imply is scored against every reading, so a label the OCR
    got wrong is named from its place in the run like a misread month is.
    """
    read = [(place, float(word)) for place, word in enumerate(texts) if word.isdigit()]
    if len(read) < ENOUGH_TO_GO_ON:
        return None

    best, best_score = None, 0.0
    for (place, value), (later_place, later_value) in combinations(read, 2):
        step = (later_value - value) / (later_place - place)
        if step <= 0 or step != int(step):
            continue                                     # axes count up, in whole numbers
        start = value - place * step
        if start < 0:
            continue
        run = [str(int(start + place * step)) for place in range(len(texts))]
        alike = [_how_alike(word, number) for word, number in zip(texts, run)]
        if sum(score >= CLEARLY_THE_SAME for score in alike) < ENOUGH_TO_GO_ON:
            continue                                     # too few read clearly to go on
        if sum(alike) > best_score:
            best, best_score = run, sum(alike)
    return best


def _how_alike(read: str, word: str) -> float:
    """How much a reading looks like a word, from 0 (nothing alike) to 1 (the same)."""
    return SequenceMatcher(None, read.lower(), word.lower()).ratio()


def _fit_value_scale(readings: list[list[float]], rows: np.ndarray, row_gap: float) -> ValueScale | None:
    """Fit value = per_row * row + at_row_zero to the labels, ignoring the ones misread.

    Axis labels climb evenly, so the right scale is the one that puts the most labels within a
    fraction of a label's spacing of where it predicts them. Every reading of every label is
    tried as a candidate, which lets a label that was read three different ways still be used
    through whichever reading agrees with its neighbours.

    Returns None if fewer than two labels agree on any scale.
    """
    tolerance = 0.3 * row_gap if row_gap > 0 else 0.0
    pairs = [(row, value) for row, values in zip(rows, readings) for value in values]
    if len(pairs) < 2 or tolerance <= 0:
        return None

    best, best_score = None, (1, 0.0)        # one agreeing label is not a fit
    for (row_a, value_a), (row_b, value_b) in combinations(pairs, 2):
        if row_a == row_b or value_a == value_b:
            continue
        per_row = (value_a - value_b) / (row_a - row_b)
        if per_row >= 0:                                         # values must grow upwards
            continue
        at_row_zero = value_a - per_row * row_a
        scale = ValueScale(per_row, at_row_zero)

        score = _how_well_it_fits(scale, readings, rows, tolerance)
        if score > best_score:
            best, best_score = scale, score

    if best is None:
        return None
    return _refit_on_agreeing_labels(best, readings, rows, tolerance)


def _how_well_it_fits(scale: ValueScale, readings, rows, tolerance) -> tuple[int, float]:
    """How well a scale explains the labels: (how many it agrees with, minus how far out they
    are). The second term only breaks ties between scales matching the same number of labels.

    If labels ever carry several candidate readings again, weight each by how well voted its
    reading was: a run of truncated readings ("25, 20, 15, 10" as "3, 2, 1, 0") sits on a line
    of its own and can otherwise outvote the labels that were read properly.
    """
    errors = []
    for row, values in zip(rows, readings):
        if not values:
            continue
        error = min(abs(scale.row(value) - row) for value in values)
        if error <= tolerance:
            errors.append(error)
    if not errors:
        return 0, 0.0
    return len(errors), -sum(errors) / len(errors)


def _refit_on_agreeing_labels(scale: ValueScale, readings, rows, tolerance) -> ValueScale:
    """Least-squares fit through every label that the scale already agrees with."""
    chosen = [(row, min(values, key=lambda value: abs(scale.row(value) - row)))
              for row, values in zip(rows, readings) if values]
    chosen = [(row, value) for row, value in chosen if abs(scale.row(value) - row) <= tolerance]
    if len(chosen) < 3:
        return scale
    fit_rows = np.array([row for row, _ in chosen])
    fit_values = np.array([value for _, value in chosen])
    per_row, at_row_zero = np.polyfit(fit_rows, fit_values, 1)
    return ValueScale(float(per_row), float(at_row_zero)) if per_row < 0 else scale

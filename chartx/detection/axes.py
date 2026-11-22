"""Find a chart's axes and read their labels. Expects a PREPROCESSED image: deskewed,
white board, no glare/shadow.

    axes = find_axes(image)
    axes.scale.value(row)      # the axis value at a pixel row
    axes.categories            # the x-axis labels and where they sit

Reading the labels needs Tesseract (see the README). Without it the axes are still
found, but `scale` is None and the categories are unnamed.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations

import cv2
import numpy as np

from chartx.preprocessing.deskew.rotation import find_ink, find_long_straight_lines

DIGITS = "0123456789"

# How to read one label: (padding round the crop, height to upscale it to, thicken the strokes?,
# Tesseract page-segmentation mode). Handwriting reads differently depending on all four, so every
# label is tried each of these ways and the readings are pooled; the fitting below then keeps
# whichever readings line up with the other labels. These five were chosen by trying every
# combination on the sample photos and keeping the smallest set that reads them all.
OCR_ATTEMPTS = ((0.2, 48, True, 7), (0.2, 128, True, 7), (0.5, 48, False, 8),
                (0.5, 96, True, 7), (0.5, 128, True, 8))

# Runs of these are what category labels nearly always are. Slanted handwriting reads badly
# enough that some labels come back as nonsense ("May" as "flosy" on one of the sample photos),
# but the ones that do read say which sequence it is and where it starts, and the rest follow.
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
                             np.array([box[1] + box[3] / 2 for box in value_boxes]),
                             _median_gap(value_boxes, axis=1))

    category_boxes = _label_boxes(ink, _x_label_strip(x_axis, origin, image.shape), axis_length)
    texts = [_read_text(gray, box) for box in category_boxes]
    texts = _named_in_sequence(texts) or texts
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

    letter_heights = cv2.connectedComponentsWithStats(patch)[2][1:, 3]
    if len(letter_heights) == 0:
        return np.empty((0, 4), int)

    along = 1 if height > width else 0          # labels run down a column, or across a row
    boxes = _words(patch, reach=max(3, int(0.45 * np.median(letter_heights))))
    boxes = _split_double_labels(_joined_into_labels(boxes, along), along)
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


def _bounding_box(boxes: np.ndarray) -> tuple[int, int, int, int]:
    """The one box that holds all of them."""
    left, top = boxes[:, 0].min(), boxes[:, 1].min()
    return left, top, (boxes[:, 0] + boxes[:, 2]).max() - left, (boxes[:, 1] + boxes[:, 3]).max() - top


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

def _ocr(crop: np.ndarray, whitelist: str, upscale_to: int, thicken: bool, psm: int) -> str:
    """Run Tesseract on one label crop. Returns "" if Tesseract isn't installed."""
    try:
        import pytesseract
    except ImportError:
        return ""
    if crop.size == 0:
        return ""

    scale = upscale_to / crop.shape[0]
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, sharp = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thicken:
        sharp = cv2.erode(sharp, np.ones((3, 3), np.uint8))      # the writing is black, so erode fattens it
    sharp = cv2.copyMakeBorder(sharp, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)

    config = f"--psm {psm}" + (f" -c tessedit_char_whitelist={whitelist}" if whitelist else "")
    try:
        return pytesseract.image_to_string(sharp, config=config).strip()
    except Exception:                                            # Tesseract installed but not working
        return ""


def _crop(gray: np.ndarray, box, pad_frac: float = 0.35) -> np.ndarray:
    left, top, width, height = box
    pad = max(3, int(height * pad_frac))
    return gray[max(0, top - pad):top + height + pad, max(0, left - pad):left + width + pad]


def _read_number(gray: np.ndarray, box) -> list[float]:
    """Every number this label might say, most-read first.

    A "4" drawn with an open top is a "9" to Tesseract, and a faint "25" is often just "2".
    Rather than commit to one reading, all of them are returned, and _fit_value_scale() keeps
    whichever ones line up with the other labels.
    """
    votes: dict[float, int] = {}
    for pad_frac, upscale_to, thicken, psm in OCR_ATTEMPTS:
        text = _ocr(_crop(gray, box, pad_frac), DIGITS, upscale_to, thicken, psm)
        if text.isdigit():
            votes[float(text)] = votes.get(float(text), 0) + 1
    return sorted(votes, key=lambda value: -votes[value])


def _read_text(gray: np.ndarray, box) -> str:
    """What a category label says, or "" if it can't be read."""
    votes: dict[str, int] = {}
    for pad_frac, upscale_to, thicken, psm in OCR_ATTEMPTS:
        text = "".join(ch for ch in _ocr(_crop(gray, box, pad_frac), "", upscale_to, thicken, psm) if ch.isalnum())
        if text:
            votes[text] = votes.get(text, 0) + 1
    return max(votes, key=lambda text: (votes[text], len(text)), default="")


def _named_in_sequence(texts: list[str]) -> list[str] | None:
    """The run of a familiar sequence these readings are, or None if they are not one.

    Every sequence is tried at every starting point, and the run that the readings look most
    like wins. A couple of labels read clearly is enough to pin down which run it is, and the
    labels that came back as nonsense are then named from their place in it rather than from
    what Tesseract made of them.
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

    best, best_score = None, (1, 0.0)
    for (row_a, value_a), (row_b, value_b) in combinations(pairs, 2):
        if row_a == row_b or value_a == value_b:
            continue
        per_row = (value_a - value_b) / (row_a - row_b)
        if per_row >= 0:                                         # values must grow upwards
            continue
        at_row_zero = value_a - per_row * row_a
        scale = ValueScale(per_row, at_row_zero)

        agreed = [min(abs(scale.row(value) - row) for value in values) for row, values in zip(rows, readings) if values]
        close = [error for error in agreed if error <= tolerance]
        score = (len(close), -sum(close) / len(close) if close else 0.0)
        if score > best_score:
            best, best_score = scale, score

    if best is None:
        return None
    return _refit_on_agreeing_labels(best, readings, rows, tolerance)


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

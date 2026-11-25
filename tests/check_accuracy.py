"""Measure the line graph extraction against the hand-read values in fixtures/line_charts.json.

    python tests/check_accuracy.py                 # every chart in the fixtures
    python tests/check_accuracy.py 121850          # only the photos whose name contains this

Prints each chart's extracted values beside the true ones and the error in each, then a summary.
Unlike the tests, this always prints the full table, so it is the thing to run while tuning.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartx.detection.line_graph import detect_line_graph
from chartx.main import preprocess

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "line_charts.json"


def extracted_lines(photo_path):
    """Run the whole pipeline on a photo. Returns {line number: [LinePoint, ...]}."""
    image = cv2.imread(str(photo_path))
    if image is None:
        raise SystemExit(f"could not read {photo_path}")
    lines = {}
    for point in detect_line_graph(preprocess(image)):
        lines.setdefault(point.line, []).append(point)
    return lines


def closest_line(points_by_line, expected, used):
    """The extracted line that best matches `expected`, of the ones not already matched."""
    free = {number: points for number, points in points_by_line.items() if number not in used}
    if not free:
        return None, None
    scored = {number: _mean_error(points, expected) for number, points in free.items()}
    best = min(scored, key=lambda number: scored[number])
    return best, free[best]


def _mean_error(points, expected):
    """Average distance between a line's values and the expected ones, or infinity if unusable."""
    values = [point.value for point in points]
    if len(values) != len(expected):
        return float("inf")
    read = [(value, true) for value, true in zip(values, expected) if value is not None]
    if not read:
        return float("inf")
    return float(np.mean([abs(value - true) for value, true in read]))


def check(chart, tolerance_fraction):
    """Compare one chart against its true values. Returns (errors, number of values missed)."""
    low, high = chart["axis_range"]
    allowed = tolerance_fraction * (high - low)
    print(f"\n{chart['photo']}  ({chart['title']})   tolerance +-{allowed:g}")

    points_by_line = extracted_lines(ROOT / "sample-images" / chart["photo"])
    print(f"  found {len(points_by_line)} line(s), expected {len(chart['lines'])}")

    errors, missed, used = [], 0, set()
    for colour, expected in chart["lines"].items():
        number, points = closest_line(points_by_line, expected, used)
        if points is None:
            print(f"  {colour:>6}: not found")
            missed += len(expected)
            continue
        used.add(number)
        print(f"  {colour:>6}: line {number} drawn in {points[0].colour}")
        print(f"      {'category':<10}{'true':>8}{'read':>8}{'error':>8}")
        for category, true_value, point in zip(chart["categories"], expected, points):
            if point.value is None:
                print(f"      {category:<10}{true_value:>8g}{'-':>8}{'-':>8}")
                missed += 1
                continue
            error = abs(point.value - true_value)
            errors.append(error)
            flag = "" if error <= allowed else "   <-- over tolerance"
            print(f"      {category:<10}{true_value:>8g}{point.value:>8g}{error:>8.1f}{flag}")
        missed += max(0, len(expected) - len(points))
    return errors, missed


def main():
    fixtures = json.loads(FIXTURES.read_text())
    wanted = sys.argv[1] if len(sys.argv) > 1 else ""
    charts = [chart for chart in fixtures["charts"] if wanted in chart["photo"]]
    if not charts:
        raise SystemExit(f"no chart in the fixtures matches {wanted!r}")

    all_errors, all_missed, over = [], 0, 0
    for chart in charts:
        low, high = chart["axis_range"]
        errors, missed = check(chart, fixtures["tolerance"])
        over += sum(error > fixtures["tolerance"] * (high - low) for error in errors)
        all_errors += [error / (high - low) for error in errors]       # as a fraction, so charts compare
        all_missed += missed

    print(f"\n{'-' * 60}\n{len(all_errors)} values read, {all_missed} missed, {over} over tolerance")
    if all_errors:
        print(f"error as a fraction of the axis range: mean {np.mean(all_errors):.1%}, "
              f"worst {np.max(all_errors):.1%}")
    return 1 if over or all_missed else 0


if __name__ == "__main__":
    sys.exit(main())

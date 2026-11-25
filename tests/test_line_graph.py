"""
Tests for chartx/detection/line_graph.py: following lines across a plot, finding their
corners, and reading values off them.

The last class runs the whole pipeline over the photos in sample-images/ and checks the
values against the ones read off them by hand in fixtures/line_charts.json.

Run from the project root:

    python3 -m unittest discover tests -v
"""

import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from chartx.detection import line_graph
from chartx.detection.axes import Axes, Category, ValueScale
from chartx.detection.line_graph import detect_line_graph
from chartx.main import preprocess

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "line_charts.json"
SAMPLES = Path(__file__).resolve().parent.parent / "sample-images"


def mask(*lines, width=400, height=300, thickness=2):
    """A plot-shaped mask with each line drawn through its given (column, row) points."""
    drawn = np.zeros((height, width), np.uint8)
    for points in lines:
        cv2.polylines(drawn, [np.array(points, np.int32)], False, 255, thickness)
    return drawn


def fake_axes(categories, scale=ValueScale(-0.1, 30.0)):
    """Axes with no image behind them, for testing the reading-off on its own."""
    return Axes(origin=(0, 300), y_axis=(0, 0, 0, 300), x_axis=(0, 300, 400, 300),
                scale=scale, categories=[Category(label, column) for label, column in categories])


class RunTests(unittest.TestCase):
    def test_finds_the_middle_of_each_stretch_of_ink_in_a_column(self):
        column = np.zeros(100, np.uint8)
        column[10:15] = 255
        column[60:63] = 255
        self.assertEqual(line_graph._run_centres(column), [12.0, 61.0])

    def test_an_empty_column_has_no_runs(self):
        self.assertEqual(line_graph._run_centres(np.zeros(100, np.uint8)), [])

    def test_counts_the_lines_by_how_many_runs_most_columns_hold(self):
        runs = [[1.0, 2.0]] * 50 + [[1.0]] * 10 + [[1.0, 2.0, 3.0]] * 5
        self.assertEqual(line_graph._how_many_lines(runs, list(range(65))), 2)

    def test_one_line_when_every_column_holds_one_run(self):
        runs = [[1.0]] * 40
        self.assertEqual(line_graph._how_many_lines(runs, list(range(40))), 1)


class FollowTests(unittest.TestCase):
    def setUp(self):
        self.axes = fake_axes([("a", 50), ("b", 200), ("c", 350)])

    def test_follows_a_single_line(self):
        traces = line_graph._separate_lines(mask([(10, 250), (200, 100), (390, 150)]), self.axes)
        self.assertEqual(len(traces), 1)
        self.assertAlmostEqual(traces[0][200], 100, delta=3)

    def test_keeps_two_crossing_lines_apart(self):
        rising = [(10, 250), (390, 60)]
        falling = [(10, 60), (390, 250)]
        traces = line_graph._separate_lines(mask(rising, falling), self.axes)
        self.assertEqual(len(traces), 2)

        rows_at_the_end = sorted(trace[385] for trace in traces)
        self.assertAlmostEqual(rows_at_the_end[0], 61, delta=6, msg="the rising line lost its way")
        self.assertAlmostEqual(rows_at_the_end[1], 249, delta=6, msg="the falling line lost its way")

    def test_ignores_a_mark_too_short_to_be_a_line(self):
        traces = line_graph._separate_lines(mask([(10, 150), (60, 150)]), self.axes)
        self.assertEqual(traces, [])

    def test_does_not_run_on_to_ink_far_beyond_the_end_of_the_line(self):
        stray = mask([(10, 200), (250, 200)], [(380, 40), (395, 40)])
        traces = line_graph._separate_lines(stray, self.axes)
        self.assertEqual(len(traces), 1)
        self.assertTrue(np.isnan(traces[0][390]), "the line was carried on to the stray mark")

    def test_bridges_a_short_break_in_a_line(self):
        broken = mask([(10, 200), (180, 200)], [(200, 200), (390, 200)])
        traces = line_graph._separate_lines(broken, self.axes)
        self.assertEqual(len(traces), 1)
        self.assertAlmostEqual(traces[0][190], 200, delta=3)


class CornerTests(unittest.TestCase):
    def test_finds_the_bend_and_both_ends_of_a_line(self):
        trace = np.full(400, np.nan)
        trace[50:200] = np.linspace(250, 100, 150)
        trace[200:350] = np.linspace(100, 200, 150)
        corners = line_graph._corners(trace)
        self.assertIn(50, corners)
        self.assertIn(349, corners)
        self.assertTrue(any(abs(corner - 200) <= 5 for corner in corners), f"no bend in {corners}")

    def test_a_straight_line_has_only_its_two_ends(self):
        trace = np.full(400, np.nan)
        trace[50:350] = np.linspace(250, 100, 300)
        self.assertEqual(len(line_graph._corners(trace)), 2)


class ReadingOffTests(unittest.TestCase):
    def test_reads_a_value_for_every_category(self):
        trace = np.full(400, np.nan)
        trace[20:380] = 200.0
        axes = fake_axes([("a", 50), ("b", 200), ("c", 350)])
        points = line_graph._points_along(trace, "#000000", 1, axes)
        self.assertEqual([point.category for point in points], ["a", "b", "c"])
        self.assertEqual([point.value for point in points], [10.0, 10.0, 10.0])

    def test_a_category_the_line_never_reaches_has_no_value(self):
        trace = np.full(400, np.nan)
        trace[20:100] = 200.0
        axes = fake_axes([("a", 50), ("b", 200), ("c", 350)])
        points = line_graph._points_along(trace, "#000000", 1, axes)
        self.assertEqual([point.value for point in points], [10.0, None, None])
        self.assertIsNone(points[2].y)

    def test_reads_the_corner_rather_than_the_label_when_one_is_near(self):
        trace = np.full(400, np.nan)
        trace[100:300] = np.linspace(100, 200, 200)
        trace[300:360] = 200.0                               # the line flattens out at column 300
        axes = fake_axes([("a", 150), ("b", 320)])
        points = line_graph._points_along(trace, "#000000", 1, axes)
        self.assertAlmostEqual(points[1].y, 200, delta=3)

    def test_no_values_without_a_scale(self):
        trace = np.full(400, np.nan)
        trace[20:380] = 200.0
        points = line_graph._points_along(trace, "#000000", 1, fake_axes([("a", 50)], scale=None))
        self.assertIsNone(points[0].value)
        self.assertEqual(points[0].y, 200)


class NotALineChartTests(unittest.TestCase):
    def test_a_blank_image_has_no_lines(self):
        self.assertEqual(detect_line_graph(np.full((400, 500, 3), 255, np.uint8)), [])

    def test_a_pie_chart_has_no_lines(self):
        image = np.full((700, 900, 3), 255, np.uint8)
        cv2.circle(image, (450, 350), 250, (0, 0, 0), 4)
        self.assertEqual(detect_line_graph(image), [])

    def test_a_photo_of_a_pie_chart_has_no_lines(self):
        """A drawn circle is easy to turn down; a real photo of one is not.

        IMG_1294.jpg offers up a passable pair of axes from the pie's own outline and its
        legend, and nine unreadable labels below them, which was enough to be followed as
        three lines and reported as a line chart with 27 valueless rows.
        """
        photo = cv2.imread(str(SAMPLES / "IMG_1294.jpg"))
        if photo is None:
            self.skipTest("sample photo not available")
        self.assertEqual(detect_line_graph(preprocess(photo)), [])


class SamplePhotoTests(unittest.TestCase):
    """The whole pipeline, measured against the values read off the photos by hand.

    tests/check_accuracy.py prints the same comparison in full, which is the thing to run
    while changing the detection.
    """

    @classmethod
    def setUpClass(cls):
        cls.fixtures = json.loads(FIXTURES.read_text())
        cls.extracted = {}
        for chart in cls.fixtures["charts"]:
            photo = cv2.imread(str(SAMPLES / chart["photo"]))
            if photo is None:
                raise unittest.SkipTest(f"{chart['photo']} is missing from sample-images/")
            cls.extracted[chart["photo"]] = detect_line_graph(preprocess(photo))

    def lines_of(self, chart):
        """The extracted points of one chart, grouped by line number."""
        lines = {}
        for point in self.extracted[chart["photo"]]:
            lines.setdefault(point.line, []).append(point)
        return lines

    def test_every_chart_is_recognised_as_a_line_chart(self):
        for chart in self.fixtures["charts"]:
            with self.subTest(chart["photo"]):
                self.assertTrue(self.extracted[chart["photo"]], "nothing was extracted")

    def test_the_right_number_of_lines_and_points(self):
        for chart in self.fixtures["charts"]:
            with self.subTest(chart["photo"]):
                lines = self.lines_of(chart)
                self.assertEqual(len(lines), len(chart["lines"]))
                for points in lines.values():
                    self.assertEqual(len(points), len(chart["categories"]))

    def test_every_value_is_within_tolerance_of_the_hand_read_one(self):
        for chart in self.fixtures["charts"]:
            with self.subTest(chart["photo"]):
                low, high = chart["axis_range"]
                allowed = self.fixtures["tolerance"] * (high - low)
                lines = self.lines_of(chart)

                for colour, expected in chart["lines"].items():
                    values = self.closest_line(lines, expected)
                    self.assertIsNotNone(values, f"no line matches the {colour} one")
                    for category, true_value, value in zip(chart["categories"], expected, values):
                        self.assertIsNotNone(value, f"{colour} {category} was not read")
                        self.assertLessEqual(abs(value - true_value), allowed,
                                             f"{colour} {category}: read {value}, expected {true_value}")

    @staticmethod
    def closest_line(lines, expected):
        """The values of whichever extracted line is nearest to `expected`."""
        best, best_error = None, float("inf")
        for points in lines.values():
            values = [point.value for point in points]
            read = [(value, true) for value, true in zip(values, expected) if value is not None]
            error = np.mean([abs(value - true) for value, true in read]) if read else float("inf")
            if error < best_error:
                best, best_error = values, error
        return best


if __name__ == "__main__":
    unittest.main()

"""
Tests for working out which kind of chart is in a photo (chartx/detection/auto.py).

Run from the project root:

    python3 -m unittest discover tests -v
"""

import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from chartx.detection import auto
from chartx.detection.bar_graph import Bar
from chartx.detection.pie_graph import Sector
from chartx.main import preprocess

SAMPLES = Path(__file__).resolve().parent.parent / "sample-images"

# Two bars side by side, spanning x 100..300 and y 100..300 (a 200 x 200 area)
BARS = [Bar(1, 100, 100, 80, 200, 1.0, None, "#ff0000"), Bar(2, 220, 200, 80, 100, 0.5, None, "#00ff00")]
SECTORS = [Sector(1, 0.0, 180.0, 50.0, None, None), Sector(2, 180.0, 0.0, 50.0, None, None)]
IMAGE = np.zeros((10, 10, 3), np.uint8)


def detect_with(bars, sectors, circle):
    """Run detect_chart with the detectors answering these."""
    with mock.patch.object(auto.bar_graph, "detect_bar_graph", return_value=bars), \
            mock.patch.object(auto.pie_graph, "find_pie", return_value=(sectors, circle)):
        return auto.detect_chart(IMAGE)


class DetectChartTests(unittest.TestCase):
    def test_nothing_found(self):
        self.assertEqual(detect_with([], [], None), [])

    def test_only_bars(self):
        self.assertEqual(detect_with(BARS, [], None), [("bar", BARS)])

    def test_only_a_pie(self):
        self.assertEqual(detect_with([], SECTORS, (500, 500, 100)), [("pie", SECTORS)])

    def test_a_pie_with_bars_inside_its_circle_is_dropped(self):
        # the pie detector fitted a circle around the bars: it is the bar chart, misread
        self.assertEqual(detect_with(BARS, SECTORS, (200, 200, 150)), [("bar", BARS)])

    def test_a_pie_beside_the_bars_is_kept_and_the_bigger_chart_comes_first(self):
        small_pie = (700, 200, 50)            # area ~7,850 < bars' 40,000
        self.assertEqual(detect_with(BARS, SECTORS, small_pie), [("bar", BARS), ("pie", SECTORS)])
        big_pie = (900, 200, 150)             # area ~70,700 > 40,000
        self.assertEqual(detect_with(BARS, SECTORS, big_pie), [("pie", SECTORS), ("bar", BARS)])


class SamplePhotoTests(unittest.TestCase):
    """The real detectors on real photos (one of each case; the large photos are slow)."""

    def chart_types(self, name):
        found = auto.detect_chart(preprocess(cv2.imread(str(SAMPLES / name))))
        return [chart_type for chart_type, _ in found]

    def test_a_bar_chart_is_a_bar_chart_not_a_pie(self):
        # the pie detector alone finds 3 "sectors" here
        self.assertEqual(self.chart_types("barchart_test.jpg"), ["bar"])

    def test_a_pie_chart_is_a_pie_chart(self):
        self.assertEqual(self.chart_types("chartarena_pie_11.png"), ["pie"])

    def test_a_line_chart_is_not_a_bar_or_pie_chart(self):
        self.assertEqual(self.chart_types("20261027_120202.jpg"), [])

    def test_a_board_with_a_pie_and_a_bar_chart_finds_both(self):
        self.assertEqual(self.chart_types("new_3.jpg"), ["pie", "bar"])


if __name__ == "__main__":
    unittest.main()

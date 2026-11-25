"""
Tests for the bar graph detector (chartx/detection/bar_graph.py), mainly the bar labels.

Run from the project root:

    python3 -m unittest discover tests -v
"""

import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from chartx.detection import bar_graph, pie_graph
from chartx.main import preprocess

SAMPLE_BAR_CHART = str(Path(__file__).resolve().parent.parent / "sample-images" / "barchart_test.jpg")

# Three bars, 80 px wide with 20 px gaps: x from 100, 200 and 300
BOXES = np.array([[100, 50, 80, 200], [200, 100, 80, 150], [300, 150, 80, 100]])


def text_at(centre_x, text):
    """An OCR result: a 40 x 20 box centred on centre_x, with this text."""
    return ([[centre_x - 20, 0], [centre_x + 20, 0], [centre_x + 20, 20], [centre_x - 20, 20]], text, 0.99)


class AssignLabelsTests(unittest.TestCase):
    def test_each_label_goes_to_the_bar_above_it(self):
        results = [text_at(340, "Green"), text_at(140, "Red"), text_at(240, "Blue")]
        self.assertEqual(bar_graph._assign_labels(results, BOXES), ["Red", "Blue", "Green"])

    def test_a_bar_with_nothing_under_it_has_no_label(self):
        results = [text_at(140, "Red"), text_at(340, "Green")]
        self.assertEqual(bar_graph._assign_labels(results, BOXES), ["Red", None, "Green"])

    def test_text_under_no_bar_is_dropped(self):
        results = [text_at(40, "0"), text_at(140, "Red"), text_at(600, "Note")]
        self.assertEqual(bar_graph._assign_labels(results, BOXES), ["Red", None, None])

    def test_a_label_a_little_off_centre_still_counts(self):
        results = [text_at(95, "Red")]           # just left of the first bar's edge
        self.assertEqual(bar_graph._assign_labels(results, BOXES)[0], "Red")

    def test_several_words_under_one_bar_are_joined_left_to_right(self):
        results = [text_at(160, "Red"), text_at(120, "Dark")]
        self.assertEqual(bar_graph._assign_labels(results, BOXES)[0], "Dark Red")

    def test_spaces_around_the_text_are_removed(self):
        results = [text_at(140, "  Red ")]
        self.assertEqual(bar_graph._assign_labels(results, BOXES)[0], "Red")


class BarLabelsTests(unittest.TestCase):
    def test_reads_the_labels_under_the_sample_bar_chart(self):
        bars = bar_graph.detect_bar_graph(preprocess(cv2.imread(SAMPLE_BAR_CHART)))
        self.assertEqual([bar.label for bar in bars], ["Red", "Blue", "Green", "Yellow", "Pink"])

    def test_without_paddleocr_the_labels_are_none_and_a_warning_is_given(self):
        image = preprocess(cv2.imread(SAMPLE_BAR_CHART))
        with mock.patch.object(pie_graph, "_paddle", side_effect=ImportError):
            with self.assertWarnsRegex(UserWarning, "PaddleOCR"):
                bars = bar_graph.detect_bar_graph(image)
        self.assertEqual(len(bars), 5)
        self.assertTrue(all(bar.label is None for bar in bars))

    def test_bars_that_reach_the_bottom_of_the_photo_have_no_labels(self):
        image = np.zeros((300, 400, 3), np.uint8)
        boxes = np.array([[50, 100, 60, 197], [150, 150, 60, 147], [250, 200, 60, 97]])
        with mock.patch.object(pie_graph, "_read_text") as read_text:
            self.assertEqual(bar_graph._bar_labels(image, boxes), [None, None, None])
        read_text.assert_not_called()

    def test_horizontal_bars_are_not_given_labels(self):
        image = preprocess(cv2.imread(SAMPLE_BAR_CHART))
        with mock.patch.object(bar_graph, "_bar_labels") as read_labels:
            bars = bar_graph.detect_bar_graph(np.rot90(image, -1).copy(), horizontal=True)
        read_labels.assert_not_called()
        self.assertTrue(all(bar.label is None for bar in bars))


if __name__ == "__main__":
    unittest.main()

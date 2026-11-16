"""
Tests for chartx/main.py: the table printer, the preprocessing steps and the main run.

Run from the project root:

    python3 -m unittest discover tests -v
"""

import contextlib
import io
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np

from chartx import main as chartx_main
from chartx.detection.bar_graph import Bar

SAMPLE_BAR_CHART = str(Path(__file__).resolve().parent.parent / "sample-images" / "barchart_test.jpg")


def printed(function, *arguments):
    """Call the function and return what it printed (and the exit code if it stopped, else None)."""
    shown = io.StringIO()
    code = None
    with contextlib.redirect_stdout(shown):
        try:
            function(*arguments)
        except SystemExit as stop:
            code = stop.code
    return code, shown.getvalue()


@dataclass
class Row:
    name: str
    value: float | None


class PrintTableTests(unittest.TestCase):
    def test_no_rows_says_so(self):
        _, shown = printed(chartx_main.print_table, [])
        self.assertEqual(shown, "(no rows)\n")

    def test_prints_a_header_a_rule_and_one_line_per_row(self):
        _, shown = printed(chartx_main.print_table, [Row("a", 1.5), Row("b", 2.0)])
        lines = shown.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0].split(), ["name", "value"])
        self.assertTrue(set(lines[1].replace(" ", "")) == {"-"})
        self.assertEqual(lines[2].split(), ["a", "1.5"])
        self.assertEqual(lines[3].split(), ["b", "2.0"])

    def test_columns_line_up_with_the_widest_entry(self):
        _, shown = printed(chartx_main.print_table, [Row("short", 1), Row("a much longer name", 2)])
        lines = shown.splitlines()
        value_column = lines[0].index("value")
        for line in lines[2:]:
            self.assertNotEqual(line[value_column], " ", f"value not in its column: {line!r}")

    def test_missing_values_are_shown_as_a_dash(self):
        _, shown = printed(chartx_main.print_table, [Row("a", None)])
        self.assertEqual(shown.splitlines()[2].split(), ["a", "-"])

    def test_prints_bars_from_the_bar_graph_detector(self):
        bar = Bar(1, 10, 20, 30, 40, 1.0, 7.5, "#ff0000")
        _, shown = printed(chartx_main.print_table, [bar])
        self.assertIn("relative_height", shown.splitlines()[0])
        self.assertIn("#ff0000", shown.splitlines()[2])


class PreprocessTests(unittest.TestCase):
    def test_straightens_first_then_cleans_the_glare_off_the_straightened_image(self):
        photo = np.zeros((10, 10, 3), np.uint8)
        straightened = np.ones((8, 8, 3), np.uint8)
        cleaned = np.full((8, 8, 3), 2, np.uint8)
        with mock.patch.object(chartx_main, "deskew", return_value=mock.Mock(image=straightened)) as deskew, \
                mock.patch.object(chartx_main.glare, "clean_whiteboard", return_value=cleaned) as clean:
            result = chartx_main.preprocess(photo)
        deskew.assert_called_once_with(photo)
        clean.assert_called_once_with(straightened)
        self.assertIs(result, cleaned)

    def test_returns_a_colour_image_for_a_real_photo(self):
        photo = np.full((300, 400, 3), 235, np.uint8)
        result = chartx_main.preprocess(photo)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[2], 3)


class MainTests(unittest.TestCase):
    def setUp(self):
        # never open a plot window during the tests
        patcher = mock.patch.object(chartx_main.plt, "show")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_photo_that_cannot_be_read_is_reported(self):
        with mock.patch.object(chartx_main, "path", "no/such/photo.jpg"):
            code, shown = printed(chartx_main.main)
        self.assertEqual(code, 1)
        self.assertIn("no/such/photo.jpg", shown)

    def test_finds_the_bars_in_the_sample_bar_chart_and_shows_the_result(self):
        with mock.patch.object(chartx_main, "path", SAMPLE_BAR_CHART), \
                mock.patch.object(chartx_main.plt, "imshow") as imshow:
            code, shown = printed(chartx_main.main)
        self.assertIsNone(code)
        table = shown[shown.index("index"):].splitlines()
        self.assertEqual(len(table) - 2, 5, f"expected 5 bars in:\n{shown}")
        imshow.assert_called_once()

    def test_a_photo_with_no_bars_prints_an_empty_table(self):
        with mock.patch.object(chartx_main, "path", SAMPLE_BAR_CHART), \
                mock.patch.object(chartx_main.bar_graph, "detect_bar_graph", return_value=[]), \
                mock.patch.object(chartx_main.plt, "imshow"):
            code, shown = printed(chartx_main.main)
        self.assertIsNone(code)
        self.assertIn("(no rows)", shown)


if __name__ == "__main__":
    unittest.main()

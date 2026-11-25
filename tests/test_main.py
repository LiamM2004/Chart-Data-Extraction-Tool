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

import cv2
import numpy as np

from chartx import main as chartx_main
from chartx.detection.bar_graph import Bar
from chartx.detection.line_graph import LinePoint
from chartx.detection.pie_graph import Sector

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

    def test_prints_points_from_the_line_graph_detector(self):
        point = LinePoint(1, "Jan", 10, 20, 7.5, "#0000ff")
        _, shown = printed(chartx_main.print_table, [point])
        self.assertIn("category", shown.splitlines()[0])
        self.assertIn("Jan", shown.splitlines()[2])


class DrawBarBoxesTests(unittest.TestCase):
    def test_draws_one_box_and_one_label_per_bar_where_the_bar_is(self):
        bars = [Bar(1, 10, 20, 30, 40, 1.0, 7.5, "#ff0000"), Bar(2, 50, 30, 30, 30, 0.75, None, "#00ff00")]
        ax = mock.Mock()
        chartx_main.draw_bar_boxes(ax, bars)
        boxes = [call.args[0] for call in ax.add_patch.call_args_list]
        self.assertEqual([(b.get_x(), b.get_y(), b.get_width(), b.get_height()) for b in boxes],
                         [(10, 20, 30, 40), (50, 30, 30, 30)])
        self.assertEqual([call.args[2] for call in ax.text.call_args_list], ["1", "2"])

    def test_no_bars_draws_nothing(self):
        ax = mock.Mock()
        chartx_main.draw_bar_boxes(ax, [])
        ax.add_patch.assert_not_called()


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

    def test_no_photo_given_prints_the_usage(self):
        with mock.patch("sys.argv", ["chartx"]):
            code, shown = printed(chartx_main.main)
        self.assertEqual(code, 1)
        self.assertIn("Usage", shown)

    def test_no_photo_after_the_chart_type_prints_the_usage(self):
        with mock.patch("sys.argv", ["chartx", "bar"]):
            code, shown = printed(chartx_main.main)
        self.assertEqual(code, 1)
        self.assertIn("Usage", shown)

    def test_an_unknown_chart_type_prints_the_usage_listing_the_types(self):
        with mock.patch("sys.argv", ["chartx", "scatter", SAMPLE_BAR_CHART]):
            code, shown = printed(chartx_main.main)
        self.assertEqual(code, 1)
        for chart_type in chartx_main.DETECTORS:
            self.assertIn(chart_type, shown)

    def test_a_photo_that_cannot_be_read_is_reported(self):
        with mock.patch("sys.argv", ["chartx", "bar", "no/such/photo.jpg"]):
            code, shown = printed(chartx_main.main)
        self.assertEqual(code, 1)
        self.assertIn("no/such/photo.jpg", shown)

    def test_finds_the_bars_in_the_sample_bar_chart_and_boxes_them(self):
        with mock.patch("sys.argv", ["chartx", "bar", SAMPLE_BAR_CHART]), \
                mock.patch.object(chartx_main.plt, "imshow") as imshow, \
                mock.patch.object(chartx_main, "draw_bar_boxes") as draw:
            code, shown = printed(chartx_main.main)
        self.assertIsNone(code)
        table = shown[shown.index("index"):].splitlines()
        self.assertEqual(len(table) - 2, 5, f"expected 5 bars in:\n{shown}")
        self.assertEqual(table[0].split()[-1], "label")
        self.assertEqual([row.split()[-1] for row in table[2:]], ["Red", "Blue", "Green", "Yellow", "Pink"])
        imshow.assert_called_once()
        self.assertEqual(len(draw.call_args.args[1]), 5)

    def test_line_uses_the_line_detector_and_draws_no_bar_boxes(self):
        point = LinePoint(1, "Jan", 10, 20, 7.5, "#0000ff")
        with mock.patch("sys.argv", ["chartx", "line", SAMPLE_BAR_CHART]), \
                mock.patch.object(chartx_main.line_graph, "detect_line_graph", return_value=[point]) as line, \
                mock.patch.object(chartx_main.bar_graph, "detect_bar_graph") as bar, \
                mock.patch.object(chartx_main.plt, "imshow"), \
                mock.patch.object(chartx_main, "draw_bar_boxes") as draw:
            code, shown = printed(chartx_main.main)
        self.assertIsNone(code)
        line.assert_called_once()
        bar.assert_not_called()
        draw.assert_not_called()
        self.assertIn("Jan", shown)

    def test_pie_uses_the_pie_detector_and_draws_no_bar_boxes(self):
        sector = Sector(1, 0.0, 90.0, 25.0, None, "Red")
        with mock.patch("sys.argv", ["chartx", "pie", SAMPLE_BAR_CHART]), \
                mock.patch.object(chartx_main.pie_graph, "detect_pie_graph", return_value=[sector]) as pie, \
                mock.patch.object(chartx_main.bar_graph, "detect_bar_graph") as bar, \
                mock.patch.object(chartx_main.plt, "imshow"), \
                mock.patch.object(chartx_main, "draw_bar_boxes") as draw:
            code, shown = printed(chartx_main.main)
        self.assertIsNone(code)
        pie.assert_called_once()
        bar.assert_not_called()
        draw.assert_not_called()
        self.assertIn("start_angle", shown)
        self.assertIn("Red", shown)

    def test_a_photo_with_no_chart_says_so_and_prints_an_empty_table(self):
        with mock.patch("sys.argv", ["chartx", "bar", SAMPLE_BAR_CHART]), \
                mock.patch.object(chartx_main.bar_graph, "detect_bar_graph", return_value=[]), \
                mock.patch.object(chartx_main.line_graph, "detect_line_graph", return_value=[]), \
                mock.patch.object(chartx_main.plt, "imshow"):
            code, shown = printed(chartx_main.main)
        self.assertIsNone(code)
        self.assertIn("No bar chart found", shown)
        self.assertIn("(no rows)", shown)


class CornerOptionTests(unittest.TestCase):
    """The board's corners can be placed by hand when deskew gets them wrong."""

    def setUp(self):
        for name in ("show", "imshow"):
            patcher = mock.patch.object(chartx_main.plt, name)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(chartx_main.bar_graph, "detect_bar_graph", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_with(self, *words):
        with mock.patch("sys.argv", ["chartx", "bar", SAMPLE_BAR_CHART, *words]), \
                mock.patch.object(chartx_main, "select_corners_manually",
                                  return_value=None) as window:
            code, shown = printed(chartx_main.main)
        return window, code, shown

    def test_the_corner_window_opens_only_when_it_is_asked_for(self):
        window, _, _ = self.run_with()
        window.assert_not_called()

    def test_manual_opens_the_corner_window(self):
        window, code, _ = self.run_with("--manual")
        window.assert_called_once()
        self.assertIsNone(code)

    def test_manual_opens_the_window_even_when_deskew_is_happy(self):
        """The photo that most needs the corners setting by hand is the one deskew was sure
        about and still got wrong, so --manual must not wait to be asked."""
        photo = cv2.imread(SAMPLE_BAR_CHART)
        sure = mock.Mock(image=photo, corners=None, needs_review=False, warnings=[])
        with mock.patch("sys.argv", ["chartx", "bar", SAMPLE_BAR_CHART, "--manual"]), \
                mock.patch.object(chartx_main, "deskew", return_value=sure), \
                mock.patch.object(chartx_main, "select_corners_manually",
                                  return_value=None) as window:
            printed(chartx_main.main)
        window.assert_called_once()

    def test_corners_given_on_the_command_line_are_used_without_a_window(self):
        with mock.patch("sys.argv", ["chartx", "bar", SAMPLE_BAR_CHART, "--corners",
                                     "10,20", "900,15", "920,700", "5,690"]), \
                mock.patch.object(chartx_main, "select_corners_manually") as window, \
                mock.patch.object(chartx_main, "deskew", wraps=chartx_main.deskew) as deskew:
            code, _ = printed(chartx_main.main)
        self.assertIsNone(code)
        window.assert_not_called()
        self.assertEqual(deskew.call_args.kwargs["corners"].shape, (4, 2))

    def test_too_few_corners_is_an_error_rather_than_quietly_ignored(self):
        with mock.patch("sys.argv", ["chartx", "bar", SAMPLE_BAR_CHART, "--corners", "1,2", "3,4"]):
            code, shown = printed(chartx_main.main)
        self.assertEqual(code, 1)
        self.assertIn("four x,y points", shown)

    def test_an_unknown_option_is_an_error_rather_than_quietly_ignored(self):
        with mock.patch("sys.argv", ["chartx", "bar", SAMPLE_BAR_CHART, "--manul"]):
            code, shown = printed(chartx_main.main)
        self.assertEqual(code, 1)
        self.assertIn("--manul", shown)

    def test_what_deskew_warns_about_reaches_the_user(self):
        _, _, shown = self.run_with()
        self.assertIn("note:", shown, "deskew's warnings should not be swallowed")


if __name__ == "__main__":
    unittest.main()

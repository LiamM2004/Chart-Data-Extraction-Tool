"""
Tests for chartx/detection/axes.py: finding the axes, grouping the writing into labels,
and fitting the y-axis scale.

Run from the project root:

    python3 -m unittest discover tests -v
"""

import unittest

import cv2
import numpy as np

from chartx.detection import axes as axes_module
from chartx.detection.axes import ValueScale, find_axes

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def blank(height=700, width=900):
    return np.full((height, width, 3), 255, np.uint8)


def with_axes(image, origin=(150, 600), top=100, right=800):
    """Draw an "L" of axes on the image and return it."""
    cv2.line(image, (origin[0], top), origin, BLACK, 3)
    cv2.line(image, origin, (right, origin[1]), BLACK, 3)
    return image


def digits(image, text, centre, scale=1.4):
    """Write `text` centred on `centre`, roughly the size axis labels are drawn at."""
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    cv2.putText(image, text, (centre[0] - width // 2, centre[1] + height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 3)
    return image


def boxes(*rectangles):
    return np.array(rectangles, int)


class AxisStrokeTests(unittest.TestCase):
    def test_finds_the_corner_where_the_two_axes_meet(self):
        found = axes_module._axis_strokes(with_axes(blank()))
        self.assertIsNotNone(found)
        _, _, origin = found
        self.assertAlmostEqual(origin[0], 150, delta=6)
        self.assertAlmostEqual(origin[1], 600, delta=6)

    def test_prefers_the_axes_over_the_longer_edges_of_the_board(self):
        image = blank()
        cv2.rectangle(image, (6, 6), (894, 694), BLACK, 3)        # the board, longer than the axes
        found = axes_module._axis_strokes(with_axes(image))
        self.assertIsNotNone(found)
        _, _, origin = found
        self.assertAlmostEqual(origin[0], 150, delta=6)
        self.assertAlmostEqual(origin[1], 600, delta=6)

    def test_nothing_to_find_in_a_blank_image(self):
        self.assertIsNone(axes_module._axis_strokes(blank()))

    def test_a_cross_is_not_a_pair_of_axes(self):
        image = blank()
        cv2.line(image, (450, 100), (450, 600), BLACK, 3)         # they meet in the middle,
        cv2.line(image, (150, 350), (800, 350), BLACK, 3)         # not at either one's end
        self.assertIsNone(axes_module._axis_strokes(image))

    def test_find_axes_gives_up_on_a_chart_with_no_axes(self):
        image = blank()
        cv2.circle(image, (450, 350), 250, BLACK, 4)              # a pie chart
        self.assertIsNone(find_axes(image))

    def test_the_plot_area_is_right_of_the_y_axis_and_above_the_baseline(self):
        found = find_axes(with_axes(blank()))
        self.assertIsNotNone(found)
        x, y, width, height = found.plot_area
        self.assertGreaterEqual(x, 145)
        self.assertLessEqual(y, 110)
        self.assertGreater(width, 500)
        self.assertGreater(height, 400)


class MergeCollinearTests(unittest.TestCase):
    def test_joins_two_pieces_of_one_stroke_into_its_whole_length(self):
        pieces = np.array([[100, 50, 100, 300], [101, 290, 101, 600]])
        merged = axes_module._merge_collinear(pieces, upright=True)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0][1], 50, delta=2)
        self.assertAlmostEqual(merged[0][3], 600, delta=2)

    def test_keeps_strokes_that_are_well_apart_separate(self):
        apart = np.array([[100, 50, 100, 600], [400, 50, 400, 600]])
        self.assertEqual(len(axes_module._merge_collinear(apart, upright=True)), 2)

    def test_upright_strokes_come_back_running_top_to_bottom(self):
        upside_down = np.array([[100, 600, 100, 50]])
        merged = axes_module._merge_collinear(upside_down, upright=True)
        self.assertLess(merged[0][1], merged[0][3])


class ValueScaleTests(unittest.TestCase):
    def test_reads_a_value_off_a_pixel_row(self):
        scale = ValueScale(per_row=-0.5, at_row_zero=300.0)
        self.assertEqual(scale.value(400), 100.0)
        self.assertEqual(scale.row(100.0), 400.0)

    def test_fits_a_scale_through_labels_that_agree(self):
        rows = np.array([500.0, 400.0, 300.0, 200.0])
        readings = [[0.0], [10.0], [20.0], [30.0]]
        scale = axes_module._fit_value_scale(readings, rows, row_gap=100.0)
        self.assertIsNotNone(scale)
        self.assertAlmostEqual(scale.value(500), 0.0, delta=0.5)
        self.assertAlmostEqual(scale.value(200), 30.0, delta=0.5)

    def test_ignores_a_label_that_was_misread(self):
        rows = np.array([500.0, 400.0, 300.0, 200.0])
        readings = [[0.0], [10.0], [90.0], [30.0]]          # 20 misread as 90
        scale = axes_module._fit_value_scale(readings, rows, row_gap=100.0)
        self.assertIsNotNone(scale)
        self.assertAlmostEqual(scale.value(300), 20.0, delta=0.5)

    def test_uses_whichever_reading_of_a_label_fits_the_others(self):
        rows = np.array([500.0, 400.0, 300.0, 200.0])
        readings = [[0.0], [70.0, 10.0], [20.0], [30.0]]    # 10 was also read as 70
        scale = axes_module._fit_value_scale(readings, rows, row_gap=100.0)
        self.assertIsNotNone(scale)
        self.assertAlmostEqual(scale.value(400), 10.0, delta=0.5)

    def test_no_scale_when_only_one_label_could_be_read(self):
        rows = np.array([500.0, 400.0])
        self.assertIsNone(axes_module._fit_value_scale([[10.0], []], rows, row_gap=100.0))

    def test_no_scale_from_labels_that_climb_the_wrong_way(self):
        rows = np.array([500.0, 400.0, 300.0])
        readings = [[30.0], [20.0], [10.0]]                 # bigger numbers lower down
        self.assertIsNone(axes_module._fit_value_scale(readings, rows, row_gap=100.0))


class LabelGroupingTests(unittest.TestCase):
    def test_joins_the_pieces_of_one_label_but_not_its_neighbours(self):
        pieces = boxes((10, 0, 8, 20), (20, 0, 18, 20),     # "1" and "990", 2px apart
                       (100, 0, 30, 20), (190, 0, 30, 20))
        joined = axes_module._joined_into_labels(pieces, along=0)
        self.assertEqual(len(joined), 3)
        self.assertEqual(tuple(joined[0]), (10, 0, 28, 20))

    def test_leaves_labels_that_are_already_whole_alone(self):
        whole = boxes((10, 0, 30, 20), (100, 0, 30, 20), (190, 0, 30, 20))
        self.assertEqual(len(axes_module._joined_into_labels(whole, along=0)), 3)

    def test_splits_a_box_twice_the_usual_width_into_two_labels(self):
        run_together = boxes((10, 0, 30, 20), (100, 0, 30, 20), (190, 0, 62, 20))
        split = axes_module._split_double_labels(run_together, along=0)
        self.assertEqual(len(split), 4)
        self.assertEqual([box[0] for box in split], [10, 100, 190, 221])

    def test_does_not_split_a_label_that_is_merely_a_bit_wide(self):
        slightly_wide = boxes((10, 0, 30, 20), (100, 0, 30, 20), (190, 0, 44, 20))
        self.assertEqual(len(axes_module._split_double_labels(slightly_wide, along=0)), 3)


class NamedSequenceTests(unittest.TestCase):
    def test_names_the_labels_that_were_misread_from_the_ones_that_were_not(self):
        read = ["Sen", "Feb", "Mar", "Apr", "flosy", "So"]     # what the sample photo gives
        self.assertEqual(axes_module._named_in_sequence(read),
                         ["Jan", "Feb", "Mar", "Apr", "May", "Jun"])

    def test_picks_up_a_run_from_the_middle_of_a_sequence(self):
        self.assertEqual(axes_module._named_in_sequence(["Jvl", "Aug", "Sep", "0ct"]),
                         ["Jul", "Aug", "Sep", "Oct"])

    def test_recognises_a_sequence_written_out_in_full(self):
        named = axes_module._named_in_sequence(["Janvary", "February", "Marcb"])
        self.assertEqual(named, ["January", "February", "March"])

    def test_leaves_labels_alone_when_they_are_not_a_familiar_sequence(self):
        self.assertIsNone(axes_module._named_in_sequence(["Red", "Green", "Blue", "Pink"]))

    def test_one_label_that_happens_to_match_is_not_enough_to_rename_the_rest(self):
        self.assertIsNone(axes_module._named_in_sequence(["Mar", "Widgets", "Sprockets"]))

    def test_a_sequence_shorter_than_the_labels_is_not_used(self):
        self.assertIsNone(axes_module._named_in_sequence(["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]))


class ReadingLabelsTests(unittest.TestCase):
    """These read both axes. PaddleOCR does the reading and `pip` installs it, so there is no
    separate program to check for and nothing to skip."""

    def test_reads_the_scale_off_a_drawn_chart(self):
        image = with_axes(blank(1000, 1100), origin=(260, 820), top=120, right=1000)
        for number, row in zip(("40", "30", "20", "10", "0"), (160, 325, 490, 655, 820)):
            digits(image, number, (185, row))
        for number, column in zip(("1", "2", "3", "4"), (400, 560, 720, 880)):
            digits(image, number, (column, 880))

        found = find_axes(image)
        self.assertIsNotNone(found)
        self.assertIsNotNone(found.scale, "the y-axis labels were not read")
        self.assertAlmostEqual(found.scale.value(820), 0.0, delta=2.0)
        self.assertAlmostEqual(found.scale.value(490), 20.0, delta=2.0)
        self.assertEqual([category.text for category in found.categories], ["1", "2", "3", "4"])


if __name__ == "__main__":
    unittest.main()

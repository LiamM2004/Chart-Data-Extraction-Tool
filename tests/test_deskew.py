"""
Regression tests for the preprocessing stage.

Run from the project root:

    python3 -m unittest discover tests -v

The synthetic tests build geometry with a known answer, so they need no photos.
The photo tests run over sample-images/ and are skipped if that folder is
missing; the expected outcomes there are what the detectors were tuned on, so a
failure means a behaviour change to look at, not necessarily a bug.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from chartx.preprocessing.deskew import three_edge_board
from chartx.preprocessing.deskew.board import estimate_aspect_ratio
from chartx.preprocessing.deskew.corner_editor import CornerEditor
from chartx.preprocessing.deskew import deskew
from chartx.preprocessing.deskew.ellipse import find_pie_ellipse, pie_to_circle_transform
from chartx.preprocessing.deskew.geometry import camera_matrix, find_edge_lines, is_a_usable_quad, order_corners
from chartx.preprocessing.deskew.rotation import find_long_straight_lines, tilt_from_chart_axes, tilt_from_writing
from chartx.preprocessing.deskew.straighten import _PhotoCorrection
from chartx.preprocessing.deskew.three_edge_board import find_three_edge_board

SAMPLES = Path(__file__).resolve().parent.parent / "sample-images"


def project_rectangle(width, height, shape, focal_frac=0.72, rvec=(0.17, 0.44, 0.0)):
    """Image corners (TL, TR, BR, BL) of a width x height rectangle seen by a known camera."""
    h, w = shape[:2]
    f = focal_frac * max(h, w)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    obj = np.array([[-width / 2, -height / 2, 0], [width / 2, -height / 2, 0],
                    [width / 2, height / 2, 0], [-width / 2, height / 2, 0]], np.float64)
    tvec = np.array([0.0, 0.0, f * width / (0.5 * w)])        # board spans ~half the photo
    pts, _ = cv2.projectPoints(obj, np.array(rvec, np.float64), tvec, K, None)
    return pts.reshape(4, 2)


class GeometryTests(unittest.TestCase):
    def test_order_corners_names_an_upright_quad_correctly_and_never_repeats_one_when_turned(self):
        shuffled = [[90, 80], [10, 10], [10, 80], [90, 10]]
        np.testing.assert_array_equal(order_corners(shuffled), [[10, 10], [90, 10], [90, 80], [10, 80]])
        diamond = order_corners([(0, 50), (50, 0), (100, 50), (50, 100)])         # a board turned 45 degrees
        self.assertEqual(len({tuple(p) for p in diamond.tolist()}), 4)
        rectangle = np.array([[-300, -200], [300, -200], [300, 200], [-300, 200]], np.float64)
        for degrees in range(0, 360, 15):
            a = np.radians(degrees)
            turned = rectangle @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) + [500, 400]
            ordered = order_corners(turned)
            self.assertEqual(len({tuple(p) for p in ordered.tolist()}), 4, f"turned by {degrees}")
            sides = np.linalg.norm(ordered - np.roll(ordered, -1, axis=0), axis=1)     # going round the shape
            np.testing.assert_allclose(sorted(sides), [400, 400, 600, 600], atol=1)    # real sides, no diagonal
            self.assertAlmostEqual(sides[0], sides[2], delta=1)
            self.assertEqual(ordered[0].sum(), ordered.sum(axis=1).min(), f"turned by {degrees}: should start top-left")
        lopsided = order_corners([(60, 0), (100, 40), (40, 100), (0, 50)])
        np.testing.assert_array_equal(lopsided[0], [0, 50])           # starts at the corner nearest the top-left

    def test_a_usable_quad_is_four_different_points_in_a_convex_shape(self):
        shape = (1000, 1600, 3)
        self.assertTrue(is_a_usable_quad([[100, 100], [900, 120], [880, 700], [90, 650]], shape))
        self.assertTrue(is_a_usable_quad([[900, 120], [100, 100], [880, 700], [90, 650]], shape))  # crossed order
        self.assertFalse(is_a_usable_quad([[500, 500], [501, 500], [501, 501], [500, 501]], shape))   # collapsed
        self.assertFalse(is_a_usable_quad([[0, 0], [400, 0], [800, 0], [1200, 0]], shape))             # in a line
        self.assertFalse(is_a_usable_quad([[0, 0], [800, 0], [800, 800], [300, 300]], shape))          # a dent
        self.assertFalse(is_a_usable_quad([[0, 0], [800, 0], [800, np.nan], [0, 800]], shape))

    def test_aspect_ratio_recovered_for_known_camera(self):
        shape = (2252, 4000, 3)
        for true_ratio in (1.0, 1.5, 2.0):
            corners = project_rectangle(true_ratio, 1.0, shape)
            self.assertAlmostEqual(estimate_aspect_ratio(corners, shape), true_ratio,
                                   delta=0.03 * true_ratio)

class EdgeLineTests(unittest.TestCase):
    def test_one_near_vertical_edge_is_reported_once(self):
        # A line written at theta ~0 can come back again as (-rho, theta ~pi): the same line twice.
        edges = np.zeros((600, 800), np.uint8)
        cv2.line(edges, (400, 20), (402, 580), 255, 3)
        lines = find_edge_lines(edges, vertical=True, k=8)
        copies = [line for line in lines if abs(abs(line[0]) - 400) < 12]
        self.assertEqual(len(copies), 1, lines)

class PieTests(unittest.TestCase):
    def make_pie(self, axes):
        img = np.full((900, 1200, 3), 235, np.uint8)
        cv2.ellipse(img, (600, 450), axes, 0, 0, 360, (20, 20, 20), 3)
        cv2.putText(img, "25%", (900, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
        return img

    def test_finds_stretched_pie_and_makes_it_round(self):
        found = find_pie_ellipse(self.make_pie((260, 215)))
        self.assertIsNotNone(found)
        self.assertAlmostEqual(max(found[1]) / min(found[1]), 260 / 215, delta=0.05)
        photo = self.make_pie((260, 215))
        matrix, size, stretch = pie_to_circle_transform(found, photo.shape)
        fixed = cv2.warpPerspective(photo, matrix, size, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        self.assertAlmostEqual(stretch, 260 / 215, delta=0.05)
        refit = find_pie_ellipse(fixed)
        self.assertIsNotNone(refit)
        self.assertLess(max(refit[1]) / min(refit[1]), 1.03)

    def test_blank_image_has_no_pie(self):
        self.assertIsNone(find_pie_ellipse(np.full((900, 1200, 3), 235, np.uint8)))


    def test_a_very_oval_shape_is_not_a_pie(self):
        self.assertIsNone(find_pie_ellipse(self.make_pie((260, 130))))

    def test_a_partial_outline_is_not_a_pie(self):
        photo = np.full((900, 1200, 3), 235, np.uint8)
        cv2.ellipse(photo, (600, 450), (230, 230), 0, 0, 200, (20, 20, 20), 3)        # a little over half a circle
        self.assertIsNone(find_pie_ellipse(photo))

    def test_the_remembered_pie_follows_the_stretch(self):
        # After the pie is stretched round, the pie we remember must be where the pie really is now.
        whole_photo = np.array([[0, 0], [1200, 0], [1200, 900], [0, 900]], np.float32)
        correction = _PhotoCorrection(self.make_pie((260, 210)), whole_photo, 0.72)
        self.assertGreater(correction.round_off_the_pie(), 1.15)
        (fx, fy), (d0, d1), _ = find_pie_ellipse(correction.preview)
        (rx, ry), (diameter, _), _ = correction._pie
        self.assertLess(np.hypot(rx - fx, ry - fy), 10)
        self.assertAlmostEqual(diameter / max(d0, d1), 1.0, delta=0.05)


def draw_axes(img, origin, x_end, y_end, thickness=4):
    cv2.line(img, origin, (x_end, origin[1]), (20, 20, 20), thickness)
    cv2.line(img, origin, (origin[0], y_end), (20, 20, 20), thickness)


class EndToEndGeometryTests(unittest.TestCase):
    def test_synthetic_board_comes_out_upright_and_the_right_shape(self):
        # A 1.5:1 board carrying a chart, photographed at an angle by a known camera.
        board = np.full((1000, 1500, 3), 240, np.uint8)
        draw_axes(board, (300, 800), 1100, 200)
        cv2.rectangle(board, (450, 500), (550, 796), (200, 60, 60), -1)
        cv2.rectangle(board, (650, 400), (750, 796), (60, 60, 200), -1)

        shape = (2252, 4000, 3)
        corners = project_rectangle(1.5, 1.0, shape)
        self.assertTrue(np.all((corners > 0) & (corners < [shape[1], shape[0]])), "test setup: board off-photo")
        board_corners = np.array([[0, 0], [1500, 0], [1500, 1000], [0, 1000]], np.float32)
        M = cv2.getPerspectiveTransform(board_corners, corners.astype(np.float32))
        photo = cv2.warpPerspective(board, M, (shape[1], shape[0]), borderValue=(120, 120, 120))

        result = deskew(photo, corners=corners)
        h, w = result.image.shape[:2]
        self.assertAlmostEqual(w / h, 1.5, delta=0.06)
        self.assertLess(abs(result.rotation_deg), 0.5)
        remaining_tilt = tilt_from_chart_axes(find_long_straight_lines(result.image))
        self.assertIsNotNone(remaining_tilt, "no axes found in the output")
        self.assertLess(abs(remaining_tilt), 1.5, "axes not horizontal/vertical after deskew")


    def test_unusable_hand_given_corners_are_flagged_instead_of_crashing(self):
        photo = np.full((1000, 1600, 3), 130, np.uint8)
        result = deskew(photo, corners=np.array([[500, 500], [501, 500], [501, 501], [500, 501]], np.float32))
        self.assertEqual(result.corner_source, "none")
        self.assertTrue(result.needs_review)
        self.assertIs(result.image, photo)
        self.assertTrue(any("usable four-sided shape" in w for w in result.warnings), result.warnings)

    def test_grayscale_and_alpha_photos_are_accepted(self):
        photo = np.full((1000, 1600, 3), 130, np.uint8)
        for converted in (cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), cv2.cvtColor(photo, cv2.COLOR_BGR2BGRA)):
            result = deskew(converted)
            self.assertEqual(result.image.shape, photo.shape)

    def test_things_that_are_not_photos_are_rejected_with_a_clear_message(self):
        for bad in (None, "photo.jpg", np.zeros((0, 0, 3), np.uint8), np.zeros((50, 50, 3), np.float32),
                    np.zeros((50, 50, 5), np.uint8), np.zeros((5, 5, 5, 3), np.uint8)):
            with self.assertRaisesRegex(ValueError, "deskew\\(\\) needs"):
                deskew(bad)

class FillTests(unittest.TestCase):
    def test_area_outside_the_photo_is_filled_with_the_board_colour(self):
        photo = np.full((900, 1200, 3), 235, np.uint8)
        draw_axes(photo, (200, 700), 1000, 150, 5)
        past_the_edge = np.array([[0, 0], [1500, 0], [1500, 900], [0, 900]], np.float32)   # 300 px past the right side
        out = deskew(photo, corners=past_the_edge).image
        outside = out[int(0.3 * out.shape[0]):int(0.7 * out.shape[0]), int(0.9 * out.shape[1]):]
        self.assertGreater(outside.mean(), 200)


class RotationTests(unittest.TestCase):
    def test_a_long_stroke_with_no_perpendicular_partner_is_not_an_axis(self):
        # No L, so no tilt estimate: levelling on a lone text line would be a guess.
        stroke = np.array([[100, 800, 1100, 780]])           # ~1.1 deg off level, nothing joins it
        self.assertIsNone(tilt_from_chart_axes(stroke))
        pie_like = np.array([[100, 800, 1100, 780], [200, 300, 900, 310]])   # two parallel strokes
        self.assertIsNone(tilt_from_chart_axes(pie_like))

    def test_axes_win_over_a_longer_stroke_that_does_not_join_them(self):
        axes = np.array([[300, 900, 900, 900], [300, 900, 300, 400]])        # level L, 600 + 500 long
        long_stroke = np.array([[100, 1200, 1500, 1170]])                    # 1.2 deg off, longer than either
        tilt = tilt_from_chart_axes(np.vstack([axes, long_stroke]))
        self.assertAlmostEqual(tilt, 0.0, delta=0.1)

    def test_the_tilt_is_the_length_weighted_average_of_the_two_axes(self):
        # a 600 px axis leaning 2 degrees and a 400 px axis dead upright, meeting in a corner
        lean = np.radians(2.0)
        axes = np.array([[100, 500, 100 + int(600 * np.cos(lean)), 500 - int(600 * np.sin(lean))],
                         [100, 500, 100, 100]])
        self.assertAlmostEqual(tilt_from_chart_axes(axes), (2.0 * 600 + 0.0 * 400) / 1000, delta=0.1)

    def test_two_lines_far_from_square_are_not_taken_as_axes(self):
        lean = np.radians(10)                              # one leaning up 10 degrees, the other the opposite way
        axes = np.array([[100, 500, 100 + int(600 * np.cos(lean)), 500 - int(600 * np.sin(lean))],
                         [100, 500, 100 + int(400 * np.sin(lean)), 500 - int(400 * np.cos(lean))]])
        self.assertIsNone(tilt_from_chart_axes(axes))

    def test_a_frame_touching_the_edge_of_the_photo_is_not_mistaken_for_axes(self):
        chart = np.full((900, 1200, 3), 240, np.uint8)
        draw_axes(chart, (300, 700), 1000, 200)
        photo = cv2.warpAffine(chart, cv2.getRotationMatrix2D((600, 450), 4.0, 1.0), (1200, 900),
                               borderValue=(240, 240, 240))
        cv2.rectangle(photo, (0, 0), (1199, 899), (20, 20, 20), 8)             # a level frame, longer than the axes
        self.assertAlmostEqual(tilt_from_chart_axes(find_long_straight_lines(photo)), 4.0, delta=0.5)


class TextTiltTests(unittest.TestCase):
    """A pie has no axes, so it is levelled on its writing instead."""
    PIE = ((600, 400), (460, 460), 0)

    def make_page(self, with_pie, tilt):
        img = np.full((800, 1200, 3), 240, np.uint8)
        if with_pie:
            cv2.ellipse(img, (600, 400), (230, 230), 0, 0, 360, (20, 20, 20), 4)
            for x in range(420, 800, 25):                      # diagonal hatching: points every which way
                cv2.line(img, (x, 250), (x - 90, 550), (20, 20, 20), 2)
        for i, text in enumerate(["Tell them I can't hear them", "Move closer so I can hear",
                                  "Repeatedly shout WHAT?", "Laugh and hope it wasn't a question"]):
            cv2.putText(img, text, (150, 90 + 45 * i) if not with_pie else (700, 640 + 40 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 3)
        cv2.putText(img, "What I do", (90, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
        M = cv2.getRotationMatrix2D((600, 400), tilt, 1.0)                 # CCW positive
        return cv2.warpAffine(img, M, (1200, 800), borderValue=(240, 240, 240))

    def test_strokes_inside_a_pie_are_not_mistaken_for_axes(self):
        # A level "L" of strokes sits inside the pie (hatching can look like this). It must not
        # outvote the writing, which says the page is tilted by 5 degrees.
        photo = self.make_page(True, 5.0)
        cv2.line(photo, (450, 520), (750, 520), (20, 20, 20), 4)
        cv2.line(photo, (450, 520), (450, 260), (20, 20, 20), 4)
        corners = np.array([[0, 0], [1200, 0], [1200, 800], [0, 800]], np.float32)
        self.assertAlmostEqual(deskew(photo, corners=corners).rotation_deg, -5.0, delta=0.7)

    def test_the_pie_is_found_once_and_remembered_between_steps(self):
        # Pie detection is borderline on some photos: it can pass for one step and fail for the
        # next. The pie found for rounding must still be the one ignored when looking for axes.
        straighten = sys.modules["chartx.preprocessing.deskew.straighten"]
        real, searches = straighten.find_pie_ellipse, []

        def finds_the_pie_only_the_first_time(img, *args, **kwargs):
            searches.append(1)
            return real(img, *args, **kwargs) if len(searches) == 1 else None

        photo = self.make_page(True, 5.0)
        cv2.line(photo, (450, 520), (750, 520), (20, 20, 20), 4)          # a level "L" inside the pie
        cv2.line(photo, (450, 520), (450, 260), (20, 20, 20), 4)
        corners = np.array([[0, 0], [1200, 0], [1200, 800], [0, 800]], np.float32)
        with mock.patch.object(straighten, "find_pie_ellipse", finds_the_pie_only_the_first_time):
            result = deskew(photo, corners=corners)
        self.assertEqual(len(searches), 1)
        self.assertAlmostEqual(result.rotation_deg, -5.0, delta=0.7)

    def test_a_tilted_pie_chart_comes_out_level(self):
        whole_photo = np.array([[0, 0], [1200, 0], [1200, 800], [0, 800]], np.float32)
        for tilt in (-4.0, 5.0):
            image = deskew(self.make_page(True, tilt), corners=whole_photo).image
            remaining = tilt_from_writing(image, find_pie_ellipse(image))
            self.assertIsNotNone(remaining, f"tilt {tilt}")
            self.assertLess(abs(remaining), 0.75, f"a page tilted by {tilt} still leans {remaining:+.2f}")

    def test_ink_with_no_rows_gives_no_estimate(self):
        rng = np.random.default_rng(1)
        photo = np.full((800, 1200, 3), 240, np.uint8)
        cv2.circle(photo, (600, 400), 230, (20, 20, 20), 4)
        for x, y in rng.integers([40, 40], [1160, 760], size=(500, 2)):
            if np.hypot(x - 600, y - 400) > 290:                              # specks scattered outside the pie
                cv2.circle(photo, (int(x), int(y)), 3, (20, 20, 20), -1)
        self.assertIsNone(tilt_from_writing(photo, self.PIE))

    def test_a_round_pie_is_left_alone(self):
        whole_photo = np.array([[0, 0], [1200, 0], [1200, 800], [0, 800]], np.float32)
        self.assertEqual(deskew(self.make_page(True, 0.0), corners=whole_photo).pie_stretch, 1.0)

    def test_sideways_writing_does_not_rotate_a_chart_with_no_pie(self):
        photo = self.make_page(False, 4.0)
        corners = np.array([[0, 0], [1200, 0], [1200, 800], [0, 800]], np.float32)
        self.assertEqual(deskew(photo, corners=corners).rotation_deg, 0.0)


class LevelledImageTests(unittest.TestCase):
    """The saved image itself must come out level, not just report a rotation."""

    def test_a_tilted_chart_comes_out_level(self):
        whole_photo = np.array([[0, 0], [1200, 0], [1200, 800], [0, 800]], np.float32)
        for tilt in (-5.0, 4.0):
            chart = np.full((800, 1200, 3), 240, np.uint8)
            draw_axes(chart, (200, 600), 1000, 200)
            photo = cv2.warpAffine(chart, cv2.getRotationMatrix2D((600, 400), tilt, 1.0), (1200, 800),
                                   borderValue=(240, 240, 240))
            remaining = tilt_from_chart_axes(find_long_straight_lines(deskew(photo, corners=whole_photo).image))
            self.assertIsNotNone(remaining, f"tilt {tilt}")
            self.assertLess(abs(remaining), 0.5, f"a chart tilted by {tilt} still leans {remaining:+.2f}")


def photo_of_steep_board(yaw: float, size=(1800, 1200)):
    """
    A 3 m x 1 m board with a dark frame, seen by the assumed camera and turned by `yaw`
    about its vertical axis. Its left side is in shot; the right side runs off the photo.
    Returns (photo, true corners TL, TR, BR, BL).
    """
    h, w = size
    board_corners = np.array([[0, -0.5, 0], [3, -0.5, 0], [3, 0.5, 0], [0, 0.5, 0]], np.float64)
    projected, _ = cv2.projectPoints(board_corners, np.array([0.05, yaw, 0.0]), np.array([-0.45, 0.0, 2.2]),
                                     camera_matrix(w, h), None)
    corners = projected.reshape(4, 2).astype(np.float32)
    board = np.full((1000, 3000, 3), 240, np.uint8)
    cv2.rectangle(board, (0, 0), (2999, 999), (90, 90, 90), 40)
    flat = np.array([[0, 0], [3000, 0], [3000, 1000], [0, 1000]], np.float32)
    photo = cv2.warpPerspective(board, cv2.getPerspectiveTransform(flat, corners), (w, h),
                                borderValue=(130, 150, 140))
    return photo, corners


class ThreeEdgeBoardTests(unittest.TestCase):
    """A board with its far side or corners out of shot, found from three of its edges."""

    def test_gentle_view_closes_the_far_side_along_the_photos_border(self):
        # Wall with a light board whose right side runs off the photo; top and bottom edges are parallel.
        photo = np.full((1000, 1600, 3), 130, np.uint8)
        cv2.rectangle(photo, (300, 250), (1900, 800), (235, 235, 235), -1)
        cv2.rectangle(photo, (300, 250), (1900, 800), (80, 80, 80), 8)

        corners, confidence, source = find_three_edge_board(photo)
        self.assertEqual(source, "partial")
        self.assertGreaterEqual(confidence, 0.8)
        tl, tr, br, bl = corners
        for found, expected in ((tl, (300, 250)), (bl, (300, 800)), (tr, (1599, 250)), (br, (1599, 800))):
            self.assertLess(np.linalg.norm(found - expected), 20, f"{found} vs {expected}")

    def test_steep_view_finds_the_near_corners_when_the_far_side_is_out_of_shot(self):
        photo, truth = photo_of_steep_board(yaw=0.6)
        self.assertTrue(np.all(truth[[1, 2]] > [1200, 0]) or np.any(truth[[1, 2]] < 0), "test setup: far side in shot")
        corners, confidence, source = find_three_edge_board(photo)
        self.assertEqual(source, "steep")
        self.assertGreaterEqual(confidence, 0.8)
        tl, tr, br, bl = corners
        self.assertLess(np.linalg.norm(tl - truth[0]), 30)
        self.assertLess(np.linalg.norm(bl - truth[3]), 30)

    def test_a_board_whose_edges_are_only_half_there_is_not_trusted(self):
        photo = np.full((1000, 1600, 3), 130, np.uint8)

        def dashed(start, end, dash=50, gap=70):
            start, end = np.array(start, float), np.array(end, float)
            length = np.linalg.norm(end - start)
            direction = (end - start) / length
            for at in np.arange(0, length, dash + gap):
                cv2.line(photo, tuple((start + direction * at).astype(int)),
                         tuple((start + direction * min(at + dash, length)).astype(int)), (240, 240, 240), 8)

        dashed((300, 250), (1900, 250))
        dashed((300, 800), (1900, 800))
        dashed((300, 250), (300, 800))
        self.assertIsNone(find_three_edge_board(photo)[0])

    @staticmethod
    def edges_of_a_known_board(yaw):
        """A 3 m x 1 m board seen by the assumed camera: (camera, corners TL TR BR BL, top, bottom, left edge lines)."""
        camera = camera_matrix(1200, 1800)
        board = np.array([[0, -0.5, 0], [3, -0.5, 0], [3, 0.5, 0], [0, 0.5, 0]], np.float64)
        pose = (np.array([0.05, yaw, 0.0]), np.array([-0.45, 0.0, 2.2]))
        corners = cv2.projectPoints(board, *pose, camera, None)[0].reshape(4, 2)

        def line(p, q):                                    # the (rho, theta) form: x*cos(theta) + y*sin(theta) = rho
            a, b = q[1] - p[1], p[0] - q[0]
            a, b = a / np.hypot(a, b), b / np.hypot(a, b)
            return float(a * p[0] + b * p[1]), float(np.arctan2(b, a))

        tl, tr, br, bl = corners
        return camera, corners, line(tl, tr), line(bl, br), line(tl, bl)

    def test_the_face_on_matrix_turns_a_known_board_into_a_rectangle(self):
        for yaw in (0.3, 0.6, 1.0, 1.3):                   # from about 17 to 74 degrees away from face-on
            camera, corners, top, bottom, side = self.edges_of_a_known_board(yaw)
            matrix = three_edge_board._face_on_homography(camera, top, bottom, side)
            self.assertIsNotNone(matrix, f"yaw {yaw}")
            tl, tr, br, bl = (three_edge_board._apply(matrix, corner) for corner in corners)
            across, down = tr - tl, bl - tl
            self.assertLess(abs(np.degrees(np.arctan2(across[1], across[0]))), 0.1, f"yaw {yaw}: top edge not level")
            self.assertLess(abs(np.degrees(np.arctan2(down[0], down[1]))), 0.1, f"yaw {yaw}: side edge not upright")
            self.assertAlmostEqual(np.linalg.norm(across) / np.linalg.norm(down), 3.0, delta=0.05, msg=f"yaw {yaw}")

    def test_a_board_turned_almost_edge_on_is_refused(self):
        camera, _, top, bottom, side = self.edges_of_a_known_board(1.45)       # about 83 degrees from face-on
        self.assertIsNone(three_edge_board._face_on_homography(camera, top, bottom, side))

    @staticmethod
    def board_running_off_the_right(left, top, bottom, size=(1000, 1600)):
        """A light board with a dark outline on a darker wall; its left edge is at `left`, and it runs off the photo."""
        height, width = size
        photo = np.full((height, width, 3), 130, np.uint8)
        cv2.rectangle(photo, (left, top), (width + 1000, bottom), (235, 235, 235), -1)
        cv2.rectangle(photo, (left, top), (width + 1000, bottom), (80, 80, 80), 8)
        return photo

    def test_a_board_with_only_a_short_stretch_of_its_edges_showing_is_not_trusted(self):
        # 300 px of a 1600 px wide photo (19%): too little of the top and bottom edges to judge by
        self.assertIsNone(find_three_edge_board(self.board_running_off_the_right(1300, 250, 800))[0])
        enough = self.board_running_off_the_right(1000, 250, 800)               # 600 px showing
        self.assertIsNotNone(find_three_edge_board(enough)[0])

    def test_a_strip_too_short_to_be_a_board_is_not_trusted(self):
        # In a tall photo, a side edge only 19% of the height cannot be a whiteboard's side
        tall = (1600, 1000)
        self.assertIsNone(find_three_edge_board(self.board_running_off_the_right(300, 600, 900, tall))[0])
        self.assertIsNotNone(find_three_edge_board(self.board_running_off_the_right(300, 600, 1000, tall))[0])  # 25%

class CornerEditorTests(unittest.TestCase):
    def setUp(self):
        self.img = np.full((300, 400, 3), 200, np.uint8)
        self.start = np.array([[50, 40], [350, 40], [350, 260], [50, 260]], np.float32)
        self.editor = CornerEditor(self.img, initial=self.start)

    def grab(self, i):
        x, y = self.editor.handles[i]
        self.editor.handle_mouse(cv2.EVENT_LBUTTONDOWN, int(x), int(y))

    def test_corner_can_be_dragged_outside_the_photo(self):
        self.grab(0)
        self.editor.handle_mouse(cv2.EVENT_MOUSEMOVE, 0, 0)                 # top-left of the grey canvas
        self.editor.handle_mouse(cv2.EVENT_LBUTTONUP, 0, 0)
        x, y = self.editor.corners()[0]
        self.assertLess(x, 0)
        self.assertLess(y, 0)

    def test_clicking_away_from_every_handle_moves_nothing(self):
        self.editor.handle_mouse(cv2.EVENT_LBUTTONDOWN, 5, 5)
        self.editor.handle_mouse(cv2.EVENT_MOUSEMOVE, 100, 100)
        np.testing.assert_allclose(self.editor.corners(), self.start, atol=0.01)

    def test_keys_pick_and_nudge_a_handle_by_one_display_pixel(self):
        self.editor.handle_key(ord("3"))
        before = self.editor.corners()[2].copy()
        self.editor.handle_key(ord("d"))
        self.editor.handle_key(ord("s"))
        step = 1 / self.editor.scale
        np.testing.assert_allclose(self.editor.corners()[2] - before, [step, step], atol=0.01)

    def test_suggested_corners_far_off_the_canvas_are_pulled_onto_it(self):
        far = np.array([[-5000, -5000], [9000, -5000], [9000, 9000], [-5000, 9000]], np.float32)
        editor = CornerEditor(self.img, initial=far)
        canvas_h, canvas_w = editor.base.shape[:2]
        for handle in range(4):
            editor.active = handle
            editor.render()                                  # the magnifier used to fail on an off-canvas handle
            x, y = editor.handles[handle]
            self.assertTrue(0 <= x < canvas_w and 0 <= y < canvas_h)

    def test_enter_accepts_escape_cancels(self):
        self.assertEqual(self.editor.handle_key(13), "accept")
        self.assertEqual(self.editor.handle_key(27), "cancel")
        self.assertIsNone(self.editor.handle_key(ord("x")))


# name -> (expected corner_source, extra check on the result or None)
PHOTOS = {
    "20261027_120202": ("auto", lambda r: r.confidence >= 0.7),
    "20261027_120212": ("auto", lambda r: 0.4 <= r.confidence < 0.7),
    "20261027_121850": ("steep", None),     # gentle enough for either method; the merged finder uses the camera one
    "20261027_121908": ("steep", None),
    "new_1": ("steep", lambda r: r.confidence >= 0.7),
    "new_2": ("steep", lambda r: r.confidence >= 0.9),
    "new_3": ("auto", lambda r: 1.03 <= r.pie_stretch <= 1.12),
    "new_4": ("partial", lambda r: 1.03 <= r.pie_stretch <= 1.12 and r.rotation_deg == 0.0),  # no axes: no rotation
    # Only the top (and one side) of the board is in shot: flagged for corners, photo left untouched.
    "IMG_1294": ("none", lambda r: r.needs_review),
    "IMG_1295": ("none", lambda r: r.needs_review),
}


@unittest.skipUnless(SAMPLES.is_dir(), "sample-images/ not found")
class SamplePhotoTests(unittest.TestCase):
    def test_photos(self):
        for name, (source, check) in PHOTOS.items():
            path = SAMPLES / f"{name}.jpg"
            if not path.exists():
                continue
            with self.subTest(photo=name):
                img = cv2.imread(str(path))
                result = deskew(img)
                self.assertEqual(result.corner_source, source)
                if source == "none":
                    # A guess we don't trust must not touch the photo.
                    self.assertIs(result.image, img)
                else:
                    self.assertFalse(result.needs_review)
                if check is not None:
                    self.assertTrue(check(result), f"extra check failed for {name}")


if __name__ == "__main__":
    unittest.main()

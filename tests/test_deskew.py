"""
Regression tests for the preprocessing stage.

Run from the project root:

    python3 -m unittest discover tests -v

The synthetic tests build geometry with a known answer, so they need no photos.
The photo tests run over sample-images/ and are skipped if that folder is
missing; the expected outcomes there are what the detectors were tuned on, so a
failure means a behaviour change to look at, not necessarily a bug.
"""

import unittest
from pathlib import Path

import cv2
import numpy as np

from chartx.preprocessing.board import estimate_aspect_ratio
from chartx.preprocessing.corner_editor import CornerEditor
from chartx.preprocessing.deskew import deskew
from chartx.preprocessing.ellipse import find_pie_ellipse, pie_to_circle_transform
from chartx.preprocessing.geometry import camera_matrix, order_corners
from chartx.preprocessing.partial_board import find_partial_board_corners
from chartx.preprocessing.rotation import find_long_straight_lines, tilt_from_chart_axes, tilt_from_writing
from chartx.preprocessing.steep_board import find_steep_board_corners

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
    def test_order_corners(self):
        shuffled = [[90, 80], [10, 10], [10, 80], [90, 10]]
        np.testing.assert_array_equal(order_corners(shuffled),
                                      [[10, 10], [90, 10], [90, 80], [10, 80]])

    def test_aspect_ratio_recovered_for_known_camera(self):
        shape = (2252, 4000, 3)
        for true_ratio in (1.0, 1.5, 2.0):
            corners = project_rectangle(true_ratio, 1.0, shape)
            self.assertAlmostEqual(estimate_aspect_ratio(corners, shape), true_ratio,
                                   delta=0.03 * true_ratio)

    def test_aspect_ratio_tolerates_wrong_focal_length(self):
        # The camera isn't the one we assume; the answer should stay usable.
        shape = (2252, 4000, 3)
        for focal_frac in (0.5, 1.0):
            corners = project_rectangle(1.5, 1.0, shape, focal_frac=focal_frac)
            self.assertAlmostEqual(estimate_aspect_ratio(corners, shape), 1.5, delta=0.2)


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


    def test_a_photo_with_no_board_or_chart_is_returned_untouched_and_flagged(self):
        wall = np.full((1000, 1600, 3), 130, np.uint8)
        result = deskew(wall)
        self.assertEqual(result.corner_source, "none")
        self.assertTrue(result.needs_review)
        self.assertIs(result.image, wall)

    def test_a_tilted_close_up_with_no_board_is_levelled_but_still_flagged(self):
        close_up = np.full((900, 1200, 3), 235, np.uint8)
        draw_axes(close_up, (200, 700), 1000, 150)
        tilted = cv2.warpAffine(close_up, cv2.getRotationMatrix2D((600, 450), 4.0, 1.0), (1200, 900),
                                borderValue=(235, 235, 235))
        result = deskew(tilted)
        self.assertEqual(result.corner_source, "none")
        self.assertTrue(result.needs_review)                     # perspective was not corrected
        self.assertAlmostEqual(result.rotation_deg, -4.0, delta=0.5)
        self.assertIsNot(result.image, tilted)


class RotationTests(unittest.TestCase):
    def test_recovers_a_known_tilt(self):
        img = np.full((800, 1200, 3), 240, np.uint8)
        draw_axes(img, (200, 600), 1000, 200)
        for tilt in (-5.0, 3.0):
            M = cv2.getRotationMatrix2D((600, 400), tilt, 1.0)             # CCW positive
            tilted = cv2.warpAffine(img, M, (1200, 800), borderValue=(240, 240, 240))
            self.assertAlmostEqual(tilt_from_chart_axes(find_long_straight_lines(tilted)), tilt, delta=0.5)

    def test_repeated_segments_of_one_stroke_do_not_outvote_the_axes(self):
        # Hough reports one handwritten stroke (say a row of labels, 2 deg off) as
        # several overlapping segments; together they must not beat two real axes.
        def seg(x, y, length, dev_deg, vertical=False):
            t = np.radians(dev_deg)
            if vertical:
                return [x, y, int(x - length * np.sin(t)), int(y + length * np.cos(t))]
            return [x, y, int(x + length * np.cos(t)), int(y - length * np.sin(t))]
        axes = [seg(383, 1024, 670, 0.26), seg(448, 1024, 529, 0.22, vertical=True)]
        stroke = [seg(501, 1069, 437, 1.97), seg(522, 1068, 415, 2.07),
                  seg(521, 1065, 415, 1.93), seg(521, 1066, 415, 1.93)]
        tilt = tilt_from_chart_axes(np.array(axes + stroke))
        self.assertLess(abs(tilt), 0.6)

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

    def test_leaves_a_level_chart_alone(self):
        img = np.full((800, 1200, 3), 240, np.uint8)
        draw_axes(img, (200, 600), 1000, 200)
        self.assertAlmostEqual(tilt_from_chart_axes(find_long_straight_lines(img)), 0.0, delta=0.1)


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

    def test_recovers_a_known_tilt_from_the_writing(self):
        for tilt in (-4.0, 3.0):
            found = tilt_from_writing(self.make_page(True, tilt), self.PIE)
            self.assertIsNotNone(found)
            self.assertAlmostEqual(found, tilt, delta=0.5)

    def test_no_writing_no_estimate(self):
        blank = np.full((800, 1200, 3), 240, np.uint8)
        self.assertIsNone(tilt_from_writing(blank, self.PIE))

    def test_a_pie_chart_is_levelled_on_its_writing_end_to_end(self):
        photo = self.make_page(True, 4.0)
        corners = np.array([[0, 0], [1200, 0], [1200, 800], [0, 800]], np.float32)
        result = deskew(photo, corners=corners)
        self.assertAlmostEqual(result.rotation_deg, -4.0, delta=0.6)

    def test_sideways_writing_does_not_rotate_a_chart_with_no_pie(self):
        photo = self.make_page(False, 4.0)
        corners = np.array([[0, 0], [1200, 0], [1200, 800], [0, 800]], np.float32)
        self.assertEqual(deskew(photo, corners=corners).rotation_deg, 0.0)


class PartialBoardTests(unittest.TestCase):
    def test_finds_three_visible_edges_and_closes_the_fourth_on_the_border(self):
        # Wall with a light board whose right side runs off the photo.
        photo = np.full((1000, 1600, 3), 130, np.uint8)
        cv2.rectangle(photo, (300, 250), (1900, 800), (235, 235, 235), -1)
        cv2.rectangle(photo, (300, 250), (1900, 800), (80, 80, 80), 8)

        corners, confidence = find_partial_board_corners(photo)
        self.assertIsNotNone(corners)
        self.assertGreaterEqual(confidence, 0.8)
        tl, tr, br, bl = corners
        for found, expected in ((tl, (300, 250)), (bl, (300, 800)), (tr, (1599, 250)), (br, (1599, 800))):
            self.assertLess(np.linalg.norm(found - expected), 20, f"{found} vs {expected}")

    def test_needs_a_visible_vertical_edge(self):
        # Board runs off both sides of the photo: top and bottom only, nothing to close the shape on.
        photo = np.full((1000, 1600, 3), 130, np.uint8)
        cv2.rectangle(photo, (-50, 250), (1700, 800), (235, 235, 235), -1)
        cv2.rectangle(photo, (-50, 250), (1700, 800), (80, 80, 80), 8)
        corners, _ = find_partial_board_corners(photo)
        self.assertIsNone(corners)

    def test_plain_wall_has_no_board(self):
        corners, confidence = find_partial_board_corners(np.full((1000, 1600, 3), 130, np.uint8))
        self.assertIsNone(corners)


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


class SteepBoardTests(unittest.TestCase):
    def test_finds_the_near_corners_when_the_far_side_is_out_of_shot(self):
        photo, truth = photo_of_steep_board(yaw=0.6)
        self.assertTrue(np.all(truth[[1, 2]] > [1200, 0]) or np.any(truth[[1, 2]] < 0), "test setup: far side in shot")
        corners, confidence = find_steep_board_corners(photo)
        self.assertIsNotNone(corners)
        self.assertGreaterEqual(confidence, 0.8)
        tl, tr, br, bl = corners
        self.assertLess(np.linalg.norm(tl - truth[0]), 30)
        self.assertLess(np.linalg.norm(bl - truth[3]), 30)

    def test_a_head_on_board_is_left_to_the_other_finders(self):
        # Top and bottom edges are parallel: they say nothing about the board's tilt.
        photo, _ = photo_of_steep_board(yaw=0.0)
        self.assertIsNone(find_steep_board_corners(photo)[0])

    def test_plain_wall_has_no_board(self):
        self.assertIsNone(find_steep_board_corners(np.full((1800, 1200, 3), 130, np.uint8))[0])


class CornerEditorTests(unittest.TestCase):
    def setUp(self):
        self.img = np.full((300, 400, 3), 200, np.uint8)
        self.start = np.array([[50, 40], [350, 40], [350, 260], [50, 260]], np.float32)
        self.editor = CornerEditor(self.img, initial=self.start)

    def grab(self, i):
        x, y = self.editor.handles[i]
        self.editor.handle_mouse(cv2.EVENT_LBUTTONDOWN, int(x), int(y))

    def test_round_trip_is_exact_when_nothing_moves(self):
        np.testing.assert_allclose(self.editor.corners(), self.start, atol=0.01)

    def test_corner_can_be_dragged_outside_the_photo(self):
        self.grab(0)
        self.editor.handle_mouse(cv2.EVENT_MOUSEMOVE, 0, 0)                 # top-left of the grey canvas
        self.editor.handle_mouse(cv2.EVENT_LBUTTONUP, 0, 0)
        x, y = self.editor.corners()[0]
        self.assertLess(x, 0)
        self.assertLess(y, 0)

    def test_drag_is_clamped_to_the_canvas(self):
        self.grab(0)
        self.editor.handle_mouse(cv2.EVENT_MOUSEMOVE, -500, -500)
        edge = self.editor.corners()[0].copy()
        self.editor.handle_mouse(cv2.EVENT_MOUSEMOVE, 0, 0)
        np.testing.assert_allclose(edge, self.editor.corners()[0])

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

    def test_enter_accepts_escape_cancels(self):
        self.assertEqual(self.editor.handle_key(13), "accept")
        self.assertEqual(self.editor.handle_key(27), "cancel")
        self.assertIsNone(self.editor.handle_key(ord("x")))


# name -> (expected corner_source, extra check on the result or None)
PHOTOS = {
    "20261027_120202": ("auto", lambda r: r.confidence >= 0.7),
    "20261027_120212": ("auto", lambda r: 0.4 <= r.confidence < 0.7),
    "20261027_121850": ("partial", None),
    "20261027_121908": ("partial", None),
    "new_1": ("steep", lambda r: r.confidence >= 0.7),
    "new_2": ("steep", lambda r: r.confidence >= 0.9),
    "new_3": ("auto", lambda r: 1.03 <= r.pie_stretch <= 1.12),
    "new_4": ("partial", lambda r: 1.03 <= r.pie_stretch <= 1.12 and r.rotation_deg == 0.0),  # no axes: no rotation
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

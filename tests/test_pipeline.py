"""
Tests for the pipeline that chains the preprocessing steps.

Run from the project root:

    python3 -m unittest discover tests -v
"""

import unittest

import cv2
import numpy as np

from chartx.pipeline import preprocess
from chartx.preprocessing import glare
from chartx.preprocessing.deskew import deskew

WALL = np.full((1000, 1600, 3), 130, np.uint8)                 # no board, no chart
BOARD_CORNERS = np.array([[100, 100], [1500, 100], [1500, 900], [100, 900]], np.float32)


def tilted_chart(tilt_deg: float = 4.0) -> np.ndarray:
    """A close-up of a bare pair of axes on white, tilted by `tilt_deg`."""
    img = np.full((900, 1200, 3), 235, np.uint8)
    cv2.line(img, (200, 700), (1000, 700), (20, 20, 20), 5)
    cv2.line(img, (200, 700), (200, 150), (20, 20, 20), 5)
    rotate = cv2.getRotationMatrix2D((600, 450), tilt_deg, 1.0)
    return cv2.warpAffine(img, rotate, (1200, 900), borderValue=(235, 235, 235))


class PreprocessTests(unittest.TestCase):
    def test_straightens_then_cleans_the_glare(self):
        photo = tilted_chart()
        whole_photo = np.array([[0, 0], [1200, 0], [1200, 900], [0, 900]], np.float32)
        result = preprocess(photo, corners=whole_photo)
        np.testing.assert_array_equal(result.image, glare.clean_whiteboard(deskew(photo, corners=whole_photo).image))
        self.assertAlmostEqual(result.straightened.rotation_deg, -4.0, delta=0.5)

    def test_can_stop_after_straightening(self):
        result = preprocess(tilted_chart(), clean_glare=False)
        self.assertIs(result.image, result.straightened.image)

    def test_corners_from_the_hook_are_used(self):
        asked = []

        def ask(photo, straightened):
            asked.append(straightened.needs_review)         # the hook sees how the first attempt went
            return BOARD_CORNERS

        result = preprocess(WALL, ask_for_corners=ask, clean_glare=False)
        self.assertEqual(asked, [True])
        self.assertEqual(result.straightened.corner_source, "manual")
        self.assertFalse(result.straightened.needs_review)

    def test_a_none_answer_keeps_the_first_attempt(self):
        result = preprocess(WALL, ask_for_corners=lambda photo, straightened: None, clean_glare=False)
        self.assertEqual(result.straightened.corner_source, "none")
        self.assertTrue(result.straightened.needs_review)

    def test_corners_the_pipeline_rejected_are_not_labelled_manual(self):
        collapsed = np.array([[500, 500], [501, 500], [501, 501], [500, 501]], np.float32)
        result = preprocess(WALL, ask_for_corners=lambda photo, straightened: collapsed, clean_glare=False)
        self.assertEqual(result.straightened.corner_source, "none")

    def test_the_hook_is_not_called_when_corners_are_given(self):
        asked = []
        result = preprocess(WALL, corners=BOARD_CORNERS,
                            ask_for_corners=lambda photo, straightened: asked.append(1), clean_glare=False)
        self.assertEqual(asked, [])
        self.assertEqual(result.straightened.corner_source, "provided")


if __name__ == "__main__":
    unittest.main()

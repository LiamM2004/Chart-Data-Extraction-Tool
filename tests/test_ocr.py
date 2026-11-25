"""
Tests for chartx/detection/ocr.py: reading text off a chart with PaddleOCR.

Run from the project root:

    python3 -m unittest discover tests -v
"""

import unittest

import cv2
import numpy as np

from chartx.detection import ocr

BLACK = (0, 0, 0)


def written(text, size=(240, 90), at=(20, 62), scale=1.8):
    """A white image with `text` written on it in black."""
    image = np.full((size[1], size[0], 3), 255, np.uint8)
    cv2.putText(image, text, at, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 4)
    return image


class ReadTests(unittest.TestCase):
    def test_reads_what_is_written_on_a_crop(self):
        self.assertEqual(ocr.read(written("2020")), "2020")

    def test_reads_a_grey_crop_as_well_as_a_colour_one(self):
        grey = cv2.cvtColor(written("1995"), cv2.COLOR_BGR2GRAY)
        self.assertEqual(ocr.read(grey), "1995")

    def test_an_empty_crop_reads_as_nothing(self):
        self.assertEqual(ocr.read(np.empty((0, 0), np.uint8)), "")

    def test_a_blank_crop_reads_as_nothing(self):
        self.assertEqual(ocr.read(np.full((60, 120, 3), 255, np.uint8)), "")

    def test_the_spaces_are_taken_out(self):
        self.assertNotIn(" ", ocr.read(written("20 20")))


class ReadPageTests(unittest.TestCase):
    def test_gives_each_piece_of_text_its_place_on_the_page(self):
        image = np.full((260, 400, 3), 255, np.uint8)
        cv2.putText(image, "Red", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, BLACK, 4)
        cv2.putText(image, "Blue", (30, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.6, BLACK, 4)

        found = ocr.read_page(image)
        self.assertEqual(len(found), 2, f"expected two pieces of text, got {found}")
        said = {}
        for corners, text, score in found:
            middle_row = np.array(corners, float)[:, 1].mean()
            said[text] = middle_row
            self.assertGreater(score, 0.5)
        self.assertEqual(sorted(said), ["Blue", "Red"])
        self.assertLess(said["Red"], said["Blue"], "Red is written above Blue")

    def test_a_blank_page_has_no_text_on_it(self):
        self.assertEqual(ocr.read_page(np.full((80, 200, 3), 255, np.uint8)), [])


if __name__ == "__main__":
    unittest.main()

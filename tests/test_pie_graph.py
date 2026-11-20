"""
Tests for chartx/detection/pie_graph.py on four hand-drawn pie charts. Every image goes through
the same preprocessing as in chartx/main.py (straighten + glare clean-up) first.

The images sample-images/chartarena_pie_*.png are from the ChartArena benchmark
(Peng et al., arXiv:2606.01348), released for research purposes only.
IMG_1294.jpg is a whiteboard photo with coloured (hatched) sectors and no drawn dividers.

The first run downloads the PaddleOCR models, so it takes a while.

Run from the project root:

    python3 -m unittest tests.test_pie_graph -v
"""

import unittest
from pathlib import Path

import cv2
import numpy as np

from chartx.detection.pie_graph import detect_pie_graph
from chartx.main import preprocess

SAMPLES = Path(__file__).resolve().parent.parent / "sample-images"

# The written values, clockwise from 3 o'clock (the order of the sectors)
WRITTEN = {
    "chartarena_pie_5.png": [30, 25, 10, 5, 10, 20],
    "chartarena_pie_11.png": [20, 80],
    "chartarena_pie_19.png": [18, 10, 9, 63],
    "IMG_1294.jpg": [60, 8, 12, 25],
}
# The OCR values the detector returns. None = not used:
# - pie_5: "Vegetables 10%" is one OCR box (read as "Vegetables 0%"), but the 10% belongs to the next
#   sector. The box goes to the fruit sector, which then has two values, and water/milk has none
# - IMG_1294: the labels and percentages are outside the circle, so they are not read
OCR_VALUES = {
    "chartarena_pie_5.png": [None] * 6,
    "chartarena_pie_11.png": [20, 80],
    "chartarena_pie_19.png": [18, 10, 9, 63],
    "IMG_1294.jpg": [None] * 4,
}
# One word of every label (IMG_1294: none, see above)
LABEL_WORDS = {
    "chartarena_pie_5.png": ["rice", "fruit", "water", "cake", "eggs", "meat"],
    "chartarena_pie_11.png": ["What", "affection"],
    "chartarena_pie_19.png": ["Hats", "Gloves", "Boots", "shoes"],
}
ANGLE_TOL = 6.0     # the drawn angles are not exact: the calculated value may be this far off (pp)


class PieGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # detect once per image: OCR is slow
        cls.found = {name: detect_pie_graph(preprocess(cv2.imread(str(SAMPLES / name)))) for name in WRITTEN}

    def test_finds_one_sector_per_written_value(self):
        for name, written in WRITTEN.items():
            with self.subTest(image=name):
                self.assertEqual(len(self.found[name]), len(written))

    def test_reads_the_written_percentages(self):
        for name, values in OCR_VALUES.items():
            with self.subTest(image=name):
                self.assertEqual([s.ocr_value for s in self.found[name]], values)

    def test_every_sector_gets_its_own_label(self):
        for name, words in LABEL_WORDS.items():
            with self.subTest(image=name):
                for sector, word in zip(self.found[name], words):
                    self.assertIn(word.lower(), (sector.label or "").lower(), f"sector {sector.index}")

    def test_calculated_values_are_close_to_the_written_ones(self):
        for name, written in WRITTEN.items():
            with self.subTest(image=name):
                for sector, value in zip(self.found[name], written):
                    self.assertAlmostEqual(sector.value, value, delta=ANGLE_TOL, msg=f"sector {sector.index}")

    def test_sectors_go_clockwise_and_add_up_to_100(self):
        for name in WRITTEN:
            with self.subTest(image=name):
                sectors = self.found[name]
                self.assertEqual([s.index for s in sectors], list(range(1, len(sectors) + 1)))
                for sector, after in zip(sectors, sectors[1:] + sectors[:1]):
                    self.assertEqual(sector.end_angle, after.start_angle)
                self.assertAlmostEqual(sum(s.value for s in sectors), 100, delta=0.5)

    def test_a_blank_page_has_no_pie(self):
        self.assertEqual(detect_pie_graph(np.full((600, 800, 3), 255, np.uint8)), [])


if __name__ == "__main__":
    unittest.main()

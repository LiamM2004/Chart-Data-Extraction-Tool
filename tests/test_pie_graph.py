"""
Tests for chartx/detection/pie_graph.py on three hand-drawn pie charts.

The images (sample-images/chartarena_pie_*.png) are from the ChartArena benchmark
(Peng et al., arXiv:2606.01348), released for research purposes only.

The first run downloads the PaddleOCR models, so it takes a while.

Run from the project root:

    python3 -m unittest tests.test_pie_graph -v
"""

import unittest
from pathlib import Path

import cv2
import numpy as np

from chartx.detection.pie_graph import detect_pie_graph

SAMPLES = Path(__file__).resolve().parent.parent / "sample-images"

# The written values and one word of every label, clockwise from 3 o'clock (the order of the sectors)
EXPECTED = {
    "chartarena_pie_5.png": [(30, "rice"), (25, "Vegetables"), (10, "water"), (5, "cake"), (10, "eggs"),
                             (20, "meat")],
    "chartarena_pie_11.png": [(20, "know"), (80, "affection")],
    "chartarena_pie_19.png": [(18, "Hats"), (10, "Gloves"), (9, "Boots"), (63, "shoes")],
}
ANGLE_TOL = 6.0     # the drawn angles are not exact: the calculated value may be this far off (pp)


class PieGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # detect once per image: OCR is slow
        cls.found = {name: detect_pie_graph(cv2.imread(str(SAMPLES / name))) for name in EXPECTED}

    def test_finds_one_sector_per_written_value(self):
        for name, expected in EXPECTED.items():
            with self.subTest(image=name):
                self.assertEqual(len(self.found[name]), len(expected))

    def test_reads_the_written_percentages(self):
        for name, expected in EXPECTED.items():
            with self.subTest(image=name):
                self.assertEqual([s.ocr_value for s in self.found[name]], [value for value, _ in expected])

    def test_every_sector_gets_its_own_label(self):
        for name, expected in EXPECTED.items():
            with self.subTest(image=name):
                for sector, (_, word) in zip(self.found[name], expected):
                    self.assertIn(word, sector.label or "", f"sector {sector.index}")

    def test_calculated_values_are_close_to_the_written_ones(self):
        for name, expected in EXPECTED.items():
            with self.subTest(image=name):
                for sector, (value, _) in zip(self.found[name], expected):
                    self.assertAlmostEqual(sector.value, value, delta=ANGLE_TOL, msg=f"sector {sector.index}")

    def test_sectors_go_clockwise_and_add_up_to_100(self):
        for name in EXPECTED:
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

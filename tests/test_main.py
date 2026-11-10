"""
Tests for the command line (python -m chartx).

Run from the project root:

    python3 -m unittest discover tests -v
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from chartx.main import main


def run(*arguments):
    """Run the command line with these arguments. Returns (exit code or None, printed text)."""
    shown = io.StringIO()
    code = None
    with mock.patch.object(sys, "argv", ["chartx", *arguments]), \
            contextlib.redirect_stdout(shown), contextlib.redirect_stderr(shown):
        try:
            main()
        except SystemExit as stop:
            code = stop.code
    return code, shown.getvalue()


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        photo = np.full((900, 1200, 3), 235, np.uint8)
        cv2.line(photo, (200, 700), (1000, 700), (20, 20, 20), 5)
        cv2.line(photo, (200, 700), (200, 150), (20, 20, 20), 5)
        self.photo = str(Path(self.folder.name) / "photo.png")
        cv2.imwrite(self.photo, photo)
        self.whole_photo = ["0,0", "1200,0", "1200,900", "0,900"]

    def test_saves_the_result_and_the_debug_image_creating_their_folder(self):
        output = Path(self.folder.name) / "not" / "there" / "result.png"
        debug = Path(self.folder.name) / "also" / "new" / "debug.png"
        code, shown = run(self.photo, "-o", str(output), "--debug", str(debug), "--no-glare",
                          "--corners", *self.whole_photo)
        self.assertIsNone(code)
        self.assertIsNotNone(cv2.imread(str(output)))
        self.assertIsNotNone(cv2.imread(str(debug)))
        self.assertIn("corners: provided", shown)

    def test_a_save_that_fails_is_reported_not_ignored(self):
        code, _ = run(self.photo, "-o", str(Path(self.folder.name) / "result.notanimage"), "--no-glare",
                      "--corners", *self.whole_photo)
        self.assertIn("Could not save", str(code))

    def test_a_photo_that_cannot_be_read_is_reported(self):
        code, _ = run(str(Path(self.folder.name) / "missing.png"), "-o", str(Path(self.folder.name) / "x.png"))
        self.assertIn("Could not read", str(code))

    def test_manual_and_corners_together_are_refused(self):
        code, shown = run(self.photo, "--manual", "--corners", *self.whole_photo)
        self.assertEqual(code, 2)
        self.assertIn("can't be used together", shown)

    def test_a_focal_length_must_be_a_positive_number(self):
        for bad in ("0", "-1", "wide"):
            code, _ = run(self.photo, "--focal", bad, "--no-prompt")
            self.assertEqual(code, 2, bad)

    def test_a_corner_must_be_written_as_x_comma_y(self):
        code, shown = run(self.photo, "--corners", "10;20", "900,15", "920,700", "5,690")
        self.assertEqual(code, 2)
        self.assertIn("is not a point", shown)


if __name__ == "__main__":
    unittest.main()

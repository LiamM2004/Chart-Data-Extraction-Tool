import unittest

import cv2
import numpy as np

from chartx.preprocessing.glare import (
    clean_whiteboard,
    estimate_background,
    intensity_scale,
    sharpen,
    white_balance,
    yen_threshold,
)


def solid(h, w, bgr):
    """Uniform BGR image."""
    return np.full((h, w, 3), bgr, dtype=np.uint8)


def shaded_board(h=240, w=480, left=120, right=220):
    """Grey 'whiteboard' whose brightness ramps left -> right (fake shadow)."""
    ramp = np.linspace(left, right, w, dtype=np.float32)
    board = np.tile(ramp, (h, 1))
    return np.repeat(board[..., None], 3, axis=2).astype(np.uint8)


class TestEstimateBackground(unittest.TestCase):
    def test_output_shape_and_dtype(self):
        # 100x130 is NOT a multiple of the 24px block, so padding is exercised
        bg = estimate_background(solid(100, 130, (200, 200, 200)))
        self.assertEqual(bg.shape, (100, 130, 3))
        self.assertEqual(bg.dtype, np.float32)

    def test_never_below_one(self):
        # Guards against divide-by-zero later in white_balance
        bg = estimate_background(solid(48, 48, (0, 0, 0)))
        self.assertGreaterEqual(bg.min(), 1)

    def test_uniform_image_gives_uniform_background(self):
        bg = estimate_background(solid(96, 96, (180, 190, 200)))
        np.testing.assert_allclose(bg[..., 0].mean(), 180, atol=2)
        np.testing.assert_allclose(bg[..., 1].mean(), 190, atol=2)
        np.testing.assert_allclose(bg[..., 2].mean(), 200, atol=2)

    def test_small_ink_mark_is_ignored(self):
        img = solid(96, 96, (220, 220, 220))
        img[40:46, 40:46] = 0  # tiny black dot, minority of its block
        bg = estimate_background(img)
        self.assertGreater(bg[43, 43].min(), 210)

    def test_large_ink_blob_is_replaced_by_board_colour(self):
        img = solid(240, 240, (220, 220, 220))
        img[96:144, 96:144] = 0  # 2x2 blocks completely ink
        bg = estimate_background(img)
        self.assertGreater(bg[120, 120].min(), 200)

    def test_tracks_a_lighting_gradient(self):
        # Probe the interior: the median filter treats the outer few blocks specially
        bg = estimate_background(shaded_board())
        left, right = float(bg[120, 150, 0]), float(bg[120, 330, 0])
        self.assertLess(left, right)
        self.assertAlmostEqual(left, 120 + (150 / 479) * 100, delta=15)
        self.assertAlmostEqual(right, 120 + (330 / 479) * 100, delta=15)


class TestWhiteBalance(unittest.TestCase):
    def test_output_shape_and_dtype(self):
        img = solid(20, 30, (100, 120, 140)).astype(np.float32)
        out = white_balance(img, np.full(img.shape, 150, np.float32))
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_image_equal_to_background_becomes_white(self):
        bg = np.full((20, 30, 3), 120, np.float32)
        out = white_balance(bg.copy(), bg)
        self.assertTrue((out >= 253).all())

    def test_pixels_brighter_than_background_are_clipped(self):
        img = solid(10, 10, (255, 255, 255)).astype(np.float32)
        out = white_balance(img, np.full(img.shape, 100, np.float32))
        self.assertTrue((out == 255).all())

    def test_grey_stays_grey(self):
        img = solid(10, 10, (90, 90, 90)).astype(np.float32)
        out = white_balance(img, np.full(img.shape, 200, np.float32))
        self.assertEqual(int(out[..., 0].max()) - int(out[..., 2].min()), 0)

    def test_saturation_gain_increases_colourfulness(self):
        img = solid(10, 10, (200, 120, 100)).astype(np.float32)  # bluish
        bg = np.full(img.shape, 255, np.float32)
        sat = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2HSV)[..., 1].mean()
        low = white_balance(img, bg, saturation=1.0)
        high = white_balance(img, bg, saturation=1.6)
        self.assertGreater(sat(high), sat(low))


class TestYenThreshold(unittest.TestCase):
    def test_separates_dark_ink_from_bright_board(self):
        gray = np.full((50, 50), 220, np.uint8)
        gray[20:30, 20:30] = 30
        t = yen_threshold(gray)
        self.assertGreaterEqual(t, 30)
        self.assertLess(t, 220)

    def test_returns_valid_level(self):
        rng = np.random.default_rng(0)
        gray = rng.integers(0, 256, (40, 40), dtype=np.uint8)
        self.assertTrue(0 <= int(yen_threshold(gray)) <= 255)


class TestIntensityScale(unittest.TestCase):
    def test_board_becomes_white_ink_stays_dark(self):
        img = solid(60, 60, (200, 200, 200))
        img[25:35, 25:35] = 20
        out = intensity_scale(img)
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreaterEqual(out[5, 5].min(), 250)
        self.assertLess(out[30, 30].max(), 60)

    def test_all_black_image_does_not_crash(self):
        out = intensity_scale(solid(20, 20, (0, 0, 0)))
        self.assertTrue((out == 0).all())

    def test_flat_image_does_not_crash(self):
        out = intensity_scale(solid(20, 20, (128, 128, 128)))
        self.assertEqual(out.shape, (20, 20, 3))
        self.assertEqual(out.dtype, np.uint8)


class TestSharpen(unittest.TestCase):
    def test_flat_image_is_unchanged(self):
        # Kernel sums to 1, so a flat region must not change
        img = solid(20, 20, (100, 150, 200))
        np.testing.assert_array_equal(sharpen(img), img)

    def test_bright_pixel_gets_brighter(self):
        img = solid(11, 11, (100, 100, 100))
        img[5, 5] = (150, 150, 150)
        self.assertGreater(sharpen(img)[5, 5, 0], 150)

    def test_neighbours_of_bright_pixel_get_darker(self):
        img = solid(11, 11, (100, 100, 100))
        img[5, 5] = (150, 150, 150)
        self.assertLess(sharpen(img)[5, 6, 0], 100)

    def test_shape_and_dtype_preserved(self):
        img = solid(15, 25, (10, 20, 30))
        out = sharpen(img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)


class TestCleanWhiteboard(unittest.TestCase):
    def setUp(self):
        self.board = shaded_board()
        self.board[110:116, 150:330] = 20  # a dark pen stroke

    def test_output_shape_and_dtype(self):
        out = clean_whiteboard(self.board)
        self.assertEqual(out.shape, self.board.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_shadow_is_removed(self):
        out = clean_whiteboard(self.board)
        left = out[20:80, 120:180].mean()
        right = out[20:80, 300:360].mean()
        before = abs(float(self.board[20:80, 120:180].mean()) - float(self.board[20:80, 300:360].mean()))
        self.assertLess(abs(left - right), before / 4)
        self.assertGreater(left, 235)
        self.assertGreater(right, 235)

    def test_ink_survives(self):
        out = clean_whiteboard(self.board)
        self.assertLess(out[112:114, 200:300].mean(), 100)

    def test_is_deterministic(self):
        np.testing.assert_array_equal(clean_whiteboard(self.board), clean_whiteboard(self.board))

    def test_does_not_modify_input(self):
        original = self.board.copy()
        clean_whiteboard(self.board)
        np.testing.assert_array_equal(self.board, original)


if __name__ == "__main__":
    unittest.main()

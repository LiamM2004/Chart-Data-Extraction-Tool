import cv2
import numpy as np
import matplotlib.pyplot as plt


def estimate_background(image, block=24, percentile=90):
    h, w = image.shape[:2]
    block_h = -(-h // block)
    block_w = -(-w // block)

    padded = np.pad(
        image,
        (
            (0, block_h * block - h),
            (0, block_w * block - w),
            (0, 0)
        ),
        mode="edge"
    )

    blocks = (padded.reshape(block_h, block, block_w, block, 3)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(block_h, block_w, block * block, 3))

    grid = np.percentile(blocks, percentile, axis=2).astype(np.uint8)
    grid = np.maximum(grid, cv2.medianBlur(grid, 9))
    grid = cv2.GaussianBlur(grid.astype(np.float32), (0, 0), 2)

    return np.maximum(cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC), 1)


def white_balance(image, bg, saturation=1.4):
    image = np.clip(image / bg * 255, 0, 255).astype(np.uint8)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    image_hsv[..., 1] = np.clip(image_hsv[..., 1] * saturation, 0, 255)
    return cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def yen_threshold(image):
    p = np.bincount(image.ravel(), minlength=256) / image.size
    p1 = np.cumsum(p)
    lo_sq = np.cumsum(p ** 2)
    hi_sq = lo_sq[-1] - lo_sq
    eps = 1e-12
    crit = -np.log(np.maximum(lo_sq * hi_sq, eps)) + 2 * np.log(np.maximum(p1 * (1 - p1), eps))
    return np.argmax(crit)


def intensity_scale(image):
    image_g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    board = image_g[image_g > yen_threshold(image_g)]
    white = max(np.median(board) * 0.97, 1) if board.size else 255
    return np.clip(image.astype(np.float32) / white * 255, 0, 255).astype(np.uint8)


def sharpen(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]],
                       dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def clean_whiteboard(image):
    bg = estimate_background(image)
    balanced = white_balance(image.astype(np.float32), bg)
    cleaned = intensity_scale(balanced)
    return sharpen(cleaned)
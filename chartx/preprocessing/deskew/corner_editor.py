"""A window for dragging the board's four corners into place, for when it can't be found automatically.

    corners = select_corners_manually(photo, initial=best_guess)    # None if cancelled
"""

from __future__ import annotations

import cv2
import numpy as np

from chartx.preprocessing.deskew.geometry import order_corners


class CornerEditor:
    """State and drawing for the drag-the-corners window.

    Kept apart from the OpenCV loop so it can be driven without a display. The photo sits on a
    larger grey canvas so a handle can be dragged off it (for a corner that is out of shot). Each
    side is extended as a guide line, and a magnifier follows the active handle. Coordinates in
    and out are original photo pixels and may lie outside the photo.
    """
    PAD = 0.3          # grey margin around the photo, as a fraction of its size
    GRAB = 25          # px radius for picking up a handle
    LOUPE, ZOOM = 150, 4
    HINT = ("drag handles, or 1-4 to pick + w/a/s/d to nudge", "Enter accept | Esc cancel")
    CANVAS_GREY = 70

    def __init__(self, img: np.ndarray, initial: np.ndarray | None = None, max_size: int = 1000):
        h, w = img.shape[:2]
        total = np.array([w, h]) * (1 + 2 * self.PAD)
        self.scale = min(1.0, max_size / total.max())
        canvas_w, canvas_h = (total * self.scale).round().astype(int)
        self.offset = (np.array([w, h]) * self.PAD * self.scale).round()
        self.base = np.full((canvas_h, canvas_w, 3), self.CANVAS_GREY, np.uint8)
        photo = cv2.resize(img, None, fx=self.scale, fy=self.scale)
        ox, oy = self.offset.astype(int)
        self.base[oy:oy + photo.shape[0], ox:ox + photo.shape[1]] = photo
        self.photo_box = (ox, oy, ox + photo.shape[1] - 1, oy + photo.shape[0] - 1)
        if initial is None:
            initial = self._default_corners(w, h)
        self.handles = [list(order_corners(initial)[i] * self.scale + self.offset) for i in range(4)]
        self.active = 0
        self.dragging = False
        for i, (x, y) in enumerate(self.handles):      # a suggested corner far off the canvas would be unreachable
            self._move_handle(i, x, y)

    @staticmethod
    def _default_corners(w: int, h: int, margin: float = 0.1) -> np.ndarray:
        """A rectangle inset from the photo's edges, as a starting point."""
        return np.array([[w * margin, h * margin], [w * (1 - margin), h * margin],
                         [w * (1 - margin), h * (1 - margin)], [w * margin, h * (1 - margin)]])

    def handle_mouse(self, event, x, y):
        if event == cv2.EVENT_LBUTTONDOWN:
            distances = [np.hypot(x - hx, y - hy) for hx, hy in self.handles]
            if min(distances) < self.GRAB:
                self.active, self.dragging = int(np.argmin(distances)), True
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self._move_handle(self.active, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def handle_key(self, key: int) -> str | None:
        """Handle a key press; returns "accept", "cancel" or None."""
        if key in (13, 10):
            return "accept"
        if key == 27:
            return "cancel"
        if ord("1") <= key <= ord("4"):
            self.active = key - ord("1")
        elif chr(key) in "wasd":
            dx, dy = {"w": (0, -1), "a": (-1, 0), "s": (0, 1), "d": (1, 0)}[chr(key)]
            x, y = self.handles[self.active]
            self._move_handle(self.active, x + dx, y + dy)
        return None

    def _move_handle(self, i: int, x: float, y: float):
        h, w = self.base.shape[:2]
        self.handles[i] = [min(max(x, 0), w - 1), min(max(y, 0), h - 1)]

    def corners(self) -> np.ndarray:
        """Corners (TL, TR, BR, BL) in original image pixels, possibly off-photo."""
        return order_corners((np.array(self.handles, np.float32) - self.offset) / self.scale)

    def render(self) -> np.ndarray:
        frame = self.base.copy()
        x0, y0, x1, y1 = self.photo_box
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), 1)   # where the photo ends
        self._draw_sides_and_guide_lines(frame)
        magnified = self._magnified_view_of_active_handle(frame)       # taken before the handles cover the spot
        self._draw_handles(frame)
        self._draw_magnifier(frame, magnified)
        self._draw_hints(frame)
        return frame

    def _draw_sides_and_guide_lines(self, frame: np.ndarray):
        """The quad's outline, with each side extended across the canvas to line up against board edges."""
        h, w = frame.shape[:2]
        pts = np.array(self.handles)
        for i in range(4):
            p, q = pts[i], pts[(i + 1) % 4]
            d = (q - p) / max(np.linalg.norm(q - p), 1e-6)
            ok, a, b = cv2.clipLine((0, 0, w, h), tuple(int(v) for v in p - d * 5000),
                                    tuple(int(v) for v in p + d * 5000))
            if ok:
                cv2.line(frame, a, b, (0, 220, 255), 1)
        cv2.polylines(frame, [pts.astype(np.int32)], True, (0, 200, 0), 2)

    def _draw_handles(self, frame: np.ndarray):
        for i, (px, py) in enumerate(np.array(self.handles).astype(int)):
            colour = (0, 255, 255) if i == self.active else (0, 0, 255)
            cv2.circle(frame, (px, py), 8, colour, -1)
            cv2.putText(frame, str(i + 1), (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_magnifier(self, frame: np.ndarray, magnified: np.ndarray):
        """Park the magnifier in a corner the active handle isn't near."""
        h, w = frame.shape[:2]
        ax, ay = self.handles[self.active]
        left = w - self.LOUPE - 10 if ax < self.LOUPE + 60 and ay < self.LOUPE + 60 else 10
        frame[10:10 + self.LOUPE, left:left + self.LOUPE] = magnified
        cv2.rectangle(frame, (left, 10), (left + self.LOUPE, 10 + self.LOUPE), (255, 255, 255), 1)

    def _draw_hints(self, frame: np.ndarray):
        h = frame.shape[0]
        for row, text in enumerate(self.HINT):
            for colour, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):   # dark outline, then white text
                cv2.putText(frame, text, (10, h - 34 + 20 * row), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colour, thickness, cv2.LINE_AA)

    def _magnified_view_of_active_handle(self, frame: np.ndarray) -> np.ndarray:
        """Zoomed-in view of `frame` around the active handle, with a crosshair."""
        half = self.LOUPE // (2 * self.ZOOM)
        padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_CONSTANT,
                                    value=(self.CANVAS_GREY,) * 3)
        x, y = (int(round(v)) for v in self.handles[self.active])
        crop = padded[y:y + 2 * half, x:x + 2 * half]
        magnified = cv2.resize(crop, (self.LOUPE, self.LOUPE), interpolation=cv2.INTER_NEAREST)
        centre = self.LOUPE // 2
        cv2.line(magnified, (centre, 0), (centre, self.LOUPE), (0, 0, 255), 1)
        cv2.line(magnified, (0, centre), (self.LOUPE, centre), (0, 0, 255), 1)
        return magnified


def select_corners_manually(img: np.ndarray,
                            initial: np.ndarray | None = None,
                            window: str = "Adjust the 4 corners") -> np.ndarray | None:
    """Show the photo with four draggable corner handles; return the corners, or None if cancelled.

    The corners may lie outside the photo. A web or mobile front end would do the same job in its
    own UI and pass the result to deskew(..., corners=...).
    """
    editor = CornerEditor(img, initial)
    cv2.namedWindow(window)
    accepted = None
    try:
        cv2.setMouseCallback(window, lambda event, x, y, flags, _: editor.handle_mouse(event, x, y))
        while True:
            cv2.imshow(window, editor.render())
            key = cv2.waitKey(20) & 0xFF
            action = editor.handle_key(key) if key != 255 else None
            if action == "accept":
                accepted = editor.corners()
            if action:
                break
    finally:
        cv2.destroyWindow(window)                      # even if interrupted
    return accepted

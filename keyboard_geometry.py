"""US ANSI keyboard geometry projected onto a calibrated camera plane."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from keyboard_layouts import normalize_key_label


BOARD_WIDTH = 15.0
BOARD_HEIGHT = 5.0
FULL_FRAME_ROI = (0.0, 0.0, 1.0, 1.0)

MODIFIER_KEYS = frozenset(
    {
        "shift",
        "shift_l",
        "shift_r",
        "ctrl",
        "ctrl_l",
        "ctrl_r",
        "cmd",
        "cmd_l",
        "cmd_r",
        "alt",
        "alt_l",
        "alt_r",
        "alt_gr",
        "caps_lock",
        "tab",
    }
)

KEY_ALIASES = {
    "shift": "shift_l",
    "ctrl": "ctrl_l",
    "cmd": "cmd_l",
    "alt": "alt_l",
    "return": "enter",
    "~": "`",
    "!": "1",
    "@": "2",
    "#": "3",
    "$": "4",
    "%": "5",
    "^": "6",
    "&": "7",
    "*": "8",
    "(": "9",
    ")": "0",
    "_": "-",
    "+": "=",
    "{": "[",
    "}": "]",
    "|": "\\",
    ":": ";",
    '"': "'",
    "<": ",",
    ">": ".",
    "?": "/",
}


@dataclass(frozen=True, slots=True)
class KeyShape:
    label: str
    polygon: np.ndarray


def is_modifier_key(label: str) -> bool:
    return normalize_key_label(label) in MODIFIER_KEYS


def _row(y: float, entries: list[tuple[str, float]], *, offset: float = 0.0) -> list[KeyShape]:
    keys: list[KeyShape] = []
    x = offset
    gap = 0.06
    for label, width in entries:
        keys.append(
            KeyShape(
                label,
                np.array(
                    [
                        [x + gap, y + gap],
                        [x + width - gap, y + gap],
                        [x + width - gap, y + 1.0 - gap],
                        [x + gap, y + 1.0 - gap],
                    ],
                    dtype=np.float32,
                ),
            )
        )
        x += width
    return keys


def _build_ansi_keys() -> dict[str, KeyShape]:
    rows = [
        _row(
            0.0,
            [("`", 1.0)]
            + [(str(value), 1.0) for value in range(1, 10)]
            + [("0", 1.0), ("-", 1.0), ("=", 1.0), ("backspace", 2.0)],
        ),
        _row(
            1.0,
            [("tab", 1.5)]
            + [(character, 1.0) for character in "qwertyuiop"]
            + [("[", 1.0), ("]", 1.0), ("\\", 1.5)],
        ),
        _row(
            2.0,
            [("caps_lock", 1.75)]
            + [(character, 1.0) for character in "asdfghjkl"]
            + [(";", 1.0), ("'", 1.0), ("enter", 2.25)],
        ),
        _row(
            3.0,
            [("shift_l", 2.25)]
            + [(character, 1.0) for character in "zxcvbnm"]
            + [(",", 1.0), (".", 1.0), ("/", 1.0), ("shift_r", 2.75)],
        ),
        _row(
            4.0,
            [
                ("ctrl_l", 1.25),
                ("cmd_l", 1.25),
                ("alt_l", 1.25),
                ("space", 6.25),
                ("alt_r", 1.25),
                ("cmd_r", 1.25),
                ("menu", 1.25),
                ("ctrl_r", 1.25),
            ],
        ),
    ]
    keys = {shape.label: shape for row in rows for shape in row}
    for alias, target in KEY_ALIASES.items():
        keys[alias] = KeyShape(alias, keys[target].polygon)
    return keys


ANSI_KEYS = _build_ansi_keys()


def normalize_plane_points(points) -> tuple[tuple[float, float], ...] | None:
    if points is None:
        return None
    try:
        array = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if array.shape != (4, 2) or not np.isfinite(array).all():
        return None
    if np.any(array < 0.0) or np.any(array > 1.0):
        return None
    if not cv2.isContourConvex(array.reshape(-1, 1, 2)):
        return None
    if abs(float(cv2.contourArea(array))) < 0.015:
        return None
    # Click order supplies the semantic corner identity. Do not reject a valid
    # convex plane merely because the camera is rotated or strongly oblique.
    return tuple((float(x), float(y)) for x, y in array)


def keyboard_roi_from_points(
    points: tuple[tuple[float, float], ...],
) -> tuple[float, float, float, float]:
    """Return the hand-tracking crop derived from four key-center anchors."""

    return KeyboardPlane(points).inference_roi()


class KeyboardPlane:
    """Perspective projection from four ANSI key centers into the camera image."""

    def __init__(self, image_corners) -> None:
        normalized = normalize_plane_points(image_corners)
        if normalized is None:
            raise ValueError(
                "keyboard anchors must be `, Backspace, right Ctrl, left Ctrl "
                "and form a visible convex plane"
            )
        self.image_corners = normalized
        # Calibration clicks are key centers, not case corners.  This remains
        # valid when an overhead camera shows the keyboard rotated or mirrored.
        source = np.array(
            [
                ANSI_KEYS["`"].polygon.mean(axis=0),
                ANSI_KEYS["backspace"].polygon.mean(axis=0),
                ANSI_KEYS["ctrl_r"].polygon.mean(axis=0),
                ANSI_KEYS["ctrl_l"].polygon.mean(axis=0),
            ],
            dtype=np.float32,
        )
        destination = np.asarray(normalized, dtype=np.float32)
        self.homography = cv2.getPerspectiveTransform(source, destination)
        self._polygons = {
            label: self.project(shape.polygon) for label, shape in ANSI_KEYS.items()
        }

    def project(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
        return cv2.perspectiveTransform(values, self.homography).reshape(-1, 2)

    def key_polygon(self, key_label: str) -> np.ndarray | None:
        return self._polygons.get(normalize_key_label(key_label))

    def key_center(self, key_label: str) -> tuple[float, float] | None:
        polygon = self.key_polygon(key_label)
        if polygon is None:
            return None
        center = polygon.mean(axis=0)
        return float(center[0]), float(center[1])

    def projected_keys(self) -> dict[str, np.ndarray]:
        # Suppress aliases so each physical key is drawn once.
        return {
            label: polygon
            for label, polygon in self._polygons.items()
            if label not in KEY_ALIASES
        }

    def inference_roi(self) -> tuple[float, float, float, float]:
        """Project a keyboard-plus-wrist region and return its clipped bounds."""

        # Extending below the spacebar is essential: MediaPipe needs the palm
        # and wrist, not only fingertips over the keycaps.  Projection through
        # the homography makes this work at any camera rotation.
        extended_plane = np.array(
            [
                [-2.0, -1.2],
                [BOARD_WIDTH + 2.0, -1.2],
                [BOARD_WIDTH + 2.0, BOARD_HEIGHT + 5.0],
                [-2.0, BOARD_HEIGHT + 5.0],
            ],
            dtype=np.float32,
        )
        projected = self.project(extended_plane)
        minimum = projected.min(axis=0)
        maximum = projected.max(axis=0)
        left = float(np.clip(minimum[0], 0.0, 1.0))
        top = float(np.clip(minimum[1], 0.0, 1.0))
        right = float(np.clip(maximum[0], 0.0, 1.0))
        bottom = float(np.clip(maximum[1], 0.0, 1.0))
        if right - left < 0.05 or bottom - top < 0.05:
            raise ValueError("projected keyboard ROI is degenerate")
        return left, top, right - left, bottom - top

    def distance_to_key_px(
        self,
        key_label: str,
        point: tuple[float, float],
        frame_size: tuple[int, int],
    ) -> tuple[float, float] | None:
        polygon = self.key_polygon(key_label)
        if polygon is None:
            return None
        width, height = frame_size
        pixels = polygon * np.array([width, height], dtype=np.float32)
        point_px = (float(point[0]) * width, float(point[1]) * height)
        signed = cv2.pointPolygonTest(pixels.astype(np.float32), point_px, True)
        diagonal = math.hypot(
            float(np.ptp(pixels[:, 0])), float(np.ptp(pixels[:, 1]))
        )
        return max(0.0, -float(signed)), max(diagonal, 1.0)


def draw_keyboard_projection(
    frame: np.ndarray,
    plane: KeyboardPlane | None,
    *,
    view_roi: tuple[float, float, float, float] = FULL_FRAME_ROI,
    active_key: str | None = None,
) -> None:
    if plane is None:
        return
    height, width = frame.shape[:2]
    roi_x, roi_y, roi_width, roi_height = view_roi
    for label, polygon in plane.projected_keys().items():
        view = polygon.copy()
        view[:, 0] = (view[:, 0] - roi_x) / roi_width
        view[:, 1] = (view[:, 1] - roi_y) / roi_height
        pixels = np.rint(view * np.array([width, height])).astype(np.int32)
        selected = active_key is not None and normalize_key_label(active_key) == label
        color = (60, 220, 255) if selected else (80, 120, 150)
        cv2.polylines(frame, [pixels], True, color, 2 if selected else 1, cv2.LINE_AA)

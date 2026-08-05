"""Native PySide6 shell for the real-time touch-typing verifier."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

from qt_bootstrap import prepare_qt_runtime

prepare_qt_runtime()

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCloseEvent, QImage, QKeyEvent, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fetch_model import DEFAULT_MODEL_PATH, ensure_model
from keyboard_geometry import (
    KeyboardPlane,
    keyboard_roi_from_points,
    normalize_plane_points,
)
from keyboard_layouts import LAYOUTS, US_ANSI_QWERTY
from kinematics import normalized_roi
from ui_workers import (
    CameraConfig,
    CameraThread,
    HighResolutionFrameRingBuffer,
    KeyboardThread,
    KeystrokeRecord,
    LatestValueSlot,
    RenderedFrame,
    VisionTelemetry,
    VisionThread,
)


DEFAULT_APP_CONFIG_PATH = Path(__file__).resolve().with_name("app_config.json")


APP_STYLE = """
QWidget {
    background: #0c1118;
    color: #e6edf5;
    font-family: "Segoe UI";
    font-size: 13px;
}
QMainWindow { background: #090d12; }
QGroupBox {
    background: #111923;
    border: 1px solid #243142;
    border-radius: 10px;
    margin-top: 12px;
    padding: 14px 10px 10px 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: #9fb3c8;
}
QComboBox {
    background: #182230;
    border: 1px solid #314158;
    border-radius: 6px;
    padding: 7px 9px;
    min-height: 20px;
}
QComboBox:hover { border-color: #4d6b8d; }
QComboBox QAbstractItemView { background: #182230; selection-background-color: #245b88; }
QPushButton {
    background: #1674b8;
    border: none;
    border-radius: 7px;
    padding: 9px 14px;
    font-weight: 600;
}
QPushButton:hover { background: #2188d0; }
QPushButton:pressed { background: #115d95; }
QPushButton:disabled { background: #293543; color: #788696; }
QTableWidget {
    background: #0f1620;
    alternate-background-color: #121c28;
    border: 1px solid #243142;
    border-radius: 9px;
    gridline-color: #202c3a;
    selection-background-color: #173b57;
}
QHeaderView::section {
    background: #182230;
    color: #aebdcb;
    padding: 8px;
    border: none;
    border-right: 1px solid #273548;
    font-weight: 600;
}
QStatusBar { background: #101721; color: #8fa1b3; border-top: 1px solid #243142; }
QFrame#telemetryCard {
    background: #141e2a;
    border: 1px solid #26374a;
    border-radius: 9px;
}
QLabel#cardTitle { color: #8fa2b5; font-size: 11px; font-weight: 600; }
QLabel#cardValue { color: #f1f7fc; font-size: 23px; font-weight: 700; }
QLabel#postureBanner {
    background: #142536;
    color: #a8d6f4;
    border: 1px solid #245071;
    border-radius: 8px;
    padding: 9px 12px;
    font-weight: 600;
}
"""


def disable_keyboard_focus(root: QWidget) -> None:
    """Keep native Qt controls mouse-only so typing belongs exclusively to pynput."""

    root.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    widgets = root.findChildren(QWidget)
    for widget in widgets:
        widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if isinstance(widget, QPushButton):
            widget.setAutoDefault(False)
            widget.setDefault(False)
        if isinstance(widget, QComboBox):
            widget.view().setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if isinstance(widget, QAbstractSpinBox):
            widget.lineEdit().setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if isinstance(widget, (QAbstractItemView, QAbstractSlider)):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)


class VideoWidget(QWidget):
    """Zero-copy QImage view backed by the current NumPy frame."""

    roi_selected = Signal(object)
    plane_selected = Signal(object)
    plane_progress = Signal(int, str)

    def __init__(self) -> None:
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._packet: RenderedFrame | None = None
        self._paint_times: deque[int] = deque()
        self._render_fps = 0.0
        self._image_rect = QRect()
        self._keyboard_roi: tuple[float, float, float, float] | None = None
        self._selecting_roi = False
        self._selecting_plane = False
        self._plane_clicks: list[QPoint] = []
        self._plane_view_roi = (0.0, 0.0, 1.0, 1.0)
        self._keyboard_plane_points: tuple[tuple[float, float], ...] | None = None
        self._drag_start: QPoint | None = None
        self._drag_current: QPoint | None = None
        self._zoomed = False

    def sizeHint(self) -> QSize:
        return QSize(960, 540)

    def set_frame(self, packet: RenderedFrame) -> None:
        self._packet = packet
        self.update()

    @property
    def has_frame(self) -> bool:
        return self._packet is not None

    def set_keyboard_roi(self, roi) -> None:
        self._keyboard_roi = None if roi is None else tuple(float(v) for v in roi)
        self.update()

    def set_keyboard_plane(self, points) -> None:
        self._keyboard_plane_points = normalize_plane_points(points)
        self.update()

    def set_zoomed(self, zoomed: bool) -> None:
        self._zoomed = zoomed
        self.update()

    def begin_roi_selection(self) -> bool:
        if self._packet is None or self._packet.view_roi != (0.0, 0.0, 1.0, 1.0):
            return False
        self._selecting_roi = True
        self._selecting_plane = False
        self._plane_clicks.clear()
        self._drag_start = self._drag_current = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()
        return True

    def begin_plane_selection(self) -> bool:
        if self._packet is None:
            return False
        self._selecting_roi = False
        self._selecting_plane = True
        self._plane_clicks.clear()
        self._plane_view_roi = tuple(float(value) for value in self._packet.view_roi)
        self._drag_start = self._drag_current = None
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()
        return True

    @property
    def is_full_frame(self) -> bool:
        if self._packet is None:
            return False
        return all(
            abs(actual - expected) < 1e-6
            for actual, expected in zip(
                self._packet.view_roi, (0.0, 0.0, 1.0, 1.0)
            )
        )

    @property
    def selecting_plane(self) -> bool:
        return self._selecting_plane

    def cancel_plane_selection(self) -> None:
        self._selecting_plane = False
        self._plane_clicks.clear()
        self.unsetCursor()
        self.update()

    def _clamp_to_image(self, point: QPoint) -> QPoint:
        return QPoint(
            max(self._image_rect.left(), min(point.x(), self._image_rect.right())),
            max(self._image_rect.top(), min(point.y(), self._image_rect.bottom())),
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            self._selecting_plane
            and event.button() == Qt.MouseButton.LeftButton
            and self._image_rect.contains(event.position().toPoint())
        ):
            point = self._clamp_to_image(event.position().toPoint())
            self._plane_clicks.append(point)
            prompts = (
                "center of the Backspace key",
                "center of the right Ctrl key",
                "center of the left Ctrl key",
                "validating keyboard plane",
            )
            self.plane_progress.emit(
                len(self._plane_clicks), prompts[len(self._plane_clicks) - 1]
            )
            if len(self._plane_clicks) == 4:
                roi_x, roi_y, roi_width, roi_height = self._plane_view_roi
                points = tuple(
                    (
                        roi_x
                        + (
                            (click.x() - self._image_rect.left())
                            / self._image_rect.width()
                        )
                        * roi_width,
                        roi_y
                        + (
                            (click.y() - self._image_rect.top())
                            / self._image_rect.height()
                        )
                        * roi_height,
                    )
                    for click in self._plane_clicks
                )
                self._selecting_plane = False
                self._plane_clicks.clear()
                self.unsetCursor()
                self.plane_selected.emit(points)
            event.accept()
            self.update()
            return
        if (
            self._selecting_roi
            and event.button() == Qt.MouseButton.LeftButton
            and self._image_rect.contains(event.position().toPoint())
        ):
            self._drag_start = self._clamp_to_image(event.position().toPoint())
            self._drag_current = self._drag_start
            event.accept()
            self.update()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._selecting_roi and self._drag_start is not None:
            self._drag_current = self._clamp_to_image(event.position().toPoint())
            event.accept()
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            self._selecting_roi
            and self._drag_start is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            end = self._clamp_to_image(event.position().toPoint())
            selection = QRect(self._drag_start, end).normalized()
            self._selecting_roi = False
            self._drag_start = self._drag_current = None
            self.unsetCursor()
            if selection.width() >= 20 and selection.height() >= 20:
                roi = (
                    (selection.left() - self._image_rect.left()) / self._image_rect.width(),
                    (selection.top() - self._image_rect.top()) / self._image_rect.height(),
                    selection.width() / self._image_rect.width(),
                    selection.height() / self._image_rect.height(),
                )
                self._keyboard_roi = roi
                self.roi_selected.emit(roi)
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    @property
    def render_fps(self) -> float:
        return self._render_fps

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#05080c"))
        if self._packet is None:
            painter.setPen(QColor("#718096"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Starting camera…")
            return

        frame = self._packet.image
        height, width = frame.shape[:2]
        image = QImage(
            frame.data,
            width,
            height,
            int(frame.strides[0]),
            QImage.Format.Format_BGR888,
        )
        scaled = QSize(width, height).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio
        )
        target = QRect(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        )
        self._image_rect = target
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(target, image)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#32d6ff"), 3, Qt.PenStyle.SolidLine))
        if self._selecting_roi and self._drag_start is not None and self._drag_current is not None:
            painter.drawRect(QRect(self._drag_start, self._drag_current).normalized())
        elif self._keyboard_roi is not None and not self._zoomed:
            x, y, roi_width, roi_height = self._keyboard_roi
            roi_rect = QRect(
                target.left() + round(x * target.width()),
                target.top() + round(y * target.height()),
                round(roi_width * target.width()),
                round(roi_height * target.height()),
            )
            painter.drawRect(roi_rect)

        if self._selecting_plane:
            prompts = (
                "1/4 — CLICK the CENTER of the ` key",
                "2/4 — CLICK the CENTER of Backspace",
                "3/4 — CLICK the CENTER of right Ctrl",
                "4/4 — CLICK the CENTER of left Ctrl",
            )
            instruction = prompts[min(len(self._plane_clicks), 3)]
            banner = QRect(
                target.left() + 12,
                target.top() + 12,
                max(100, target.width() - 24),
                48,
            )
            painter.fillRect(banner, QColor(15, 20, 28, 225))
            painter.setPen(QPen(QColor("#ffd166"), 2, Qt.PenStyle.SolidLine))
            painter.drawRect(banner)
            painter.drawText(
                banner.adjusted(12, 0, -12, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                f"KEYBOARD PLANE CALIBRATION: {instruction} — use the MOUSE on this video",
            )
            painter.setPen(QPen(QColor("#ffd166"), 3, Qt.PenStyle.SolidLine))
            for index, point in enumerate(self._plane_clicks):
                painter.drawEllipse(point, 6, 6)
                painter.drawText(point + QPoint(9, -9), str(index + 1))
            for start, end in zip(self._plane_clicks, self._plane_clicks[1:]):
                painter.drawLine(start, end)
        elif self._keyboard_plane_points is not None and not self._zoomed:
            painter.setPen(QPen(QColor("#ffd166"), 2, Qt.PenStyle.SolidLine))
            projected = [
                QPoint(
                    target.left() + round(x * target.width()),
                    target.top() + round(y * target.height()),
                )
                for x, y in self._keyboard_plane_points
            ]
            for start, end in zip(projected, projected[1:] + projected[:1]):
                painter.drawLine(start, end)

        now_ns = time.perf_counter_ns()
        self._paint_times.append(now_ns)
        cutoff = now_ns - 1_000_000_000
        while len(self._paint_times) > 1 and self._paint_times[0] < cutoff:
            self._paint_times.popleft()
        if len(self._paint_times) > 1:
            elapsed = self._paint_times[-1] - self._paint_times[0]
            self._render_fps = (len(self._paint_times) - 1) * 1_000_000_000 / elapsed


class TelemetryCard(QFrame):
    def __init__(self, title: str, unit: str) -> None:
        super().__init__()
        self.setObjectName("telemetryCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(1)
        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        self.value_label = QLabel("—")
        self.value_label.setObjectName("cardValue")
        self.unit_label = QLabel(unit)
        self.unit_label.setObjectName("cardTitle")
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.unit_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class TypingVerifierWindow(QMainWindow):
    def __init__(
        self,
        model_path: Path,
        *,
        processing_height: int,
        inference_fps: float,
        app_config_path: Path = DEFAULT_APP_CONFIG_PATH,
    ):
        super().__init__()
        self.model_path = model_path
        self.processing_height = processing_height
        self.inference_fps = inference_fps
        self.app_config_path = Path(app_config_path)
        self._app_config, self._config_warning = self._load_app_config()
        self.keyboard_roi = normalized_roi(self._app_config.get("keyboard_roi"))
        self.keyboard_plane_points = normalize_plane_points(
            self._app_config.get("keyboard_plane_points")
        )
        if self.keyboard_plane_points is not None:
            try:
                # Recompute this on every launch so older, overly tight crops
                # are repaired when projection geometry improves.
                self.keyboard_roi = keyboard_roi_from_points(
                    self.keyboard_plane_points
                )
            except ValueError:
                self.keyboard_plane_points = None
        self.digital_zoom = bool(
            self._app_config.get("digital_zoom", False) and self.keyboard_roi is not None
        )
        self.camera_thread: CameraThread | None = None
        self.vision_thread: VisionThread | None = None
        self.raw_frames = None
        self.frame_history: HighResolutionFrameRingBuffer | None = None
        self.rendered_frames = None
        self._last_sequence = -1
        self._latest_telemetry: VisionTelemetry | None = None
        self._last_dashboard_update_ns = 0
        self._pending_plane_activation = False
        self._plane_calibration_active = False

        self.setWindowTitle("Real-Time Touch-Typing Verifier")
        self.resize(1500, 920)
        self.setMinimumSize(1120, 720)
        self._build_ui()
        disable_keyboard_focus(self)
        if self._config_warning:
            self.statusBar().showMessage(self._config_warning)

        self.keyboard_thread = KeyboardThread()
        self.keyboard_thread.failed.connect(self._show_error)
        self.keyboard_thread.start()

        self._start_pipeline()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.refresh_timer.setInterval(16)
        self.refresh_timer.timeout.connect(self._refresh)
        self.refresh_timer.start()

    def _load_app_config(self) -> tuple[dict, str | None]:
        defaults = {
            "camera": {
                "camera_idx": 0,
                "backend": "msmf",
                "width": 1920,
                "height": 1080,
                "fps": 60.0,
            },
            "keyboard_roi": None,
            "keyboard_plane_points": None,
            "digital_zoom": False,
        }
        if not self.app_config_path.exists():
            return defaults, None
        try:
            loaded = json.loads(self.app_config_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("configuration root must be an object")
            camera = loaded.get("camera")
            if isinstance(camera, dict):
                candidate = {
                    "camera_idx": int(camera.get("camera_idx", 0)),
                    "backend": str(camera.get("backend", "msmf")).lower(),
                    "width": int(camera.get("width", 1920)),
                    "height": int(camera.get("height", 1080)),
                    "fps": float(camera.get("fps", 60.0)),
                }
                if candidate["camera_idx"] < 0:
                    raise ValueError("camera index cannot be negative")
                if candidate["backend"] not in ("msmf", "dshow"):
                    raise ValueError("camera backend must be msmf or dshow")
                if candidate["width"] <= 0 or candidate["height"] <= 0 or candidate["fps"] <= 0:
                    raise ValueError("camera resolution and FPS must be positive")
                defaults["camera"] = candidate
            loaded_roi = loaded.get("keyboard_roi")
            defaults["keyboard_roi"] = normalized_roi(loaded_roi)
            defaults["keyboard_plane_points"] = normalize_plane_points(
                loaded.get("keyboard_plane_points")
            )
            defaults["digital_zoom"] = bool(
                loaded.get("digital_zoom", False)
                and defaults["keyboard_roi"] is not None
            )
            return defaults, None
        except (OSError, ValueError, TypeError) as exc:
            return defaults, f"Ignoring invalid {self.app_config_path.name}: {exc}"

    def _save_app_config(self) -> None:
        self._app_config["keyboard_roi"] = (
            None if self.keyboard_roi is None else list(self.keyboard_roi)
        )
        self._app_config["keyboard_plane_points"] = (
            None
            if self.keyboard_plane_points is None
            else [list(point) for point in self.keyboard_plane_points]
        )
        self._app_config["digital_zoom"] = self.digital_zoom
        self.app_config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.app_config_path.with_suffix(self.app_config_path.suffix + ".part")
        temporary.write_text(json.dumps(self._app_config, indent=2), encoding="utf-8")
        os.replace(temporary, self.app_config_path)

    def _on_camera_opened(self, config: CameraConfig) -> None:
        self._app_config["camera"] = {
            "camera_idx": config.camera_idx,
            "backend": config.backend,
            "width": config.width,
            "height": config.height,
            "fps": config.fps,
        }
        try:
            self._save_app_config()
        except OSError as exc:
            self.statusBar().showMessage(f"Could not save app settings: {exc}")
            return

        backend_index = self.backend_combo.findData(config.backend)
        if backend_index >= 0:
            self.backend_combo.setCurrentIndex(backend_index)
        resolution = (config.width, config.height)
        resolution_index = self.resolution_combo.findData(resolution)
        if resolution_index < 0:
            self.resolution_combo.addItem(f"{config.width} × {config.height}", resolution)
            resolution_index = self.resolution_combo.count() - 1
        self.resolution_combo.setCurrentIndex(resolution_index)

    def _begin_roi_selection(self) -> None:
        if not self.video.has_frame:
            self._show_error("Wait for the first camera frame before setting the keyboard ROI")
            return
        if self.zoom_checkbox.isChecked():
            self.zoom_checkbox.setChecked(False)
            self.statusBar().showMessage("Returning to the full camera view…")
            QTimer.singleShot(150, self._activate_roi_selection)
            return
        self._activate_roi_selection()

    def _begin_plane_selection(self) -> None:
        if self._pending_plane_activation or self._plane_calibration_active:
            self._cancel_plane_selection()
            return
        if not self.video.has_frame:
            self._show_error("Wait for the first camera frame before calibrating the keyboard plane")
            return
        self._pending_plane_activation = True
        self.plane_button.setText("Cancel Plane Calibration")
        self._show_plane_instruction(
            "Starting mouse calibration on the current camera view… Do not press the reference keys. "
            "You will click their positions in the video with the mouse."
        )
        self._try_activate_plane_selection()

    def _try_activate_plane_selection(self) -> None:
        if not self._pending_plane_activation:
            return
        if self.video.begin_plane_selection():
            self._pending_plane_activation = False
            self._plane_calibration_active = True
            self._show_plane_instruction(
                "Calibration 1/4: use the MOUSE to click the CENTER of the ` key "
                "in the camera preview, regardless of camera rotation."
            )
        else:
            self._show_plane_instruction(
                "Waiting for a camera frame… calibration will start automatically."
            )

    def _on_plane_progress(self, completed: int, next_prompt: str) -> None:
        if completed < 4:
            self._show_plane_instruction(
                f"Calibration {completed + 1}/4: use the MOUSE to click the {next_prompt} "
                "in the camera preview."
            )
        else:
            self._show_plane_instruction("Validating the four-point keyboard plane…")

    def _show_plane_instruction(self, message: str) -> None:
        self._set_posture("warning", message)
        self.statusBar().showMessage(message)

    def _finish_plane_selection_ui(self) -> None:
        self._pending_plane_activation = False
        self._plane_calibration_active = False
        self.plane_button.setText("Calibrate Keyboard Plane (4 points)")

    def _cancel_plane_selection(self) -> None:
        self.video.cancel_plane_selection()
        self._finish_plane_selection_ui()
        self._set_posture("info", "Keyboard-plane calibration cancelled")
        self.statusBar().showMessage("Keyboard-plane calibration cancelled")

    def _on_plane_selected(self, points) -> None:
        self._finish_plane_selection_ui()
        selected = normalize_plane_points(points)
        if selected is None:
            self._show_error(
                "Invalid keyboard plane. Click Calibrate Keyboard Plane to retry in "
                "`, Backspace, right Ctrl, left Ctrl order."
            )
            return
        try:
            KeyboardPlane(selected)
        except ValueError as exc:
            self._show_error(f"Invalid keyboard plane: {exc}")
            return
        self.keyboard_plane_points = selected
        self.keyboard_roi = keyboard_roi_from_points(selected)
        self.video.set_keyboard_plane(selected)
        self.video.set_keyboard_roi(self.keyboard_roi)
        self.zoom_checkbox.setEnabled(True)
        if self.vision_thread is not None:
            self.vision_thread.set_keyboard_plane(selected)
            self.vision_thread.set_keyboard_view(self.keyboard_roi, self.digital_zoom)
        try:
            self._save_app_config()
            message = "Keyboard plane saved; ANSI key polygons are now active"
            self._set_posture("ok", message)
            self.statusBar().showMessage(message)
        except OSError as exc:
            self._show_error(f"Could not save keyboard plane: {exc}")

    def _activate_roi_selection(self) -> None:
        if self.video.begin_roi_selection():
            self.statusBar().showMessage(
                "Drag from one corner of the physical keyboard to the opposite corner"
            )
        else:
            self.statusBar().showMessage("Waiting for the full camera view; click Set Keyboard ROI again")

    def _on_roi_selected(self, roi) -> None:
        selected = normalized_roi(roi)
        if selected is None:
            self._show_error("Keyboard ROI is too small; draw a larger rectangle")
            return
        self.keyboard_roi = selected
        self.video.set_keyboard_roi(selected)
        self.zoom_checkbox.setEnabled(True)
        if self.vision_thread is not None:
            self.vision_thread.set_keyboard_view(selected, self.digital_zoom)
        try:
            self._save_app_config()
            self.statusBar().showMessage("Keyboard ROI saved")
        except OSError as exc:
            self._show_error(f"Could not save keyboard ROI: {exc}")

    def _toggle_digital_zoom(self, enabled: bool) -> None:
        if enabled and self.keyboard_roi is None:
            self.zoom_checkbox.blockSignals(True)
            self.zoom_checkbox.setChecked(False)
            self.zoom_checkbox.blockSignals(False)
            self._show_error("Set the keyboard ROI before enabling digital zoom")
            return
        self.digital_zoom = bool(enabled)
        self.video.set_zoomed(self.digital_zoom)
        if self.vision_thread is not None:
            self.vision_thread.set_keyboard_view(self.keyboard_roi, self.digital_zoom)
        try:
            self._save_app_config()
        except OSError as exc:
            self._show_error(f"Could not save digital zoom setting: {exc}")

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 10)
        root.setSpacing(14)

        left = QVBoxLayout()
        left.setSpacing(10)
        title = QLabel("Live Touch-Typing Analysis")
        title.setStyleSheet("font-size: 21px; font-weight: 700; color: #f2f7fb;")
        self.posture_banner = QLabel("Initializing posture monitor…")
        self.posture_banner.setObjectName("postureBanner")
        self.video = VideoWidget()
        self.video.set_keyboard_roi(self.keyboard_roi)
        self.video.set_keyboard_plane(self.keyboard_plane_points)
        self.video.set_zoomed(self.digital_zoom)
        self.video.roi_selected.connect(self._on_roi_selected)
        self.video.plane_selected.connect(self._on_plane_selected)
        self.video.plane_progress.connect(self._on_plane_progress)
        self.keystroke_table = self._build_keystroke_table()
        left.addWidget(title)
        left.addWidget(self.posture_banner)
        left.addWidget(self.video, 1)
        left.addWidget(QLabel("Recent keystrokes"))
        left.addWidget(self.keystroke_table, 0)

        sidebar = QVBoxLayout()
        sidebar.setSpacing(10)
        sidebar.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        sidebar.addWidget(self._build_controls())
        sidebar.addWidget(self._build_dashboard())
        sidebar.addStretch(1)
        sidebar_widget = QWidget()
        sidebar_widget.setMinimumWidth(320)
        sidebar_widget.setLayout(sidebar)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        sidebar_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setFixedWidth(342)
        sidebar_scroll.setWidget(sidebar_widget)

        root.addLayout(left, 1)
        root.addWidget(sidebar_scroll)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Starting background workers…")

    def _build_controls(self) -> QGroupBox:
        group = QGroupBox("Capture & mapping")
        form = QFormLayout(group)
        form.setSpacing(10)

        self.camera_combo = QComboBox()
        try:
            from PySide6.QtMultimedia import QMediaDevices

            devices = QMediaDevices.videoInputs()
        except BaseException:
            devices = []
        if devices:
            for index, device in enumerate(devices):
                self.camera_combo.addItem(f"{index}: {device.description()}", index)
        else:
            for index in range(4):
                self.camera_combo.addItem(f"Camera {index}", index)

        saved_camera = self._app_config.get("camera", {})
        saved_camera_idx = int(saved_camera.get("camera_idx", 0))
        if self.camera_combo.findData(saved_camera_idx) < 0:
            self.camera_combo.addItem(f"Camera {saved_camera_idx}", saved_camera_idx)
        self.camera_combo.setCurrentIndex(self.camera_combo.findData(saved_camera_idx))

        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Media Foundation (MSMF)", "msmf")
        self.backend_combo.addItem("DirectShow (DSHOW)", "dshow")
        saved_backend = str(saved_camera.get("backend", "msmf")).lower()
        backend_index = self.backend_combo.findData(saved_backend)
        self.backend_combo.setCurrentIndex(max(0, backend_index))

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("1920 × 1080", (1920, 1080))
        self.resolution_combo.addItem("1280 × 720", (1280, 720))
        saved_resolution = (
            int(saved_camera.get("width", 1920)),
            int(saved_camera.get("height", 1080)),
        )
        resolution_index = self.resolution_combo.findData(saved_resolution)
        if resolution_index < 0:
            self.resolution_combo.addItem(
                f"{saved_resolution[0]} × {saved_resolution[1]}", saved_resolution
            )
            resolution_index = self.resolution_combo.count() - 1
        self.resolution_combo.setCurrentIndex(resolution_index)

        self.fps_combo = QComboBox()
        self.fps_combo.addItem("60 FPS", 60.0)
        self.fps_combo.addItem("30 FPS", 30.0)
        saved_fps = float(saved_camera.get("fps", 60.0))
        fps_index = self.fps_combo.findData(saved_fps)
        if fps_index < 0:
            self.fps_combo.addItem(f"{saved_fps:g} FPS", saved_fps)
            fps_index = self.fps_combo.count() - 1
        self.fps_combo.setCurrentIndex(fps_index)

        self.layout_combo = QComboBox()
        for layout_name in LAYOUTS:
            self.layout_combo.addItem(layout_name, layout_name)
        self.layout_combo.setCurrentText(US_ANSI_QWERTY)

        self.apply_button = QPushButton("Apply camera settings")
        self.apply_button.clicked.connect(self._apply_settings)
        self.roi_button = QPushButton("Set Keyboard ROI")
        self.roi_button.clicked.connect(self._begin_roi_selection)
        self.plane_button = QPushButton("Calibrate Keyboard Plane (4 points)")
        # Prevent QFormLayout/sidebar compression from clipping the Segoe UI
        # glyph ascent/descent on Windows display scaling above 100%.
        self.plane_button.setFixedHeight(40)
        self.plane_button.setToolTip(
            "Click key centers in this order: `, Backspace, right Ctrl, left Ctrl"
        )
        self.plane_button.clicked.connect(self._begin_plane_selection)
        plane_help = QLabel(
            "Mouse calibration: click four locations on the live video; do not press those keys."
        )
        plane_help.setWordWrap(True)
        plane_help.setStyleSheet("color:#ffd166;font-size:11px;")
        self.zoom_checkbox = QCheckBox("Digital Zoom to ROI")
        self.zoom_checkbox.setChecked(self.digital_zoom)
        self.zoom_checkbox.setEnabled(self.keyboard_roi is not None)
        self.zoom_checkbox.toggled.connect(self._toggle_digital_zoom)
        form.addRow("Camera", self.camera_combo)
        form.addRow("Backend", self.backend_combo)
        form.addRow("Resolution", self.resolution_combo)
        form.addRow("Target FPS", self.fps_combo)
        form.addRow("Keyboard", self.layout_combo)
        form.addRow(self.apply_button)
        form.addRow(self.roi_button)
        form.addRow(self.plane_button)
        form.addRow(plane_help)
        form.addRow(self.zoom_checkbox)
        form.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        group.setMinimumHeight(group.sizeHint().height())
        group.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        return group

    def _build_dashboard(self) -> QGroupBox:
        group = QGroupBox("Live telemetry")
        grid = QGridLayout(group)
        self.capture_card = TelemetryCard("CAPTURE", "FPS")
        self.render_card = TelemetryCard("RENDER", "FPS")
        self.latency_card = TelemetryCard("MEDIAPIPE", "ms")
        self.delta_card = TelemetryCard("KEY ↔ FRAME", "ms")
        grid.addWidget(self.capture_card, 0, 0)
        grid.addWidget(self.render_card, 0, 1)
        grid.addWidget(self.latency_card, 1, 0)
        grid.addWidget(self.delta_card, 1, 1)
        grid.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        group.setMinimumHeight(group.sizeHint().height())
        group.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        return group

    @staticmethod
    def _build_keystroke_table() -> QTableWidget:
        table = QTableWidget(0, 5)
        table.setHorizontalHeaderLabels(
            ["Timestamp", "Key", "Expected finger", "Observed finger", "Confidence"]
        )
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.verticalHeader().setVisible(False)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        table.setFixedHeight(235)
        return table

    def _selected_config(self) -> CameraConfig:
        width, height = self.resolution_combo.currentData()
        return CameraConfig(
            camera_idx=int(self.camera_combo.currentData()),
            backend=str(self.backend_combo.currentData()),
            width=width,
            height=height,
            fps=float(self.fps_combo.currentData()),
        )

    def _start_pipeline(self) -> None:
        self.raw_frames = LatestValueSlot()
        self.frame_history = HighResolutionFrameRingBuffer(duration_ns=250_000_000)
        self.rendered_frames = LatestValueSlot()
        self._last_sequence = -1
        config = self._selected_config()
        self.camera_thread = CameraThread(config, self.raw_frames, self.frame_history)
        self.camera_thread.ready.connect(self.statusBar().showMessage)
        self.camera_thread.opened.connect(self._on_camera_opened)
        self.camera_thread.camera_error.connect(self._show_error)
        self.camera_thread.failed.connect(self._show_error)
        self.vision_thread = VisionThread(
            model_path=self.model_path,
            raw_frames=self.raw_frames,
            frame_history=self.frame_history,
            rendered_frames=self.rendered_frames,
            camera=self.camera_thread,
            keyboard=self.keyboard_thread,
            processing_height=self.processing_height,
            inference_fps=self.inference_fps,
            keyboard_layout=str(self.layout_combo.currentData()),
            keyboard_roi=self.keyboard_roi,
            keyboard_plane_points=self.keyboard_plane_points,
            digital_zoom=self.digital_zoom,
        )
        self.vision_thread.ready.connect(self.statusBar().showMessage)
        self.vision_thread.failed.connect(self._show_error)
        self.vision_thread.keystroke_ready.connect(self._append_keystroke)
        self.camera_thread.start()
        self.vision_thread.start()

    def _stop_pipeline(self) -> None:
        if self.vision_thread is not None:
            self.vision_thread.stop()
            self.vision_thread.wait(3000)
            self.vision_thread = None
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread.wait(3000)
            self.camera_thread = None

    def _apply_settings(self) -> None:
        self.apply_button.setEnabled(False)
        self.statusBar().showMessage("Restarting capture pipeline…")
        self._stop_pipeline()
        self._start_pipeline()
        self.apply_button.setEnabled(True)

    def _refresh(self) -> None:
        if self.rendered_frames is None:
            return
        packet, _drops = self.rendered_frames.consume()
        if packet is not None and packet.sequence != self._last_sequence:
            self._last_sequence = packet.sequence
            self._latest_telemetry = packet.telemetry
            self.video.set_frame(packet)
            if self._pending_plane_activation and self.video.has_frame:
                self._try_activate_plane_selection()

        now_ns = time.perf_counter_ns()
        if now_ns - self._last_dashboard_update_ns < 100_000_000:
            return
        self._last_dashboard_update_ns = now_ns
        telemetry = self._latest_telemetry
        if telemetry is None:
            return
        self.capture_card.set_value(f"{telemetry.capture_fps:.1f}")
        self.render_card.set_value(f"{self.video.render_fps:.1f}")
        self.latency_card.set_value(f"{telemetry.mediapipe_latency_ms:.1f}")
        self.delta_card.set_value(
            "—" if telemetry.key_frame_delta_ms is None else f"{telemetry.key_frame_delta_ms:+.1f}"
        )
        if self._pending_plane_activation or self._plane_calibration_active:
            return
        self._set_posture(telemetry.posture.severity, telemetry.posture.message)
        self.statusBar().showMessage(
            f"MediaPipe {telemetry.mediapipe_fps:.1f} FPS · "
            f"drops capture/vision/MP: {telemetry.capture_drops}/"
            f"{telemetry.processing_drops}/{telemetry.inference_drops}"
        )

    def _set_posture(self, severity: str, message: str) -> None:
        colors = {
            "ok": ("#123222", "#55d88a", "#26734b"),
            "warning": ("#3b2b11", "#ffd166", "#8d681d"),
            "error": ("#3d171b", "#ff8b95", "#91303a"),
            "info": ("#142536", "#a8d6f4", "#245071"),
        }
        background, foreground, border = colors.get(severity, colors["info"])
        self.posture_banner.setText(message)
        self.posture_banner.setStyleSheet(
            f"background:{background};color:{foreground};border:1px solid {border};"
            "border-radius:8px;padding:9px 12px;font-weight:600;"
        )

    def _show_error(self, message: str) -> None:
        self._set_posture("error", message)
        self.statusBar().showMessage(message)

    def _append_keystroke(self, record: KeystrokeRecord) -> None:
        # Four-point plane calibration is mouse-only. Hide incidental physical
        # key presses so they cannot be mistaken for calibration input.
        if self._pending_plane_activation or self._plane_calibration_active:
            return
        table = self.keystroke_table
        table.insertRow(0)
        observed = record.observed
        if observed == "Unknown" and record.rejection_reason:
            observed = f"Unknown ({record.rejection_reason})"
        decision_tooltip = (
            f"Decision source: {record.decision_source}\n"
            f"Confidence: {record.confidence:.1%}\n"
            f"Guardrail: {record.rejection_reason or 'Accepted'}"
        )
        values = (record.timestamp, record.key, record.expected, observed)
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column in (0, 1):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if column == 3:
                item.setToolTip(decision_tooltip)
            table.setItem(0, column, item)

        confidence = QLabel(f"{record.confidence:.0%}")
        confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        confidence.setToolTip(decision_tooltip)
        if record.confidence >= 0.60:
            colors = ("#153b29", "#65e69b")
        elif record.confidence >= 0.35:
            colors = ("#453713", "#ffd166")
        else:
            colors = ("#3d2024", "#ff9aa3")
        confidence.setStyleSheet(
            f"background:{colors[0]};color:{colors[1]};border-radius:8px;"
            "padding:3px 7px;font-weight:700;margin:3px;"
        )
        table.setCellWidget(0, 4, confidence)
        if table.rowCount() > 200:
            table.removeRow(table.rowCount() - 1)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        event.ignore()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        event.ignore()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.refresh_timer.stop()
        self._stop_pipeline()
        self.keyboard_thread.stop()
        self.keyboard_thread.wait(2000)
        event.accept()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--processing-height", type=int, default=720)
    parser.add_argument("--inference-fps", type=float, default=30.0)
    parser.add_argument(
        "--disable-ml",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--app-config", type=Path, default=DEFAULT_APP_CONFIG_PATH)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.processing_height <= 0 or args.inference_fps <= 0:
        print("processing height and inference FPS must be positive", file=sys.stderr)
        return 2
    try:
        model_path = ensure_model(args.model)
    except BaseException as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        return 1

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Touch-Typing Verifier")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    window = TypingVerifierWindow(
        model_path,
        processing_height=args.processing_height,
        inference_fps=args.inference_fps,
        app_config_path=args.app_config,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

"""Native PySide6 shell for the real-time touch-typing verifier."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

from qt_bootstrap import prepare_qt_runtime

prepare_qt_runtime()

from PySide6.QtCore import QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCloseEvent, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration_collector import (
    DEFAULT_DATA_PATH as DEFAULT_CALIBRATION_PATH,
    DEFAULT_SEQUENCE,
    CalibrationProgress,
)
from fetch_model import DEFAULT_MODEL_PATH, ensure_model
from finger_classifier import DEFAULT_MODEL_PATH as DEFAULT_FINGER_MODEL_PATH
from finger_classifier import TrainingResult
from keyboard_layouts import LAYOUTS, US_ANSI_QWERTY
from ui_workers import (
    CameraConfig,
    CameraThread,
    KeyboardThread,
    KeystrokeRecord,
    LatestValueSlot,
    ModelTrainingThread,
    RenderedFrame,
    VisionTelemetry,
    VisionThread,
)


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
QProgressBar {
    background: #111923;
    border: 1px solid #314158;
    border-radius: 6px;
    min-height: 16px;
    text-align: center;
}
QProgressBar::chunk { background: #2188d0; border-radius: 5px; }
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


class VideoWidget(QWidget):
    """Zero-copy QImage view backed by the current NumPy frame."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._packet: RenderedFrame | None = None
        self._paint_times: deque[int] = deque()
        self._render_fps = 0.0

    def sizeHint(self) -> QSize:
        return QSize(960, 540)

    def set_frame(self, packet: RenderedFrame) -> None:
        self._packet = packet
        self.update()

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
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(target, image)

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


class CalibrationDialog(QDialog):
    cancelled = Signal()

    def __init__(self, sequence: str, repeats: int, parent=None) -> None:
        super().__init__(parent)
        self._finished = False
        self.setWindowTitle("Guided finger calibration")
        self.setModal(False)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        heading = QLabel("Type the highlighted key")
        heading.setStyleSheet("font-size: 18px; font-weight: 700;")
        description = QLabel(
            f"Sequence: {sequence!r} · {repeats} passes\n"
            "Wrong keys are ignored. Keep both hands visible and type one prompt at a time."
        )
        description.setWordWrap(True)
        self.key_label = QLabel("—")
        self.key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.key_label.setStyleSheet(
            "font-size: 54px;font-weight:800;background:#182b3d;"
            "border:2px solid #2188d0;border-radius:12px;padding:18px;"
        )
        self.progress = QProgressBar()
        self.count_label = QLabel("Waiting for collector…")
        self.message_label = QLabel("")
        self.message_label.setWordWrap(True)
        self.cancel_button = QPushButton("Cancel calibration")
        self.cancel_button.clicked.connect(self.reject)

        layout.addWidget(heading)
        layout.addWidget(description)
        layout.addWidget(self.key_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.count_label)
        layout.addWidget(self.message_label)
        layout.addWidget(self.cancel_button)

    def update_progress(self, update: CalibrationProgress) -> None:
        self.progress.setRange(0, max(1, update.total))
        self.progress.setValue(update.index)
        prompt = update.current_key
        self.key_label.setText("SPACE" if prompt == " " else (prompt.upper() or "DONE"))
        self.count_label.setText(
            f"Accepted {update.accepted}/{update.total} · Ignored {update.rejected}"
        )
        self.message_label.setText(update.message)
        if update.complete:
            self._finished = True
            self.key_label.setText("✓")
            self.cancel_button.setText("Close")

    def reject(self) -> None:
        if not self._finished:
            self.cancelled.emit()
        super().reject()


class TypingVerifierWindow(QMainWindow):
    def __init__(
        self,
        model_path: Path,
        *,
        processing_height: int,
        inference_fps: float,
        calibration_path: Path = DEFAULT_CALIBRATION_PATH,
        finger_model_path: Path = DEFAULT_FINGER_MODEL_PATH,
        spatial_threshold_px: float = 120.0,
    ):
        super().__init__()
        self.model_path = model_path
        self.processing_height = processing_height
        self.inference_fps = inference_fps
        self.calibration_path = Path(calibration_path)
        self.finger_model_path = Path(finger_model_path)
        self.spatial_threshold_px = spatial_threshold_px
        self.camera_thread: CameraThread | None = None
        self.vision_thread: VisionThread | None = None
        self.raw_frames = None
        self.rendered_frames = None
        self._last_sequence = -1
        self._latest_telemetry: VisionTelemetry | None = None
        self._last_dashboard_update_ns = 0
        self.calibration_dialog: CalibrationDialog | None = None
        self.training_thread: ModelTrainingThread | None = None

        self.setWindowTitle("Real-Time Touch-Typing Verifier")
        self.resize(1500, 920)
        self.setMinimumSize(1120, 720)
        self._build_ui()

        self.keyboard_thread = KeyboardThread()
        self.keyboard_thread.escape_requested.connect(self.close)
        self.keyboard_thread.failed.connect(self._show_error)
        self.keyboard_thread.start()

        self._start_pipeline()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.refresh_timer.setInterval(16)
        self.refresh_timer.timeout.connect(self._refresh)
        self.refresh_timer.start()

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
        self.keystroke_table = self._build_keystroke_table()
        left.addWidget(title)
        left.addWidget(self.posture_banner)
        left.addWidget(self.video, 1)
        left.addWidget(QLabel("Recent keystrokes"))
        left.addWidget(self.keystroke_table, 0)

        sidebar = QVBoxLayout()
        sidebar.setSpacing(10)
        sidebar.addWidget(self._build_controls())
        sidebar.addWidget(self._build_calibration_controls())
        sidebar.addWidget(self._build_dashboard())
        sidebar.addStretch(1)
        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(330)
        sidebar_widget.setLayout(sidebar)

        root.addLayout(left, 1)
        root.addWidget(sidebar_widget)
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

        self.backend_combo = QComboBox()
        self.backend_combo.addItem("DirectShow (DSHOW)", "dshow")
        self.backend_combo.addItem("Media Foundation (MSMF)", "msmf")

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("1920 × 1080", (1920, 1080))
        self.resolution_combo.addItem("1280 × 720", (1280, 720))

        self.fps_combo = QComboBox()
        self.fps_combo.addItem("60 FPS", 60.0)
        self.fps_combo.addItem("30 FPS", 30.0)

        self.layout_combo = QComboBox()
        for layout_name in LAYOUTS:
            self.layout_combo.addItem(layout_name, layout_name)
        self.layout_combo.setCurrentText(US_ANSI_QWERTY)

        self.apply_button = QPushButton("Apply camera settings")
        self.apply_button.clicked.connect(self._apply_settings)
        form.addRow("Camera", self.camera_combo)
        form.addRow("Backend", self.backend_combo)
        form.addRow("Resolution", self.resolution_combo)
        form.addRow("Target FPS", self.fps_combo)
        form.addRow("Keyboard", self.layout_combo)
        form.addRow(self.apply_button)
        return group

    def _build_calibration_controls(self) -> QGroupBox:
        group = QGroupBox("Personal finger model")
        layout = QVBoxLayout(group)
        explanation = QLabel(
            "Collect key-aligned motion samples, then train a local lightweight classifier."
        )
        explanation.setWordWrap(True)
        self.calibrate_button = QPushButton("Start Guided Calibration")
        self.train_button = QPushButton("Train Model")
        self.calibrate_button.clicked.connect(self._start_calibration)
        self.train_button.clicked.connect(self._train_model)
        layout.addWidget(explanation)
        layout.addWidget(self.calibrate_button)
        layout.addWidget(self.train_button)
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
        self.rendered_frames = LatestValueSlot()
        self._last_sequence = -1
        config = self._selected_config()
        self.camera_thread = CameraThread(config, self.raw_frames)
        self.camera_thread.ready.connect(self.statusBar().showMessage)
        self.camera_thread.camera_error.connect(self._show_error)
        self.camera_thread.failed.connect(self._show_error)
        self.vision_thread = VisionThread(
            model_path=self.model_path,
            raw_frames=self.raw_frames,
            rendered_frames=self.rendered_frames,
            camera=self.camera_thread,
            keyboard=self.keyboard_thread,
            processing_height=self.processing_height,
            inference_fps=self.inference_fps,
            keyboard_layout=str(self.layout_combo.currentData()),
            calibration_path=self.calibration_path,
            finger_model_path=self.finger_model_path,
            spatial_threshold_px=self.spatial_threshold_px,
        )
        self.vision_thread.ready.connect(self.statusBar().showMessage)
        self.vision_thread.failed.connect(self._show_error)
        self.vision_thread.keystroke_ready.connect(self._append_keystroke)
        self.vision_thread.calibration_progress.connect(self._on_calibration_progress)
        self.vision_thread.calibration_finished.connect(self._on_calibration_finished)
        self.vision_thread.model_status.connect(self.statusBar().showMessage)
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

    def _start_calibration(self) -> None:
        if self.vision_thread is None:
            self._show_error("Vision worker is not running")
            return
        if self.calibration_dialog is not None:
            self.calibration_dialog.close()
        dialog = CalibrationDialog(DEFAULT_SEQUENCE, 5, self)
        dialog.cancelled.connect(self._cancel_calibration)
        dialog.finished.connect(lambda _result: self._clear_calibration_dialog(dialog))
        self.calibration_dialog = dialog
        dialog.show()
        dialog.raise_()
        self.vision_thread.start_calibration(DEFAULT_SEQUENCE, 5)

    def _cancel_calibration(self) -> None:
        if self.vision_thread is not None:
            self.vision_thread.cancel_calibration()

    def _clear_calibration_dialog(self, dialog: CalibrationDialog) -> None:
        if self.calibration_dialog is dialog:
            self.calibration_dialog = None

    def _on_calibration_progress(self, update: CalibrationProgress) -> None:
        if self.calibration_dialog is not None:
            self.calibration_dialog.update_progress(update)

    def _on_calibration_finished(self, path: str) -> None:
        self.train_button.setEnabled(True)
        self.statusBar().showMessage(f"Calibration dataset saved: {path}")

    def _train_model(self) -> None:
        if self.training_thread is not None and self.training_thread.isRunning():
            return
        if not self.calibration_path.exists():
            self._show_error("Run Guided Calibration before training the model")
            return
        self.train_button.setEnabled(False)
        self.statusBar().showMessage("Training RandomForest in a background thread…")
        worker = ModelTrainingThread(self.calibration_path, self.finger_model_path)
        worker.completed.connect(self._on_training_complete)
        worker.failed.connect(self._on_training_failed)
        worker.finished.connect(self._release_training_thread)
        self.training_thread = worker
        worker.start()

    def _on_training_complete(self, result: TrainingResult) -> None:
        self.statusBar().showMessage(
            f"Model trained: {result.samples} samples, {result.classes} classes, "
            f"training accuracy {result.training_accuracy:.1%}"
        )
        if self.vision_thread is not None:
            self.vision_thread.request_model_reload()

    def _on_training_failed(self, message: str) -> None:
        self._show_error(message)

    def _release_training_thread(self) -> None:
        self.train_button.setEnabled(True)
        if self.training_thread is not None:
            self.training_thread.deleteLater()
            self.training_thread = None

    def _refresh(self) -> None:
        if self.rendered_frames is None:
            return
        packet, _drops = self.rendered_frames.consume()
        if packet is not None and packet.sequence != self._last_sequence:
            self._last_sequence = packet.sequence
            self._latest_telemetry = packet.telemetry
            self.video.set_frame(packet)

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
        table = self.keystroke_table
        table.insertRow(0)
        values = (record.timestamp, record.key, record.expected, record.observed)
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column in (0, 1):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(0, column, item)

        confidence = QLabel(f"{record.confidence:.0%}")
        confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if record.confidence >= 0.65:
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

    def closeEvent(self, event: QCloseEvent) -> None:
        self.refresh_timer.stop()
        self._stop_pipeline()
        self.keyboard_thread.stop()
        self.keyboard_thread.wait(2000)
        if self.training_thread is not None:
            self.training_thread.wait(5000)
        event.accept()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--processing-height", type=int, default=720)
    parser.add_argument("--inference-fps", type=float, default=30.0)
    parser.add_argument("--calibration-data", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--finger-model", type=Path, default=DEFAULT_FINGER_MODEL_PATH)
    parser.add_argument("--spatial-threshold-px", type=float, default=120.0)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if (
        args.processing_height <= 0
        or args.inference_fps <= 0
        or args.spatial_threshold_px <= 0
    ):
        print("processing height, inference FPS, and spatial threshold must be positive", file=sys.stderr)
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
        calibration_path=args.calibration_data,
        finger_model_path=args.finger_model,
        spatial_threshold_px=args.spatial_threshold_px,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

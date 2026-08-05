"""Qt-owned background workers for capture, hooks, inference, and composition."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar

import cv2
import numpy as np

from qt_bootstrap import prepare_qt_runtime

prepare_qt_runtime()

from PySide6.QtCore import QThread, Signal

from calibration_collector import (
    DEFAULT_DATA_PATH,
    DEFAULT_SEQUENCE,
    GuidedCalibrationCollector,
    extract_feature_window,
)
from finger_classifier import DEFAULT_MODEL_PATH as DEFAULT_FINGER_MODEL_PATH
from finger_classifier import FingerClassifier, train_model
from keyboard_layouts import expected_finger
from kinematics import (
    HandInferencePipeline,
    HybridDecisionEngine,
    KinematicsSnapshot,
    draw_hand_overlay,
)
from mvp_sync import (
    CAMERA_READ_TIMEOUT_S,
    MAX_CONSECUTIVE_READ_FAILURES,
    FramePacket,
    KeyboardCollector,
    RollingRate,
    TimedCaptureReader,
)
from posture import PostureMonitor, PostureStatus


T = TypeVar("T")


class LatestValueSlot(Generic[T]):
    """Thread-safe latest-only handoff with overwrite accounting."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: T | None = None
        self._dirty = False
        self._drops = 0

    def publish(self, item: T) -> None:
        with self._lock:
            if self._dirty:
                self._drops += 1
            self._item = item
            self._dirty = True

    def consume(self) -> tuple[T | None, int]:
        with self._lock:
            item = self._item
            self._dirty = False
            return item, self._drops

    def peek(self) -> T | None:
        with self._lock:
            return self._item

    @property
    def drops(self) -> int:
        with self._lock:
            return self._drops


@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_idx: int = 0
    backend: str = "dshow"
    width: int = 1920
    height: int = 1080
    fps: float = 60.0


@dataclass(frozen=True, slots=True)
class CameraStatus:
    backend: str
    width: int
    height: int
    capture_fps: float
    read_failures: int
    error: str | None


@dataclass(frozen=True, slots=True)
class VisionTelemetry:
    capture_fps: float
    mediapipe_fps: float
    mediapipe_latency_ms: float
    key_frame_delta_ms: float | None
    capture_drops: int
    inference_drops: int
    processing_drops: int
    posture: PostureStatus


@dataclass(frozen=True, slots=True)
class RenderedFrame:
    sequence: int
    timestamp_ns: int
    image: np.ndarray
    telemetry: VisionTelemetry


@dataclass(frozen=True, slots=True)
class KeystrokeRecord:
    timestamp: str
    key: str
    expected: str
    observed: str
    confidence: float


@dataclass(slots=True)
class _PendingKey:
    timestamp_ns: int
    key: str
    expected: str
    wall_timestamp: str
    best_quality: float = float("-inf")
    observed: str = "Unknown"
    confidence: float = 0.0


class CameraThread(QThread):
    ready = Signal(str)
    camera_error = Signal(str)
    failed = Signal(str)

    BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}

    def __init__(self, config: CameraConfig, output: LatestValueSlot[FramePacket]):
        super().__init__()
        self.setObjectName("camera-capture")
        self.config = config
        self.output = output
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._status = CameraStatus(config.backend.upper(), 0, 0, 0.0, 0, None)

    def stop(self) -> None:
        self.requestInterruption()
        self._stop_event.set()

    def status(self) -> CameraStatus:
        with self._state_lock:
            return self._status

    def _set_status(self, **changes) -> None:
        with self._state_lock:
            values = {
                "backend": self._status.backend,
                "width": self._status.width,
                "height": self._status.height,
                "capture_fps": self._status.capture_fps,
                "read_failures": self._status.read_failures,
                "error": self._status.error,
            }
            values.update(changes)
            self._status = CameraStatus(**values)

    def _increment_read_failures(self) -> None:
        with self._state_lock:
            self._status = CameraStatus(
                self._status.backend,
                self._status.width,
                self._status.height,
                self._status.capture_fps,
                self._status.read_failures + 1,
                self._status.error,
            )

    def _backend_candidates(self) -> list[tuple[str, int]]:
        order = (
            ("dshow", "msmf")
            if self.config.backend == "dshow"
            else ("msmf", "dshow")
        )
        return [(name, self.BACKENDS[name]) for name in order]

    def _configure_camera(self, capture: cv2.VideoCapture) -> None:
        # Windows drivers negotiate the compressed media type from this order.
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)

    def run(self) -> None:
        sequence = 0
        failures: list[str] = []
        try:
            candidates = self._backend_candidates()
            for candidate_index, (name, backend_id) in enumerate(candidates):
                if self._stop_event.is_set() or self.isInterruptionRequested():
                    return

                self._set_status(
                    backend=name.upper(), width=0, height=0, capture_fps=0.0, error=None
                )
                capture = cv2.VideoCapture(self.config.camera_idx, backend_id)
                if not capture.isOpened():
                    message = f"{name.upper()} could not open camera {self.config.camera_idx}"
                    failures.append(message)
                    capture.release()
                    if candidate_index + 1 < len(candidates):
                        next_name = candidates[candidate_index + 1][0].upper()
                        self.camera_error.emit(f"{message}; trying {next_name}")
                    continue

                self._configure_camera(capture)
                reader = TimedCaptureReader(capture)
                reader.start()
                rate = RollingRate()
                consecutive_failures = 0
                announced = False
                failure_reason = ""
                try:
                    while not self._stop_event.is_set() and not self.isInterruptionRequested():
                        result = reader.get(CAMERA_READ_TIMEOUT_S)
                        if result is None or not result.ok or result.frame is None:
                            consecutive_failures += 1
                            self._increment_read_failures()
                            if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                                failure_reason = (
                                    f"{name.upper()} opened camera {self.config.camera_idx} but "
                                    f"returned no frame for {MAX_CONSECUTIVE_READ_FAILURES} "
                                    "consecutive reads/timeouts"
                                )
                                break
                            continue

                        consecutive_failures = 0
                        frame = result.frame
                        rate.tick(result.timestamp_ns)
                        height, width = frame.shape[:2]
                        self._set_status(
                            backend=name.upper(),
                            width=width,
                            height=height,
                            capture_fps=rate.value(),
                        )
                        self.output.publish(FramePacket(sequence, result.timestamp_ns, frame))
                        sequence += 1
                        if not announced:
                            announced = True
                            self.ready.emit(
                                f"Camera {self.config.camera_idx} · {name.upper()} · {width}×{height}"
                            )
                finally:
                    clean_close = reader.close()

                if self._stop_event.is_set() or self.isInterruptionRequested():
                    return
                if not clean_close:
                    failure_reason += " (backend reader did not stop cleanly)"
                failures.append(failure_reason or f"{name.upper()} capture ended")
                if candidate_index + 1 < len(candidates):
                    next_name = candidates[candidate_index + 1][0].upper()
                    self.camera_error.emit(f"{failures[-1]}; trying {next_name}")

            detail = "; ".join(failures) if failures else "capture cancelled"
            raise RuntimeError(
                f"Unable to acquire camera {self.config.camera_idx}: {detail}"
            )
        except BaseException as exc:
            message = f"Camera error: {exc}"
            self._set_status(error=message)
            self.failed.emit(message)


class KeyboardThread(QThread):
    escape_requested = Signal()
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("keyboard-hook")
        self._stop_event = threading.Event()
        self.collector = KeyboardCollector(self._stop_event)
        self.events = self.collector.events

    @property
    def dropped(self) -> int:
        return self.collector.dropped

    def stop(self) -> None:
        self.requestInterruption()
        self._stop_event.set()
        if self.collector.listener is not None:
            self.collector.listener.stop()

    def run(self) -> None:
        try:
            self.collector.start()
            while not self.isInterruptionRequested() and not self._stop_event.wait(0.05):
                listener = self.collector.listener
                if listener is not None and not listener.is_alive():
                    raise RuntimeError("keyboard listener stopped unexpectedly")
            if self._stop_event.is_set() and not self.isInterruptionRequested():
                self.escape_requested.emit()
        except BaseException as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(f"Keyboard hook error: {exc}")
        finally:
            self.collector.stop()


class ModelTrainingThread(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, data_path: Path, model_path: Path) -> None:
        super().__init__()
        self.setObjectName("finger-model-training")
        self.data_path = Path(data_path)
        self.model_path = Path(model_path)

    def run(self) -> None:
        try:
            self.completed.emit(train_model(self.data_path, self.model_path))
        except BaseException as exc:
            self.failed.emit(f"Model training failed: {exc}")


class VisionThread(QThread):
    ready = Signal(str)
    failed = Signal(str)
    keystroke_ready = Signal(object)
    calibration_progress = Signal(object)
    calibration_finished = Signal(str)
    model_status = Signal(str)

    def __init__(
        self,
        *,
        model_path: Path,
        raw_frames: LatestValueSlot[FramePacket],
        rendered_frames: LatestValueSlot[RenderedFrame],
        camera: CameraThread,
        keyboard: KeyboardThread,
        processing_height: int = 720,
        inference_fps: float = 30.0,
        keyboard_layout: str,
        calibration_path: Path = DEFAULT_DATA_PATH,
        finger_model_path: Path = DEFAULT_FINGER_MODEL_PATH,
        spatial_threshold_px: float = 120.0,
    ) -> None:
        super().__init__()
        self.setObjectName("vision-compositor")
        self.model_path = model_path
        self.raw_frames = raw_frames
        self.rendered_frames = rendered_frames
        self.camera = camera
        self.keyboard = keyboard
        self.processing_height = processing_height
        self.inference_fps = inference_fps
        self.keyboard_layout = keyboard_layout
        self.calibration_path = Path(calibration_path)
        self.finger_model_path = Path(finger_model_path)
        self.spatial_threshold_px = spatial_threshold_px
        self._stop_event = threading.Event()
        self._control_lock = threading.Lock()
        self._calibration_command: tuple[str, str, int] | None = None
        self._reload_model = False

    def start_calibration(self, sequence: str = DEFAULT_SEQUENCE, repeats: int = 5) -> None:
        with self._control_lock:
            self._calibration_command = ("start", sequence, repeats)

    def cancel_calibration(self) -> None:
        with self._control_lock:
            self._calibration_command = ("cancel", "", 0)

    def request_model_reload(self) -> None:
        with self._control_lock:
            self._reload_model = True

    def stop(self) -> None:
        self.requestInterruption()
        self._stop_event.set()

    @staticmethod
    def _resize(frame: np.ndarray, maximum_height: int) -> np.ndarray:
        height, width = frame.shape[:2]
        if height <= maximum_height:
            return frame.copy()
        scale = maximum_height / height
        return cv2.resize(
            frame,
            (max(1, round(width * scale)), maximum_height),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _update_pending(
        pending: list[_PendingKey], snapshot: KinematicsSnapshot
    ) -> None:
        if snapshot.timestamp_ns == 0:
            return
        for item in pending:
            delta = abs(snapshot.timestamp_ns - item.timestamp_ns)
            if delta > 80_000_000:
                continue
            candidates = [
                (hand.active_score, hand.handedness, hand.active_finger)
                for hand in snapshot.hands
                if hand.active_finger is not None
            ]
            score, handedness, finger = max(candidates, default=(0.0, None, None))
            quality = score * (1.0 - 0.35 * delta / 80_000_000)
            if quality > item.best_quality:
                item.best_quality = quality
                item.confidence = score
                item.observed = (
                    f"{handedness} {finger}" if score >= 0.28 and finger else "Unknown"
                )

    def _drain_keys(
        self,
        pipeline: HandInferencePipeline,
        pending: list[_PendingKey],
        wall_clock_offset_ns: int,
        collector: GuidedCalibrationCollector,
    ) -> float | None:
        latest_delta: float | None = None
        while True:
            try:
                event = self.keyboard.events.get_nowait()
            except queue.Empty:
                return latest_delta
            if event.action != "press":
                continue
            associated = self.raw_frames.peek()
            if associated is not None:
                latest_delta = (event.timestamp_ns - associated.timestamp_ns) / 1_000_000
            pipeline.engine.register_keypress(event.timestamp_ns, event.label)
            calibration_update = collector.handle_keypress(event.timestamp_ns, event.label)
            if calibration_update is not None:
                self.calibration_progress.emit(calibration_update)
            wall_ns = event.timestamp_ns + wall_clock_offset_ns
            timestamp = datetime.fromtimestamp(wall_ns / 1_000_000_000).strftime("%H:%M:%S.%f")[:-3]
            pending.append(
                _PendingKey(
                    event.timestamp_ns,
                    event.label,
                    expected_finger(event.label, self.keyboard_layout),
                    timestamp,
                )
            )

    def _handle_control_commands(
        self,
        collector: GuidedCalibrationCollector,
        decision: HybridDecisionEngine,
    ) -> None:
        with self._control_lock:
            command = self._calibration_command
            self._calibration_command = None
            reload_model = self._reload_model
            self._reload_model = False

        if command is not None:
            action, sequence, repeats = command
            update = (
                collector.start(sequence, repeats)
                if action == "start"
                else collector.cancel()
            )
            self.calibration_progress.emit(update)

        if reload_model:
            try:
                decision.classifier = FingerClassifier.load(self.finger_model_path)
                self.model_status.emit(f"Loaded {self.finger_model_path.name}")
            except BaseException as exc:
                decision.classifier = None
                self.model_status.emit(f"Finger model unavailable: {exc}")

    def run(self) -> None:
        pipeline: HandInferencePipeline | None = None
        posture = PostureMonitor()
        collector = GuidedCalibrationCollector(
            self.calibration_path, keyboard_layout=self.keyboard_layout
        )
        decision = HybridDecisionEngine(spatial_threshold_px=self.spatial_threshold_px)
        pending: list[_PendingKey] = []
        last_frame_sequence = -1
        last_analysis_timestamp = -1
        latest_key_delta: float | None = None
        wall_clock_offset_ns = time.time_ns() - time.perf_counter_ns()
        try:
            pipeline = HandInferencePipeline(
                self.model_path,
                num_hands=2,
                max_fps=self.inference_fps,
            )
            if self.finger_model_path.exists():
                try:
                    decision.classifier = FingerClassifier.load(self.finger_model_path)
                    self.model_status.emit(f"Loaded {self.finger_model_path.name}")
                except BaseException as exc:
                    self.model_status.emit(f"Finger model unavailable: {exc}")
            self.ready.emit("MediaPipe CPU pipeline ready")

            while not self._stop_event.is_set() and not self.isInterruptionRequested():
                self._handle_control_commands(collector, decision)
                new_delta = self._drain_keys(
                    pipeline, pending, wall_clock_offset_ns, collector
                )
                if new_delta is not None:
                    latest_key_delta = new_delta

                packet, capture_drops = self.raw_frames.consume()
                if packet is None or packet.sequence == last_frame_sequence:
                    self.msleep(1)
                    continue
                last_frame_sequence = packet.sequence

                display = self._resize(packet.image, self.processing_height)
                pipeline.submit(display, packet.timestamp_ns)
                snapshot = pipeline.engine.snapshot()
                self._update_pending(pending, snapshot)

                calibration_update = collector.process_ready(
                    pipeline.engine, packet.timestamp_ns
                )
                if calibration_update is not None:
                    self.calibration_progress.emit(calibration_update)
                    if calibration_update.complete:
                        self.calibration_finished.emit(str(self.calibration_path))

                for item in list(pending):
                    if packet.timestamp_ns - item.timestamp_ns >= 110_000_000:
                        observed = item.observed
                        confidence = item.confidence
                        window = extract_feature_window(pipeline.engine, item.timestamp_ns)
                        if window is not None and decision.classifier is not None:
                            result = decision.classify(
                                window.values,
                                item.key,
                                window.fingertip_positions,
                                (display.shape[1], display.shape[0]),
                                heuristic_label=item.observed,
                                heuristic_confidence=item.confidence,
                            )
                            observed = result.label
                            confidence = result.confidence
                        self.keystroke_ready.emit(
                            KeystrokeRecord(
                                item.wall_timestamp,
                                item.key,
                                item.expected,
                                observed,
                                confidence,
                            )
                        )
                        pending.remove(item)

                if snapshot.timestamp_ns != last_analysis_timestamp:
                    posture.update(snapshot, time.perf_counter_ns())
                    last_analysis_timestamp = snapshot.timestamp_ns
                draw_hand_overlay(display, snapshot)

                camera_status = self.camera.status()
                mp_fps, mp_latency, mp_drops, mp_error = pipeline.stats()
                if mp_error:
                    raise RuntimeError(mp_error)
                telemetry = VisionTelemetry(
                    camera_status.capture_fps,
                    mp_fps,
                    mp_latency,
                    latest_key_delta,
                    capture_drops,
                    mp_drops,
                    self.rendered_frames.drops,
                    posture.status,
                )
                self.rendered_frames.publish(
                    RenderedFrame(packet.sequence, packet.timestamp_ns, display, telemetry)
                )
        except BaseException as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(f"Vision pipeline error: {exc}")
        finally:
            if pipeline is not None:
                pipeline.close()

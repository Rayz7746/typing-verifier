"""Qt-owned background workers for capture, hooks, inference, and composition."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
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
from keyboard_geometry import (
    KeyboardPlane,
    draw_keyboard_projection,
    is_modifier_key,
)
from keyboard_layouts import expected_finger
from kinematics import (
    FULL_FRAME_ROI,
    HandInferencePipeline,
    HybridDecisionEngine,
    KeyConditionedContactVerifier,
    KinematicsSnapshot,
    draw_hand_overlay,
    normalized_roi,
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


class HighResolutionFrameRingBuffer:
    """Thread-safe rolling camera history used for key-aligned verification."""

    def __init__(self, duration_ns: int = 250_000_000, maximum_frames: int = 64) -> None:
        self.duration_ns = duration_ns
        self._frames: deque[FramePacket] = deque(maxlen=maximum_frames)
        self._lock = threading.Lock()

    def append(self, packet: FramePacket) -> None:
        with self._lock:
            self._frames.append(packet)
            cutoff = packet.timestamp_ns - self.duration_ns
            while self._frames and self._frames[0].timestamp_ns < cutoff:
                self._frames.popleft()

    def window(
        self,
        center_timestamp_ns: int,
        *,
        before_ns: int = 150_000_000,
        after_ns: int = 50_000_000,
    ) -> tuple[FramePacket, ...]:
        start = center_timestamp_ns - before_ns
        end = center_timestamp_ns + after_ns
        with self._lock:
            return tuple(
                packet
                for packet in self._frames
                if start <= packet.timestamp_ns <= end
            )


@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_idx: int = 0
    backend: str = "msmf"
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
    view_roi: tuple[float, float, float, float] = FULL_FRAME_ROI


@dataclass(frozen=True, slots=True)
class KeystrokeRecord:
    timestamp: str
    key: str
    expected: str
    observed: str
    confidence: float
    rejection_reason: str = ""
    decision_source: str = "heuristic"


@dataclass(slots=True)
class _PendingKey:
    timestamp_ns: int
    key: str
    expected: str
    wall_timestamp: str
    frames_before: tuple[FramePacket, ...] = ()
    best_quality: float = float("-inf")
    observed: str = "Unknown"
    confidence: float = 0.0


class CameraThread(QThread):
    ready = Signal(str)
    opened = Signal(object)
    camera_error = Signal(str)
    failed = Signal(str)

    BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}

    def __init__(
        self,
        config: CameraConfig,
        output: LatestValueSlot[FramePacket],
        frame_history: HighResolutionFrameRingBuffer | None = None,
    ):
        super().__init__()
        self.setObjectName("camera-capture")
        self.config = config
        self.output = output
        self.frame_history = frame_history
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
                        packet = FramePacket(sequence, result.timestamp_ns, frame)
                        if self.frame_history is not None:
                            self.frame_history.append(packet)
                        self.output.publish(packet)
                        sequence += 1
                        if not announced:
                            announced = True
                            self.opened.emit(
                                CameraConfig(
                                    self.config.camera_idx,
                                    name,
                                    width,
                                    height,
                                    self.config.fps,
                                )
                            )
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
        self.collector = KeyboardCollector(self._stop_event, stop_on_escape=False)
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
        frame_history: HighResolutionFrameRingBuffer,
        rendered_frames: LatestValueSlot[RenderedFrame],
        camera: CameraThread,
        keyboard: KeyboardThread,
        processing_height: int = 720,
        inference_fps: float = 30.0,
        keyboard_layout: str,
        calibration_path: Path = DEFAULT_DATA_PATH,
        finger_model_path: Path = DEFAULT_FINGER_MODEL_PATH,
        spatial_threshold_px: float = 120.0,
        keyboard_roi: tuple[float, float, float, float] | None = None,
        keyboard_plane_points: tuple[tuple[float, float], ...] | None = None,
        digital_zoom: bool = False,
        disable_ml: bool = False,
    ) -> None:
        super().__init__()
        self.setObjectName("vision-compositor")
        self.model_path = model_path
        self.raw_frames = raw_frames
        self.frame_history = frame_history
        self.rendered_frames = rendered_frames
        self.camera = camera
        self.keyboard = keyboard
        self.processing_height = processing_height
        self.inference_fps = inference_fps
        self.keyboard_layout = keyboard_layout
        self.calibration_path = Path(calibration_path)
        self.finger_model_path = Path(finger_model_path)
        self.spatial_threshold_px = spatial_threshold_px
        self._keyboard_roi = normalized_roi(keyboard_roi)
        self._keyboard_plane_points = keyboard_plane_points
        self._digital_zoom = bool(digital_zoom and self._keyboard_roi is not None)
        self.disable_ml = disable_ml
        self._stop_event = threading.Event()
        self._control_lock = threading.Lock()
        self._calibration_command: tuple[str, str, int, str] | None = None
        self._reload_model = False

    def start_calibration(
        self,
        sequence: str = DEFAULT_SEQUENCE,
        repeats: int = 5,
        set_name: str = "Custom",
    ) -> None:
        with self._control_lock:
            self._calibration_command = ("start", sequence, repeats, set_name)

    def cancel_calibration(self) -> None:
        with self._control_lock:
            self._calibration_command = ("cancel", "", 0, "")

    def set_keyboard_view(
        self,
        roi: tuple[float, float, float, float] | None,
        digital_zoom: bool,
    ) -> None:
        with self._control_lock:
            self._keyboard_roi = normalized_roi(roi)
            self._digital_zoom = bool(digital_zoom and self._keyboard_roi is not None)

    def set_keyboard_plane(self, points) -> None:
        with self._control_lock:
            self._keyboard_plane_points = points

    def request_model_reload(self) -> None:
        with self._control_lock:
            self._reload_model = True

    def stop(self) -> None:
        self.requestInterruption()
        self._stop_event.set()

    @staticmethod
    def _resize(frame: np.ndarray, maximum_height: int) -> np.ndarray:
        height, width = frame.shape[:2]
        if height == maximum_height:
            return frame.copy()
        scale = maximum_height / height
        return cv2.resize(
            frame,
            (max(1, round(width * scale)), maximum_height),
            interpolation=(
                cv2.INTER_AREA if height > maximum_height else cv2.INTER_LINEAR
            ),
        )

    @staticmethod
    def _crop_to_roi(
        frame: np.ndarray,
        roi: tuple[float, float, float, float] | None,
        enabled: bool,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        if not enabled or roi is None:
            return frame, FULL_FRAME_ROI
        height, width = frame.shape[:2]
        x, y, roi_width, roi_height = roi
        left = int(round(x * width))
        top = int(round(y * height))
        right = int(round((x + roi_width) * width))
        bottom = int(round((y + roi_height) * height))
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        actual_roi = (
            left / width,
            top / height,
            (right - left) / width,
            (bottom - top) / height,
        )
        return frame[top:bottom, left:right], actual_roi

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
            wall_ns = event.timestamp_ns + wall_clock_offset_ns
            timestamp = datetime.fromtimestamp(
                wall_ns / 1_000_000_000
            ).strftime("%H:%M:%S.%f")[:-3]
            if is_modifier_key(event.label):
                self.keystroke_ready.emit(
                    KeystrokeRecord(
                        timestamp,
                        event.label,
                        "Modifier",
                        "Modifier",
                        1.0,
                        "Contact verification skipped for held modifier",
                        "modifier",
                    )
                )
                continue
            pipeline.engine.register_keypress(event.timestamp_ns, event.label)
            calibration_update = collector.handle_keypress(event.timestamp_ns, event.label)
            if calibration_update is not None:
                self.calibration_progress.emit(calibration_update)
            pending.append(
                _PendingKey(
                    event.timestamp_ns,
                    event.label,
                    expected_finger(event.label, self.keyboard_layout),
                    timestamp,
                    self.frame_history.window(
                        event.timestamp_ns,
                        before_ns=150_000_000,
                        after_ns=0,
                    ),
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
            action, sequence, repeats, set_name = command
            update = (
                collector.start(sequence, repeats, set_name)
                if action == "start"
                else collector.cancel()
            )
            self.calibration_progress.emit(update)

        if reload_model and self.disable_ml:
            self.model_status.emit("ML disabled; model reload skipped")
        elif reload_model:
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
        decision = HybridDecisionEngine(
            spatial_threshold_px=self.spatial_threshold_px,
            keyboard_roi=self._keyboard_roi,
        )
        contact_verifier = KeyConditionedContactVerifier()
        pending: list[_PendingKey] = []
        last_frame_sequence = -1
        last_analysis_timestamp = -1
        latest_key_delta: float | None = None
        keyboard_plane: KeyboardPlane | None = None
        applied_plane_points: object = object()
        wall_clock_offset_ns = time.time_ns() - time.perf_counter_ns()
        try:
            pipeline = HandInferencePipeline(
                self.model_path,
                num_hands=2,
                max_fps=self.inference_fps,
            )
            if not self.disable_ml and self.finger_model_path.exists():
                try:
                    decision.classifier = FingerClassifier.load(self.finger_model_path)
                    self.model_status.emit(
                        f"Loaded legacy fallback {self.finger_model_path.name}"
                    )
                except BaseException as exc:
                    self.model_status.emit(f"Finger model unavailable: {exc}")
            elif self.disable_ml:
                self.model_status.emit("ML disabled: key-conditioned contact mode only")
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

                with self._control_lock:
                    keyboard_roi = self._keyboard_roi
                    keyboard_plane_points = self._keyboard_plane_points
                    digital_zoom = self._digital_zoom
                decision.set_keyboard_roi(keyboard_roi)
                if keyboard_plane_points != applied_plane_points:
                    try:
                        keyboard_plane = (
                            KeyboardPlane(keyboard_plane_points)
                            if keyboard_plane_points is not None
                            else None
                        )
                    except ValueError:
                        keyboard_plane = None
                    applied_plane_points = keyboard_plane_points
                source, view_roi = self._crop_to_roi(
                    packet.image, keyboard_roi, digital_zoom
                )
                display = self._resize(source, self.processing_height)
                pipeline.submit(display, packet.timestamp_ns, view_roi)
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
                    if (
                        packet.timestamp_ns - item.timestamp_ns >= 80_000_000
                        and snapshot.timestamp_ns >= item.timestamp_ns + 30_000_000
                    ):
                        observed = item.observed
                        confidence = item.confidence
                        rejection_reason = ""
                        decision_source = "heuristic"
                        if keyboard_plane is not None:
                            history = pipeline.engine.history_window(
                                item.timestamp_ns,
                                before_ns=150_000_000,
                                after_ns=50_000_000,
                            )
                            current_frames = self.frame_history.window(item.timestamp_ns)
                            by_sequence = {
                                frame.sequence: frame
                                for frame in (*item.frames_before, *current_frames)
                            }
                            frame_window = tuple(
                                sorted(
                                    by_sequence.values(),
                                    key=lambda frame: frame.timestamp_ns,
                                )
                            )
                            result = contact_verifier.verify(
                                item.key,
                                item.timestamp_ns,
                                history,
                                keyboard_plane,
                                (packet.image.shape[1], packet.image.shape[0]),
                                frame_timestamps=tuple(
                                    frame.timestamp_ns for frame in frame_window
                                ),
                            )
                            observed = result.label
                            confidence = result.confidence
                            rejection_reason = (
                                result.reason if result.label == "Unknown" else ""
                            )
                            decision_source = "contact"
                        elif not self.disable_ml:
                            window = extract_feature_window(pipeline.engine, item.timestamp_ns)
                            if window is None:
                                observed = "Unknown"
                                confidence = 0.0
                                rejection_reason = "Hand Occluded"
                                decision_source = "guardrail"
                            else:
                                result = decision.classify(
                                    window.values,
                                    item.key,
                                    window.fingertip_positions,
                                    (packet.image.shape[1], packet.image.shape[0]),
                                    heuristic_label=item.observed,
                                    heuristic_confidence=item.confidence,
                                )
                                observed = result.label
                                confidence = result.confidence
                                rejection_reason = (
                                    result.reason if result.label == "Unknown" else ""
                                )
                                decision_source = f"legacy-{result.source}"
                        else:
                            observed = "Unknown"
                            confidence = 0.0
                            rejection_reason = "Keyboard Plane Not Calibrated"
                            decision_source = "contact"
                        self.keystroke_ready.emit(
                            KeystrokeRecord(
                                item.wall_timestamp,
                                item.key,
                                item.expected,
                                observed,
                                confidence,
                                rejection_reason,
                                decision_source,
                            )
                        )
                        pending.remove(item)

                if snapshot.timestamp_ns != last_analysis_timestamp:
                    posture.update(snapshot, time.perf_counter_ns())
                    last_analysis_timestamp = snapshot.timestamp_ns
                active_key = snapshot.key_prediction.key_label if snapshot.key_prediction else None
                draw_keyboard_projection(
                    display,
                    keyboard_plane,
                    view_roi=view_roi,
                    active_key=active_key,
                )
                draw_hand_overlay(display, snapshot, view_roi)

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
                    RenderedFrame(
                        packet.sequence,
                        packet.timestamp_ns,
                        display,
                        telemetry,
                        view_roi,
                    )
                )
        except BaseException as exc:
            if not self.isInterruptionRequested():
                self.failed.emit(f"Vision pipeline error: {exc}")
        finally:
            if pipeline is not None:
                pipeline.close()

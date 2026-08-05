"""MediaPipe LIVE_STREAM inference, palm-local kinematics, and hand overlays."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from keyboard_geometry import KeyboardPlane


NS_PER_SECOND = 1_000_000_000
KEY_MATCH_WINDOW_NS = 80_000_000
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
FINGER_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
TIP_INDICES = {name: joints[-1] for name, joints in FINGER_JOINTS.items()}
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
FINGER_COLORS = {
    "thumb": (255, 180, 60),
    "index": (70, 240, 90),
    "middle": (80, 220, 255),
    "ring": (255, 130, 220),
    "pinky": (180, 120, 255),
}
FULL_FRAME_ROI = (0.0, 0.0, 1.0, 1.0)


def normalized_roi(roi) -> tuple[float, float, float, float] | None:
    if roi is None or len(roi) != 4:
        return None
    x, y, width, height = (float(value) for value in roi)
    x = float(np.clip(x, 0.0, 1.0))
    y = float(np.clip(y, 0.0, 1.0))
    width = float(np.clip(width, 0.0, 1.0 - x))
    height = float(np.clip(height, 0.0, 1.0 - y))
    if width < 0.02 or height < 0.02:
        return None
    return x, y, width, height


class OneEuroFilter:
    """Vectorized adaptive low-pass filter with high-speed responsiveness."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 20.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._time_s: float | None = None
        self._raw: np.ndarray | None = None
        self._filtered: np.ndarray | None = None
        self._derivative: np.ndarray | None = None

    @staticmethod
    def _alpha(dt: float, cutoff: float | np.ndarray) -> float | np.ndarray:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._time_s = self._raw = self._filtered = self._derivative = None

    def filter(self, values: np.ndarray, time_s: float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if self._time_s is None or self._raw is None or time_s - self._time_s > 0.25:
            self._time_s = time_s
            self._raw = values.copy()
            self._filtered = values.copy()
            self._derivative = np.zeros_like(values)
            return values.copy()

        dt = max(time_s - self._time_s, 1e-6)
        raw_derivative = (values - self._raw) / dt
        derivative_alpha = self._alpha(dt, self.d_cutoff)
        derivative = derivative_alpha * raw_derivative + (1.0 - derivative_alpha) * self._derivative
        cutoff = self.min_cutoff + self.beta * np.abs(derivative)
        signal_alpha = self._alpha(dt, cutoff)
        filtered = signal_alpha * values + (1.0 - signal_alpha) * self._filtered

        self._time_s = time_s
        self._raw = values.copy()
        self._filtered = filtered
        self._derivative = derivative
        return filtered.copy()


@dataclass(frozen=True, slots=True)
class HandAnalysis:
    handedness: str
    handedness_score: float
    timestamp_ns: int
    image_landmarks: np.ndarray
    local_landmarks: np.ndarray
    palm_normal: np.ndarray
    tip_velocity_image: dict[str, np.ndarray]
    tip_velocity_local: dict[str, np.ndarray]
    downward_velocity: dict[str, float]
    pip_flexion_rate: dict[str, float]
    dip_flexion_rate: dict[str, float]
    flexion_rate: dict[str, float]
    finger_scores: dict[str, float]
    active_finger: str | None
    active_score: float


@dataclass(frozen=True, slots=True)
class KeyPrediction:
    key_label: str
    key_timestamp_ns: int
    frame_timestamp_ns: int
    handedness: str | None
    finger: str | None
    score: float

    @property
    def delta_ms(self) -> float:
        return (self.key_timestamp_ns - self.frame_timestamp_ns) / 1_000_000


@dataclass(frozen=True, slots=True)
class KinematicsSnapshot:
    timestamp_ns: int
    hands: tuple[HandAnalysis, ...]
    key_prediction: KeyPrediction | None


@dataclass(frozen=True, slots=True)
class ContactVerification:
    """Result of a key-conditioned, model-free contact decision."""

    label: str
    confidence: float
    reason: str
    proximity_score: float = 0.0
    stroke_score: float = 0.0
    reversal_score: float = 0.0
    distance_px: float | None = None


class KeyConditionedContactVerifier:
    """Identify the finger whose trajectory best explains a known keydown.

    The OS hook supplies the target key.  The verifier therefore compares each
    visible fingertip only against that key's projected polygon instead of
    trying to learn a global pose-to-finger mapping.
    """

    def __init__(self, *, minimum_score: float = 0.42, ambiguity_margin: float = 0.035):
        self.minimum_score = minimum_score
        self.ambiguity_margin = ambiguity_margin

    @staticmethod
    def _median(values: list[float]) -> float | None:
        return None if not values else float(np.median(np.asarray(values)))

    def verify(
        self,
        key_label: str,
        key_timestamp_ns: int,
        snapshots: tuple[KinematicsSnapshot, ...],
        keyboard_plane: KeyboardPlane | None,
        frame_size: tuple[int, int],
        *,
        frame_timestamps: tuple[int, ...] = (),
    ) -> ContactVerification:
        if keyboard_plane is None:
            return ContactVerification("Unknown", 0.0, "Keyboard Plane Not Calibrated")
        if keyboard_plane.key_polygon(key_label) is None:
            return ContactVerification("Unknown", 0.0, "Unmapped Key")
        if frame_timestamps:
            relative = [timestamp - key_timestamp_ns for timestamp in frame_timestamps]
            # Allow roughly one 30 FPS interval of timestamp quantization at
            # either edge of the requested [-150 ms, +50 ms] trajectory.
            if min(relative) > -130_000_000 or max(relative) < 30_000_000:
                return ContactVerification("Unknown", 0.0, "Incomplete Frame Window")

        trajectories: dict[str, list[tuple[float, tuple[float, float], float, float]]] = {}
        for snapshot in snapshots:
            relative_ms = (snapshot.timestamp_ns - key_timestamp_ns) / 1_000_000.0
            if relative_ms < -150.0 or relative_ms > 50.0:
                continue
            for hand in snapshot.hands:
                for finger, tip_index in TIP_INDICES.items():
                    label = f"{hand.handedness} {finger}"
                    point = hand.image_landmarks[tip_index, :2]
                    trajectories.setdefault(label, []).append(
                        (
                            relative_ms,
                            (float(point[0]), float(point[1])),
                            float(hand.local_landmarks[tip_index, 2]),
                            float(hand.downward_velocity[finger]),
                        )
                    )

        if not trajectories:
            return ContactVerification("Unknown", 0.0, "No Hand Landmarks in Window")

        candidates: list[ContactVerification] = []
        for label, samples in trajectories.items():
            samples.sort(key=lambda value: value[0])
            contact_samples = [sample for sample in samples if -70.0 <= sample[0] <= 50.0]
            if len(samples) < 2 or not contact_samples:
                continue

            distances = [
                keyboard_plane.distance_to_key_px(key_label, sample[1], frame_size)
                for sample in contact_samples
            ]
            distances = [distance for distance in distances if distance is not None]
            if not distances:
                continue
            distance_px, key_diagonal_px = min(distances, key=lambda value: value[0])
            proximity = math.exp(-distance_px / max(0.85 * key_diagonal_px, 1.0))

            early_z = self._median([sample[2] for sample in samples if -150.0 <= sample[0] <= -55.0])
            contact_z = self._median([sample[2] for sample in samples if -45.0 <= sample[0] <= 20.0])
            release_z = self._median([sample[2] for sample in samples if 20.0 <= sample[0] <= 50.0])
            displacement_down = (
                max(0.0, early_z - contact_z)
                if early_z is not None and contact_z is not None
                else 0.0
            )
            displacement_up = (
                max(0.0, release_z - contact_z)
                if release_z is not None and contact_z is not None
                else 0.0
            )
            peak_down_velocity = max(
                (sample[3] for sample in samples if -120.0 <= sample[0] <= 25.0),
                default=0.0,
            )
            stroke = max(
                min(displacement_down / 0.055, 1.0),
                min(peak_down_velocity / 1.8, 1.0),
            )
            reversal = math.sqrt(
                min(displacement_down / 0.045, 1.0)
                * min(displacement_up / 0.030, 1.0)
            )
            spatial_gate = 0.20 + 0.80 * proximity
            score = float(
                np.clip(
                    0.55 * proximity
                    + spatial_gate * (0.30 * stroke + 0.15 * reversal),
                    0.0,
                    1.0,
                )
            )
            candidates.append(
                ContactVerification(
                    label,
                    score,
                    "Accepted",
                    proximity,
                    stroke,
                    reversal,
                    distance_px,
                )
            )

        if not candidates:
            available_frames = max(
                (len(samples) for samples in trajectories.values()), default=0
            )
            return ContactVerification(
                "Unknown",
                0.0,
                f"Insufficient Trajectory: {available_frames} landmark frame(s)",
            )
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        best = candidates[0]
        runner_up = candidates[1].confidence if len(candidates) > 1 else 0.0
        if best.confidence < self.minimum_score:
            return ContactVerification(
                "Unknown",
                best.confidence,
                f"Weak Contact: {best.confidence:.0%} < {self.minimum_score:.0%}",
                best.proximity_score,
                best.stroke_score,
                best.reversal_score,
                best.distance_px,
            )
        if best.confidence < 0.65 and best.confidence - runner_up < self.ambiguity_margin:
            return ContactVerification(
                "Unknown",
                best.confidence,
                f"Ambiguous Contact: margin {(best.confidence - runner_up):.0%}",
                best.proximity_score,
                best.stroke_score,
                best.reversal_score,
                best.distance_px,
            )
        return best


@dataclass(slots=True)
class _TrackState:
    world_filter: OneEuroFilter
    image_filter: OneEuroFilter
    local_filter: OneEuroFilter
    timestamp_ns: int | None = None
    local_landmarks: np.ndarray | None = None
    image_landmarks: np.ndarray | None = None
    flexion: dict[str, tuple[float, float]] | None = None


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("degenerate palm geometry")
    return vector / norm


def _palm_local_frame(world_landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(world_landmarks, dtype=np.float64)
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    if palm_width < 1e-6:
        raise ValueError("palm width is too small")

    origin = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
    x_axis = _unit(points[5] - points[17])  # pinky side -> index side
    y_seed = points[9] - points[0]          # wrist -> middle MCP
    y_axis = _unit(y_seed - np.dot(y_seed, x_axis) * x_axis)
    z_axis = _unit(np.cross(x_axis, y_axis))

    relative = points - origin
    # Orient local -Z toward the fingertips/keyboard side for both hands.
    if float(np.mean(relative[list(TIP_INDICES.values())] @ z_axis)) > 0.0:
        z_axis = -z_axis
    basis = np.column_stack((x_axis, y_axis, z_axis))
    return (relative @ basis) / palm_width, basis


def palm_local_coordinates(world_landmarks: np.ndarray) -> np.ndarray:
    """Return landmarks in an anatomical palm basis, scaled by palm width."""

    local, _basis = _palm_local_frame(world_landmarks)
    return local


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = _unit(a - b)
    second = _unit(c - b)
    return math.acos(float(np.clip(np.dot(first, second), -1.0, 1.0)))


def _finger_flexion(points: np.ndarray, joints: tuple[int, int, int, int]) -> tuple[float, float]:
    mcp, pip, dip, tip = joints
    return (
        math.pi - _joint_angle(points[mcp], points[pip], points[dip]),
        math.pi - _joint_angle(points[pip], points[dip], points[tip]),
    )


class KinematicsEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tracks: dict[str, _TrackState] = {}
        self._history: deque[tuple[int, tuple[HandAnalysis, ...]]] = deque(maxlen=24)
        self._latest = KinematicsSnapshot(0, (), None)
        self._key_label: str | None = None
        self._key_timestamp_ns: int | None = None
        self._prediction: KeyPrediction | None = None
        self._prediction_quality = float("-inf")

    def register_keypress(self, timestamp_ns: int, label: str) -> None:
        with self._lock:
            self._key_label = label
            self._key_timestamp_ns = timestamp_ns
            self._prediction = None
            self._prediction_quality = float("-inf")
            for frame_timestamp_ns, hands in self._history:
                self._consider_prediction(frame_timestamp_ns, hands)
            self._latest = KinematicsSnapshot(
                self._latest.timestamp_ns, self._latest.hands, self._prediction
            )

    def _consider_prediction(
        self, frame_timestamp_ns: int, hands: tuple[HandAnalysis, ...]
    ) -> None:
        if self._key_timestamp_ns is None or self._key_label is None:
            return
        delta = abs(frame_timestamp_ns - self._key_timestamp_ns)
        if delta > KEY_MATCH_WINDOW_NS:
            return
        candidates = [
            (hand.active_score, hand.handedness, hand.active_finger)
            for hand in hands
            if hand.active_finger is not None
        ]
        score, handedness, finger = max(candidates, default=(0.0, None, None))
        temporal_weight = 1.0 - 0.35 * delta / KEY_MATCH_WINDOW_NS
        quality = score * temporal_weight
        if quality <= self._prediction_quality:
            return
        if score < 0.28:
            handedness, finger = None, None
        self._prediction_quality = quality
        self._prediction = KeyPrediction(
            self._key_label,
            self._key_timestamp_ns,
            frame_timestamp_ns,
            handedness,
            finger,
            score,
        )

    def process_result(
        self,
        result: vision.HandLandmarkerResult,
        timestamp_ns: int,
        image_roi: tuple[float, float, float, float] = FULL_FRAME_ROI,
    ) -> None:
        time_s = timestamp_ns / NS_PER_SECOND
        analyses: list[HandAnalysis] = []

        with self._lock:
            for index, image_hand in enumerate(result.hand_landmarks):
                if index >= len(result.hand_world_landmarks):
                    continue
                category = result.handedness[index][0] if result.handedness[index] else None
                handedness = category.category_name if category else f"Hand {index + 1}"
                handedness_score = float(category.score) if category else 0.0
                track = self._tracks.setdefault(
                    handedness,
                    _TrackState(OneEuroFilter(), OneEuroFilter(), OneEuroFilter()),
                )

                image_points = np.array(
                    [[point.x, point.y, point.z] for point in image_hand], dtype=np.float64
                )
                roi_x, roi_y, roi_width, roi_height = image_roi
                image_points[:, 0] = roi_x + image_points[:, 0] * roi_width
                image_points[:, 1] = roi_y + image_points[:, 1] * roi_height
                # MediaPipe landmark Z uses approximately the same normalized
                # scale as image X, so restore it to full-frame width units.
                image_points[:, 2] *= roi_width
                world_points = np.array(
                    [
                        [point.x, point.y, point.z]
                        for point in result.hand_world_landmarks[index]
                    ],
                    dtype=np.float64,
                )
                filtered_image = track.image_filter.filter(image_points, time_s)
                filtered_world = track.world_filter.filter(world_points, time_s)

                try:
                    # Canonicalize each result before temporal filtering. This
                    # removes MediaPipe world-scale changes caused by ROI crop
                    # context before coordinates and velocities enter features.
                    raw_local, _raw_basis = _palm_local_frame(world_points)
                    local = track.local_filter.filter(raw_local, time_s)
                    _posture_local, palm_basis = _palm_local_frame(filtered_world)
                    flexion = {
                        name: _finger_flexion(local, joints)
                        for name, joints in FINGER_JOINTS.items()
                    }
                except ValueError:
                    continue

                dt = 0.0 if track.timestamp_ns is None else (
                    timestamp_ns - track.timestamp_ns
                ) / NS_PER_SECOND
                tip_velocity_image: dict[str, np.ndarray] = {}
                tip_velocity_local: dict[str, np.ndarray] = {}
                downward_velocity: dict[str, float] = {}
                pip_flexion_rate: dict[str, float] = {}
                dip_flexion_rate: dict[str, float] = {}
                flexion_rate: dict[str, float] = {}

                for name, tip_index in TIP_INDICES.items():
                    if dt > 1e-4 and track.local_landmarks is not None and track.image_landmarks is not None:
                        local_velocity = (local[tip_index] - track.local_landmarks[tip_index]) / dt
                        image_velocity = (
                            filtered_image[tip_index, :2]
                            - track.image_landmarks[tip_index, :2]
                        ) / dt
                        previous_flexion = (track.flexion or {}).get(name, flexion[name])
                        pip_rate = (flexion[name][0] - previous_flexion[0]) / dt
                        dip_rate = (flexion[name][1] - previous_flexion[1]) / dt
                    else:
                        local_velocity = np.zeros(3)
                        image_velocity = np.zeros(2)
                        pip_rate = 0.0
                        dip_rate = 0.0
                    tip_velocity_image[name] = image_velocity
                    tip_velocity_local[name] = local_velocity
                    downward_velocity[name] = max(0.0, -float(local_velocity[2]))
                    pip_flexion_rate[name] = pip_rate
                    dip_flexion_rate[name] = dip_rate
                    flexion_rate[name] = max(0.0, pip_rate + dip_rate)

                depths = np.array([-local[index, 2] for index in TIP_INDICES.values()])
                depth_center = float(np.median(depths))
                depth_scale = max(float(np.ptp(depths)), 0.15)
                scores: dict[str, float] = {}
                for name in FINGER_NAMES:
                    proximity = float(
                        np.clip(0.5 + (depths[FINGER_NAMES.index(name)] - depth_center) / depth_scale, 0.0, 1.0)
                    )
                    velocity_term = min(downward_velocity[name] / 2.5, 1.0)
                    flexion_term = min(flexion_rate[name] / 8.0, 1.0)
                    scores[name] = 0.55 * velocity_term + 0.30 * flexion_term + 0.15 * proximity

                active_finger = max(scores, key=scores.get)
                active_score = scores[active_finger]
                if active_score < 0.28:
                    active_finger = None

                analysis = HandAnalysis(
                    handedness,
                    handedness_score,
                    timestamp_ns,
                    filtered_image,
                    local,
                    palm_basis[:, 2].copy(),
                    tip_velocity_image,
                    tip_velocity_local,
                    downward_velocity,
                    pip_flexion_rate,
                    dip_flexion_rate,
                    flexion_rate,
                    scores,
                    active_finger,
                    active_score,
                )
                analyses.append(analysis)
                track.timestamp_ns = timestamp_ns
                track.local_landmarks = local
                track.image_landmarks = filtered_image
                track.flexion = flexion

            hands_tuple = tuple(analyses)
            self._history.append((timestamp_ns, hands_tuple))
            while self._history and timestamp_ns - self._history[0][0] > 250_000_000:
                self._history.popleft()
            self._consider_prediction(timestamp_ns, hands_tuple)
            self._latest = KinematicsSnapshot(timestamp_ns, hands_tuple, self._prediction)

    def snapshot(self) -> KinematicsSnapshot:
        with self._lock:
            return self._latest

    def history_window(
        self,
        center_timestamp_ns: int,
        *,
        before_ns: int = 50_000_000,
        after_ns: int = 50_000_000,
    ) -> tuple[KinematicsSnapshot, ...]:
        """Return an immutable copy of inference results around a keydown."""

        start = center_timestamp_ns - before_ns
        end = center_timestamp_ns + after_ns
        with self._lock:
            return tuple(
                KinematicsSnapshot(timestamp_ns, hands, None)
                for timestamp_ns, hands in self._history
                if start <= timestamp_ns <= end
            )


class HandInferencePipeline:
    """Rate-limited MediaPipe Tasks LIVE_STREAM wrapper using the CPU delegate."""

    def __init__(self, model_path: Path, *, num_hands: int = 2, max_fps: float = 30.0):
        self.engine = KinematicsEngine()
        self.max_fps = max_fps
        self._lock = threading.Lock()
        self._clock_origin_ns: int | None = None
        self._last_submit_ns = 0
        self._last_timestamp_ms = -1
        self._submitted = 0
        self._completed = 0
        self._result_times: deque[int] = deque()
        self._latency_ms = 0.0
        self._error: str | None = None
        self._source_rois: dict[int, tuple[float, float, float, float]] = {}

        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_result,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def submit(
        self,
        frame_bgr: np.ndarray,
        frame_timestamp_ns: int,
        source_roi: tuple[float, float, float, float] = FULL_FRAME_ROI,
    ) -> bool:
        minimum_interval_ns = int(NS_PER_SECOND / self.max_fps)
        if frame_timestamp_ns - self._last_submit_ns < minimum_interval_ns:
            return False
        if self._clock_origin_ns is None:
            self._clock_origin_ns = frame_timestamp_ns
        timestamp_ms = (frame_timestamp_ns - self._clock_origin_ns) // 1_000_000
        if timestamp_ms <= self._last_timestamp_ms:
            return False

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        roi = normalized_roi(source_roi) or FULL_FRAME_ROI
        with self._lock:
            self._source_rois[int(timestamp_ms)] = roi
            if len(self._source_rois) > 32:
                oldest = min(self._source_rois)
                self._source_rois.pop(oldest, None)
        try:
            self._landmarker.detect_async(image, int(timestamp_ms))
        except BaseException:
            with self._lock:
                self._source_rois.pop(int(timestamp_ms), None)
            raise
        self._last_submit_ns = frame_timestamp_ns
        self._last_timestamp_ms = int(timestamp_ms)
        with self._lock:
            self._submitted += 1
        return True

    def _on_result(
        self,
        result: vision.HandLandmarkerResult,
        _image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        try:
            if self._clock_origin_ns is None:
                return
            frame_timestamp_ns = self._clock_origin_ns + timestamp_ms * 1_000_000
            with self._lock:
                source_roi = self._source_rois.pop(timestamp_ms, FULL_FRAME_ROI)
            self.engine.process_result(result, frame_timestamp_ns, source_roi)
            now_ns = time.perf_counter_ns()
            with self._lock:
                self._completed += 1
                self._latency_ms = (now_ns - frame_timestamp_ns) / 1_000_000
                self._result_times.append(now_ns)
                cutoff = now_ns - NS_PER_SECOND
                while len(self._result_times) > 1 and self._result_times[0] < cutoff:
                    self._result_times.popleft()
        except BaseException as exc:
            with self._lock:
                self._error = f"MediaPipe callback failed: {exc}"

    def stats(self) -> tuple[float, float, int, str | None]:
        with self._lock:
            if len(self._result_times) < 2:
                result_fps = 0.0
            else:
                elapsed = self._result_times[-1] - self._result_times[0]
                result_fps = (len(self._result_times) - 1) * NS_PER_SECOND / elapsed
            dropped = max(0, self._submitted - self._completed - 1)
            return result_fps, self._latency_ms, dropped, self._error

    def close(self) -> None:
        self._landmarker.close()


def draw_hand_overlay(
    frame: np.ndarray,
    snapshot: KinematicsSnapshot,
    view_roi: tuple[float, float, float, float] = FULL_FRAME_ROI,
) -> None:
    height, width = frame.shape[:2]
    prediction = snapshot.key_prediction
    roi_x, roi_y, roi_width, roi_height = view_roi

    for hand in snapshot.hands:
        view_points = hand.image_landmarks.copy()
        view_points[:, 0] = (view_points[:, 0] - roi_x) / roi_width
        view_points[:, 1] = (view_points[:, 1] - roi_y) / roi_height
        points = np.column_stack(
            (view_points[:, 0] * width, view_points[:, 1] * height)
        ).astype(np.int32)
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, tuple(points[start]), tuple(points[end]), (190, 190, 190), 1, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, tuple(point), 2, (230, 230, 230), -1, cv2.LINE_AA)

        highlighted = hand.active_finger
        if prediction and prediction.handedness == hand.handedness and prediction.finger:
            highlighted = prediction.finger

        for name, tip_index in TIP_INDICES.items():
            start = points[tip_index]
            velocity = hand.tip_velocity_image[name]
            vector = np.array(
                [velocity[0] * width / roi_width, velocity[1] * height / roi_height]
            ) * 0.06
            magnitude = float(np.linalg.norm(vector))
            if magnitude > 70.0:
                vector *= 70.0 / magnitude
            end = np.rint(start + vector).astype(np.int32)
            color = FINGER_COLORS[name]
            cv2.arrowedLine(frame, tuple(start), tuple(end), color, 2, cv2.LINE_AA, tipLength=0.25)
            cv2.circle(frame, tuple(start), 8 if name == highlighted else 5, color, 2, cv2.LINE_AA)

        wrist = points[0]
        active_text = "steady"
        if hand.active_finger:
            active_text = f"{hand.active_finger} {hand.active_score:.2f}"
        cv2.putText(
            frame,
            f"{hand.handedness}: {active_text}",
            (int(wrist[0]) + 8, max(22, int(wrist[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 255, 120),
            2,
            cv2.LINE_AA,
        )

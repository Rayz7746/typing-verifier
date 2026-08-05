"""Guided, timestamp-aligned calibration dataset collection."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from keyboard_layouts import US_ANSI_QWERTY, expected_finger, normalize_key_label
from kinematics import FINGER_NAMES, TIP_INDICES, KinematicsEngine, KinematicsSnapshot


SCHEMA_VERSION = 2
WINDOW_OFFSETS_MS = (-50, -25, 0, 25, 50)
HAND_ORDER = ("Left", "Right")
FEATURES_PER_FINGER = 9
FEATURE_SCALES = np.asarray((4.0, 4.0, 4.0, 20.0, 20.0, 20.0, 50.0, 50.0, 1.0))
DEFAULT_SEQUENCE = "asdf jkl;"
DEFAULT_DATA_PATH = Path(__file__).resolve().with_name("calibration_data.json")
CALIBRATION_SETS = {
    "Home Row": "asdf jkl;",
    "Top Row": "qwertyuiop",
    "Bottom Row": "zxcvbnm",
    "Numbers": "1234567890",
    "Full Keyboard": "qwertyuiopasdfghjkl;zxcvbnm 1234567890",
}


def feature_names() -> list[str]:
    names: list[str] = []
    for offset in WINDOW_OFFSETS_MS:
        for hand in HAND_ORDER:
            for finger in FINGER_NAMES:
                prefix = f"t{offset:+d}.{hand.lower()}.{finger}"
                names.extend(
                    [
                        f"{prefix}.x",
                        f"{prefix}.y",
                        f"{prefix}.z",
                        f"{prefix}.vx",
                        f"{prefix}.vy",
                        f"{prefix}.vz",
                        f"{prefix}.pip_rate",
                        f"{prefix}.dip_rate",
                        f"{prefix}.present",
                    ]
                )
    return names


FEATURE_NAMES = feature_names()


def normalize_feature_vector(values: list[float] | np.ndarray) -> np.ndarray:
    """Bound canonical kinematics to stable ranges shared by train and predict."""

    matrix = np.asarray(values, dtype=np.float32).reshape(-1, FEATURES_PER_FINGER)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix[:, :8] = np.clip(matrix[:, :8] / FEATURE_SCALES[:8], -1.0, 1.0)
    matrix[:, 8] = np.clip(matrix[:, 8], 0.0, 1.0)
    return matrix.reshape(-1)


@dataclass(frozen=True, slots=True)
class FeatureWindow:
    center_timestamp_ns: int
    values: np.ndarray
    fingertip_positions: dict[str, tuple[float, float]]
    finger_scores: dict[str, float]
    heuristic_label: str
    heuristic_confidence: float


@dataclass(frozen=True, slots=True)
class CalibrationProgress:
    index: int
    total: int
    current_key: str
    accepted: int
    rejected: int
    waiting_for_window: bool
    complete: bool
    message: str
    round_index: int = 0
    rounds: int = 0
    target_sample_count: int = 0
    dataset_samples: int = 0
    set_name: str = ""


@dataclass(slots=True)
class _PendingPress:
    timestamp_ns: int
    key: str


def _hand_lookup(snapshot: KinematicsSnapshot) -> dict[str, object]:
    return {hand.handedness.title(): hand for hand in snapshot.hands}


def extract_feature_window(
    engine: KinematicsEngine,
    center_timestamp_ns: int,
) -> FeatureWindow | None:
    """Resample the 100 ms kinematic window into a fixed 450-value vector."""

    snapshots = engine.history_window(center_timestamp_ns)
    if not snapshots or not any(snapshot.hands for snapshot in snapshots):
        return None

    values: list[float] = []
    selected: list[KinematicsSnapshot] = []
    for offset_ms in WINDOW_OFFSETS_MS:
        target_ns = center_timestamp_ns + offset_ms * 1_000_000
        snapshot = min(snapshots, key=lambda item: abs(item.timestamp_ns - target_ns))
        selected.append(snapshot)
        hands = _hand_lookup(snapshot)
        for handedness in HAND_ORDER:
            hand = hands.get(handedness)
            for finger in FINGER_NAMES:
                if hand is None:
                    values.extend((0.0,) * FEATURES_PER_FINGER)
                    continue
                tip_index = TIP_INDICES[finger]
                local_tip = hand.local_landmarks[tip_index]
                velocity = hand.tip_velocity_local[finger]
                values.extend(
                    (
                        float(local_tip[0]),
                        float(local_tip[1]),
                        float(local_tip[2]),
                        float(velocity[0]),
                        float(velocity[1]),
                        float(velocity[2]),
                        float(hand.pip_flexion_rate[finger]),
                        float(hand.dip_flexion_rate[finger]),
                        1.0,
                    )
                )

    fingertip_positions: dict[str, tuple[float, float]] = {}
    finger_scores: dict[str, float] = {}
    for snapshot in sorted(selected, key=lambda item: abs(item.timestamp_ns - center_timestamp_ns)):
        for hand in snapshot.hands:
            canonical_hand = hand.handedness.title()
            if canonical_hand not in HAND_ORDER:
                continue
            for finger in FINGER_NAMES:
                label = f"{canonical_hand} {finger}"
                tip = hand.image_landmarks[TIP_INDICES[finger]]
                fingertip_positions.setdefault(label, (float(tip[0]), float(tip[1])))
                finger_scores[label] = max(
                    finger_scores.get(label, 0.0), float(hand.finger_scores[finger])
                )

    score, label = max(
        ((score, label) for label, score in finger_scores.items()),
        default=(0.0, "Unknown"),
    )
    if score < 0.28:
        label = "Unknown"
    vector = normalize_feature_vector(values)
    if vector.size != len(FEATURE_NAMES):
        raise RuntimeError(f"feature schema mismatch: {vector.size} != {len(FEATURE_NAMES)}")
    return FeatureWindow(
        center_timestamp_ns,
        vector,
        fingertip_positions,
        finger_scores,
        label,
        score,
    )


class GuidedCalibrationCollector:
    """Accept only the prompted key and finalize it after the +50 ms window exists."""

    def __init__(
        self,
        output_path: Path = DEFAULT_DATA_PATH,
        *,
        keyboard_layout: str = US_ANSI_QWERTY,
    ) -> None:
        self.output_path = Path(output_path)
        self.keyboard_layout = keyboard_layout
        self._prompts: list[str] = []
        self._sequence = ""
        self._rounds = 0
        self._set_name = "Custom"
        self._index = 0
        self._rejected = 0
        self._pending: _PendingPress | None = None
        self._samples: list[dict] = []
        self._existing_samples: list[dict] = []
        self._existing_sessions: list[dict] = []
        self._sample_counts: dict[str, int] = {}
        self._started_wall_ns = 0
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def start(
        self,
        sequence: str = DEFAULT_SEQUENCE,
        repeats: int = 5,
        set_name: str = "Custom",
    ) -> CalibrationProgress:
        if not sequence or repeats < 1:
            raise ValueError("sequence must be non-empty and repeats must be positive")
        self._sequence = sequence
        self._rounds = repeats
        self._set_name = set_name
        self._prompts = list(sequence) * repeats
        self._index = self._rejected = 0
        self._pending = None
        self._samples = []
        self._sample_counts = {}
        self._existing_samples = []
        self._existing_sessions = []
        if self.output_path.exists():
            try:
                existing = json.loads(self.output_path.read_text(encoding="utf-8"))
                if (
                    existing.get("schema_version") == SCHEMA_VERSION
                    and existing.get("feature_names") == FEATURE_NAMES
                ):
                    self._existing_samples = list(existing.get("samples", []))
                    self._existing_sessions = list(existing.get("sessions", []))
            except (OSError, ValueError, TypeError):
                pass
        self._started_wall_ns = time.time_ns()
        self._active = True
        return self.progress("Type the highlighted key")

    def cancel(self) -> CalibrationProgress:
        self._active = False
        self._pending = None
        return self.progress("Calibration cancelled")

    def progress(self, message: str = "") -> CalibrationProgress:
        current = "" if self._index >= len(self._prompts) else self._prompts[self._index]
        round_index = (
            0
            if not self._sequence or self._index >= len(self._prompts)
            else self._index // len(self._sequence) + 1
        )
        normalized_current = "space" if current == " " else current.lower()
        return CalibrationProgress(
            self._index,
            len(self._prompts),
            current,
            len(self._samples),
            self._rejected,
            self._pending is not None,
            bool(self._prompts) and self._index >= len(self._prompts),
            message,
            round_index,
            self._rounds,
            self._sample_counts.get(normalized_current, 0),
            len(self._existing_samples) + len(self._samples),
            self._set_name,
        )

    def handle_keypress(self, timestamp_ns: int, label: str) -> CalibrationProgress | None:
        if not self._active:
            return None
        if self._pending is not None:
            return self.progress("Hold on while the 100 ms window is finalized")

        expected_key = self._prompts[self._index]
        expected_label = "space" if expected_key == " " else expected_key.lower()
        actual_label = normalize_key_label(label)
        if actual_label != expected_label:
            self._rejected += 1
            return self.progress(f"Ignored {actual_label!r}; expected {expected_label!r}")

        self._pending = _PendingPress(timestamp_ns, expected_label)
        return self.progress("Matched; collecting +50 ms of motion")

    def process_ready(
        self,
        engine: KinematicsEngine,
        now_timestamp_ns: int,
    ) -> CalibrationProgress | None:
        if not self._active or self._pending is None:
            return None
        if now_timestamp_ns < self._pending.timestamp_ns + 55_000_000:
            return None
        if engine.snapshot().timestamp_ns < self._pending.timestamp_ns + 50_000_000:
            return None

        pending = self._pending
        self._pending = None
        window = extract_feature_window(engine, pending.timestamp_ns)
        if window is None:
            return self.progress("No hand was visible; please retry the same key")

        target_label = expected_finger(pending.key, self.keyboard_layout)
        if target_label == "Thumb":
            target_label = max(
                ("Left thumb", "Right thumb"),
                key=lambda label: window.finger_scores.get(label, 0.0),
            )

        target_point = window.fingertip_positions.get(target_label)
        if target_point is None:
            return self.progress(f"{target_label} was not visible; please retry the same key")

        self._samples.append(
            {
                "key": pending.key,
                "label": target_label,
                "keydown_perf_counter_ns": pending.timestamp_ns,
                "features": window.values.tolist(),
                "target_fingertip_xy": list(target_point),
            }
        )
        self._sample_counts[pending.key] = self._sample_counts.get(pending.key, 0) + 1
        self._index += 1
        if self._index >= len(self._prompts):
            self._active = False
            self.save()
            return self.progress(f"Saved {len(self._samples)} samples to {self.output_path.name}")
        return self.progress("Sample accepted")

    def save(self) -> None:
        session = {
            "set_name": self._set_name,
            "sequence": self._sequence,
            "rounds": self._rounds,
            "accepted_samples": len(self._samples),
            "completed_unix_ns": time.time_ns(),
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_unix_ns": self._started_wall_ns,
            "keyboard_layout": self.keyboard_layout,
            "window_offsets_ms": list(WINDOW_OFFSETS_MS),
            "coordinate_space": "palm_local_normalized_v2",
            "feature_scaling": {
                "position_palm_widths": 4.0,
                "velocity_palm_widths_per_second": 20.0,
                "flexion_radians_per_second": 50.0,
            },
            "feature_names": FEATURE_NAMES,
            "sessions": [*self._existing_sessions, session],
            "samples": [*self._existing_samples, *self._samples],
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(self.output_path.suffix + ".part")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.output_path)

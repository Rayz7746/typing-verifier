"""Hysteretic posture alerts based on calibrated palm orientation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from kinematics import KinematicsSnapshot


@dataclass(frozen=True, slots=True)
class PostureStatus:
    severity: str
    message: str


@dataclass(slots=True)
class _HandState:
    baseline: np.ndarray | None = None
    calibration_started_ns: int | None = None
    warning_started_ns: int | None = None
    clear_started_ns: int | None = None
    warning: bool = False
    angle_degrees: float = 0.0
    last_seen_ns: int = 0


class PostureMonitor:
    """Palm-angle proxy; Hand Landmarker does not observe the forearm."""

    def __init__(
        self,
        *,
        calibration_seconds: float = 1.5,
        warning_angle: float = 25.0,
        clear_angle: float = 15.0,
        warning_hold_seconds: float = 0.6,
        clear_hold_seconds: float = 0.8,
    ) -> None:
        self.calibration_ns = int(calibration_seconds * 1_000_000_000)
        self.warning_angle = warning_angle
        self.clear_angle = clear_angle
        self.warning_hold_ns = int(warning_hold_seconds * 1_000_000_000)
        self.clear_hold_ns = int(clear_hold_seconds * 1_000_000_000)
        self._hands: dict[str, _HandState] = {}
        self._status = PostureStatus("info", "Place hands in a neutral typing posture")

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / max(norm, 1e-8)

    def update(self, snapshot: KinematicsSnapshot, now_ns: int) -> PostureStatus:
        if not snapshot.hands:
            self._status = PostureStatus("info", "Hands not visible — posture monitoring paused")
            return self._status

        calibrating = False
        for hand in snapshot.hands:
            state = self._hands.setdefault(hand.handedness, _HandState())
            state.last_seen_ns = now_ns
            normal = self._unit(hand.palm_normal)
            if state.baseline is None:
                state.baseline = normal.copy()
                state.calibration_started_ns = now_ns
                calibrating = True
                continue

            calibration_started_ns = (
                state.calibration_started_ns
                if state.calibration_started_ns is not None
                else now_ns
            )
            if now_ns - calibration_started_ns < self.calibration_ns:
                state.baseline = self._unit(0.92 * state.baseline + 0.08 * normal)
                calibrating = True
                continue

            if float(np.dot(state.baseline, normal)) < 0.0:
                normal = -normal
            cosine = float(np.clip(np.dot(state.baseline, normal), -1.0, 1.0))
            state.angle_degrees = math.degrees(math.acos(cosine))

            if state.warning:
                state.warning_started_ns = None
                if state.angle_degrees <= self.clear_angle:
                    state.clear_started_ns = state.clear_started_ns or now_ns
                    if now_ns - state.clear_started_ns >= self.clear_hold_ns:
                        state.warning = False
                        state.clear_started_ns = None
                else:
                    state.clear_started_ns = None
            else:
                state.clear_started_ns = None
                if state.angle_degrees >= self.warning_angle:
                    state.warning_started_ns = state.warning_started_ns or now_ns
                    if now_ns - state.warning_started_ns >= self.warning_hold_ns:
                        state.warning = True
                else:
                    state.warning_started_ns = None
                    if state.angle_degrees < 8.0:
                        state.baseline = self._unit(0.995 * state.baseline + 0.005 * normal)

        warnings = [
            f"{name} {state.angle_degrees:.0f}°"
            for name, state in self._hands.items()
            if state.warning and now_ns - state.last_seen_ns < 500_000_000
        ]
        if warnings:
            self._status = PostureStatus(
                "warning",
                "Wrist/palm angle warning: " + ", ".join(warnings),
            )
        elif calibrating:
            self._status = PostureStatus("info", "Calibrating neutral wrist/palm angle…")
        else:
            angles = ", ".join(
                f"{name} {state.angle_degrees:.0f}°" for name, state in self._hands.items()
            )
            self._status = PostureStatus("ok", f"Posture stable — {angles}")
        return self._status

    @property
    def status(self) -> PostureStatus:
        return self._status

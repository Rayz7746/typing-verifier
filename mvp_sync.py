"""Real-time keyboard/camera synchronization MVP for Windows.

The keyboard hook and camera acquisition run outside the OpenCV UI loop. All
timestamps use time.perf_counter_ns(), so event/frame deltas share one clock.
"""

from __future__ import annotations

import argparse
import itertools
import queue
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Final

import cv2
from pynput import keyboard


WINDOW_NAME: Final = "Touch-Typing Verifier - Synchronization MVP"
KEY_QUEUE_SIZE: Final = 256
NS_PER_SECOND: Final = 1_000_000_000
NS_PER_MILLISECOND: Final = 1_000_000


@dataclass(frozen=True, slots=True)
class KeyEvent:
    sequence: int
    timestamp_ns: int
    action: str
    label: str


@dataclass(frozen=True, slots=True)
class FramePacket:
    sequence: int
    timestamp_ns: int
    image: object


class LatestFrameSlot:
    """A single-frame mailbox; publishing replaces any unrendered frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: FramePacket | None = None
        self._last_rendered_sequence = -1
        self._stale_drops = 0

    def publish(self, packet: FramePacket) -> None:
        with self._lock:
            if (
                self._latest is not None
                and self._latest.sequence != self._last_rendered_sequence
            ):
                self._stale_drops += 1
            self._latest = packet

    def peek(self) -> FramePacket | None:
        with self._lock:
            return self._latest

    def consume_for_render(self) -> tuple[FramePacket | None, int]:
        with self._lock:
            packet = self._latest
            if packet is not None:
                self._last_rendered_sequence = packet.sequence
            return packet, self._stale_drops


class RollingRate:
    """One-second rolling event rate."""

    def __init__(self) -> None:
        self._timestamps: deque[int] = deque()

    def tick(self, timestamp_ns: int) -> None:
        self._timestamps.append(timestamp_ns)
        cutoff = timestamp_ns - NS_PER_SECOND
        while len(self._timestamps) > 1 and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def value(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed_ns = self._timestamps[-1] - self._timestamps[0]
        if elapsed_ns <= 0:
            return 0.0
        return (len(self._timestamps) - 1) * NS_PER_SECOND / elapsed_ns


class CaptureWorker(threading.Thread):
    """Continuously drains a Windows camera into a LatestFrameSlot."""

    BACKENDS: Final = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
    }

    def __init__(
        self,
        *,
        camera_idx: int,
        backend: str,
        width: int,
        height: int,
        fps: float,
        slot: LatestFrameSlot,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="camera-capture", daemon=True)
        self.camera_idx = camera_idx
        self.backend = backend
        self.width = width
        self.height = height
        self.fps = fps
        self.slot = slot
        self.stop_event = stop_event
        self.ready = threading.Event()

        self._state_lock = threading.Lock()
        self._selected_backend = "not-opened"
        self._capture_fps = 0.0
        self._read_failures = 0
        self._error: str | None = None

    def _backend_candidates(self) -> list[tuple[str, int]]:
        if self.backend == "auto":
            return [(name, self.BACKENDS[name]) for name in ("dshow", "msmf")]
        return [(self.backend, self.BACKENDS[self.backend])]

    def _open_camera(self) -> tuple[cv2.VideoCapture | None, FramePacket | None]:
        failures: list[str] = []
        sequence = 0

        for name, backend_id in self._backend_candidates():
            if self.stop_event.is_set():
                break

            print(f"Opening camera {self.camera_idx} with {name.upper()}...")
            capture = cv2.VideoCapture(self.camera_idx, backend_id)
            if not capture.isOpened():
                failures.append(f"{name.upper()}: open failed")
                capture.release()
                continue

            # Drivers may ignore individual properties; accepted values are
            # reflected by the actual frames and OSD rather than assumed here.
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            read_started_ns = time.perf_counter_ns()
            ok, frame = capture.read()
            read_finished_ns = time.perf_counter_ns()
            if not ok or frame is None:
                failures.append(f"{name.upper()}: opened but first read failed")
                capture.release()
                continue

            timestamp_ns = (read_started_ns + read_finished_ns) // 2
            packet = FramePacket(sequence, timestamp_ns, frame)
            with self._state_lock:
                self._selected_backend = name.upper()
            print(
                f"Using {name.upper()}: {frame.shape[1]}x{frame.shape[0]} "
                f"(requested {self.width}x{self.height} @ {self.fps:g} FPS)"
            )
            return capture, packet

        detail = "; ".join(failures) if failures else "capture cancelled"
        with self._state_lock:
            self._error = f"Unable to acquire camera {self.camera_idx}: {detail}"
        return None, None

    def run(self) -> None:
        capture: cv2.VideoCapture | None = None
        rate = RollingRate()
        sequence = 0

        try:
            capture, first_packet = self._open_camera()
            if capture is None or first_packet is None:
                return

            self.slot.publish(first_packet)
            rate.tick(first_packet.timestamp_ns)
            sequence = first_packet.sequence + 1
            self.ready.set()

            consecutive_failures = 0
            while not self.stop_event.is_set():
                read_started_ns = time.perf_counter_ns()
                ok, frame = capture.read()
                read_finished_ns = time.perf_counter_ns()

                if not ok or frame is None:
                    consecutive_failures += 1
                    with self._state_lock:
                        self._read_failures += 1
                    if consecutive_failures >= 30:
                        with self._state_lock:
                            self._error = "Camera stopped returning frames."
                        self.stop_event.set()
                        break
                    time.sleep(0.005)
                    continue

                consecutive_failures = 0
                timestamp_ns = (read_started_ns + read_finished_ns) // 2
                self.slot.publish(FramePacket(sequence, timestamp_ns, frame))
                sequence += 1
                rate.tick(timestamp_ns)
                with self._state_lock:
                    self._capture_fps = rate.value()
        except BaseException as exc:
            with self._state_lock:
                self._error = f"Capture thread failed: {exc}"
            self.stop_event.set()
        finally:
            if capture is not None:
                capture.release()
            self.ready.set()

    def snapshot(self) -> tuple[str, float, int, str | None]:
        with self._state_lock:
            return (
                self._selected_backend,
                self._capture_fps,
                self._read_failures,
                self._error,
            )


class KeyboardCollector:
    """Non-blocking keyboard callbacks backed by a bounded queue."""

    def __init__(self, stop_event: threading.Event) -> None:
        self.events: queue.Queue[KeyEvent] = queue.Queue(maxsize=KEY_QUEUE_SIZE)
        self.stop_event = stop_event
        self.listener: keyboard.Listener | None = None
        self._sequence = itertools.count()
        self._dropped = 0
        self._dropped_lock = threading.Lock()

    @staticmethod
    def _label(key: keyboard.Key | keyboard.KeyCode | None) -> str:
        if key is None:
            return "unknown"
        if isinstance(key, keyboard.KeyCode):
            if key.char is not None:
                return repr(key.char)
            if key.vk is not None:
                return f"vk:{key.vk}"
        text = str(key)
        return text.removeprefix("Key.")

    def _publish(self, action: str, key: keyboard.Key | keyboard.KeyCode | None) -> bool | None:
        event = KeyEvent(
            sequence=next(self._sequence),
            timestamp_ns=time.perf_counter_ns(),
            action=action,
            label=self._label(key),
        )
        try:
            self.events.put_nowait(event)
        except queue.Full:
            with self._dropped_lock:
                self._dropped += 1

        if action == "press" and key == keyboard.Key.esc:
            self.stop_event.set()
            return False
        return None

    def start(self) -> None:
        self.listener = keyboard.Listener(
            on_press=lambda key: self._publish("press", key),
            on_release=lambda key: self._publish("release", key),
        )
        self.listener.start()
        self.listener.wait()

    def stop(self) -> None:
        if self.listener is None:
            return
        self.listener.stop()
        try:
            self.listener.join(timeout=1.0)
        except BaseException as exc:
            print(f"Keyboard listener error: {exc}", file=sys.stderr)

    @property
    def dropped(self) -> int:
        with self._dropped_lock:
            return self._dropped


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Timestamp and correlate global keyboard events with camera frames."
    )
    parser.add_argument("--camera-idx", type=non_negative_int, default=0)
    parser.add_argument(
        "--backend",
        choices=("auto", "dshow", "msmf"),
        default="auto",
        help="auto tries DSHOW first, then MSMF (default: auto)",
    )
    parser.add_argument("--width", type=positive_int, default=1280)
    parser.add_argument("--height", type=positive_int, default=720)
    parser.add_argument("--fps", type=positive_float, default=60.0)
    return parser.parse_args(argv)


def draw_osd(frame: object, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.58
    thickness = 1
    line_height = 25
    padding = 10
    panel_height = padding * 2 + line_height * len(lines)
    panel_width = min(frame.shape[1], 650)

    panel_height = min(panel_height, frame.shape[0])
    panel = frame[:panel_height, :panel_width]
    black = panel.copy()
    black.fill(0)
    cv2.addWeighted(black, 0.62, panel, 0.38, 0.0, panel)
    for index, line in enumerate(lines):
        y = padding + 18 + index * line_height
        cv2.putText(
            frame,
            line,
            (padding, y),
            font,
            font_scale,
            (80, 255, 120),
            thickness,
            cv2.LINE_AA,
        )


def run(args: argparse.Namespace) -> int:
    stop_event = threading.Event()
    frame_slot = LatestFrameSlot()
    capture = CaptureWorker(
        camera_idx=args.camera_idx,
        backend=args.backend,
        width=args.width,
        height=args.height,
        fps=args.fps,
        slot=frame_slot,
        stop_event=stop_event,
    )
    keyboard_collector = KeyboardCollector(stop_event)
    render_rate = RollingRate()

    latest_keypress = "none"
    latest_delta_ms: float | None = None
    last_displayed_sequence = -1
    window_created = False

    capture.start()
    if not capture.ready.wait(timeout=10.0):
        stop_event.set()
        capture.join(timeout=3.0)
        print("Timed out while opening the camera.", file=sys.stderr)
        return 1

    _, _, _, capture_error = capture.snapshot()
    if capture_error is not None:
        stop_event.set()
        capture.join(timeout=3.0)
        print(capture_error, file=sys.stderr)
        return 1

    exit_code = 0
    try:
        keyboard_collector.start()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        window_created = True

        while not stop_event.is_set():
            while True:
                try:
                    event = keyboard_collector.events.get_nowait()
                except queue.Empty:
                    break

                if event.action == "press":
                    latest_keypress = event.label
                    associated_frame = frame_slot.peek()
                    if associated_frame is not None:
                        latest_delta_ms = (
                            event.timestamp_ns - associated_frame.timestamp_ns
                        ) / NS_PER_MILLISECOND

            packet, stale_drops = frame_slot.consume_for_render()
            if packet is None:
                time.sleep(0.001)
                continue

            if packet.sequence != last_displayed_sequence:
                display = packet.image.copy()
                now_ns = time.perf_counter_ns()
                render_rate.tick(now_ns)
                backend, capture_fps, read_failures, capture_error = capture.snapshot()

                height, width = display.shape[:2]
                delta_text = (
                    "n/a"
                    if latest_delta_ms is None
                    else f"{latest_delta_ms:+.3f} ms (key - frame)"
                )
                draw_osd(
                    display,
                    [
                        f"Resolution: {width}x{height} | Backend: {backend}",
                        f"Capture FPS: {capture_fps:5.1f} | Render FPS: {render_rate.value():5.1f}",
                        f"Latest keypress: {latest_keypress}",
                        f"Event-to-frame delta: {delta_text}",
                        (
                            "Dropped: "
                            f"stale frames={stale_drops} | read failures={read_failures} "
                            f"| key events={keyboard_collector.dropped}"
                        ),
                        "Press Escape or close this window to exit.",
                    ],
                )
                cv2.imshow(WINDOW_NAME, display)
                last_displayed_sequence = packet.sequence

                if capture_error is not None:
                    print(capture_error, file=sys.stderr)
                    stop_event.set()

            pressed = cv2.waitKey(1) & 0xFF
            if pressed == 27:
                stop_event.set()
                break

            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    stop_event.set()
                    break
            except cv2.error:
                stop_event.set()
                break

            if (
                keyboard_collector.listener is not None
                and not keyboard_collector.listener.is_alive()
                and not stop_event.is_set()
            ):
                print("Keyboard listener stopped unexpectedly.", file=sys.stderr)
                exit_code = 1
                stop_event.set()
                break

        _, _, _, final_capture_error = capture.snapshot()
        if final_capture_error is not None:
            print(final_capture_error, file=sys.stderr)
            exit_code = 1
        return exit_code
    except KeyboardInterrupt:
        return 0
    finally:
        stop_event.set()
        keyboard_collector.stop()
        capture.join(timeout=3.0)
        if window_created:
            cv2.destroyAllWindows()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

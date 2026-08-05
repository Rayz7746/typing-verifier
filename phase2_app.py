"""Phase 2 live preview: synchronized keys plus MediaPipe finger kinematics."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import cv2

from fetch_model import DEFAULT_MODEL_PATH, ensure_model
from kinematics import HandInferencePipeline, draw_hand_overlay
from mvp_sync import (
    NS_PER_MILLISECOND,
    CaptureWorker,
    KeyboardCollector,
    LatestFrameSlot,
    RollingRate,
    draw_osd,
    non_negative_int,
    positive_float,
    positive_int,
)


WINDOW_NAME = "Touch-Typing Verifier - MediaPipe Kinematics"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-idx", type=non_negative_int, default=0)
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="auto")
    parser.add_argument("--width", type=positive_int, default=1920)
    parser.add_argument("--height", type=positive_int, default=1080)
    parser.add_argument("--fps", type=positive_float, default=60.0)
    parser.add_argument(
        "--processing-height",
        type=positive_int,
        default=720,
        help="maximum height used for inference and display (default: 720)",
    )
    parser.add_argument(
        "--inference-fps",
        type=positive_float,
        default=30.0,
        help="maximum MediaPipe submissions per second (default: 30)",
    )
    parser.add_argument("--num-hands", type=positive_int, choices=(1, 2), default=2)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args(argv)


def resize_for_processing(frame, maximum_height: int):
    height, width = frame.shape[:2]
    if height <= maximum_height:
        return frame.copy()
    scale = maximum_height / height
    size = (max(1, round(width * scale)), maximum_height)
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def prediction_text(snapshot) -> str:
    prediction = snapshot.key_prediction
    if prediction is None:
        return "Observed finger: waiting for keypress"
    if prediction.finger is None:
        return f"Observed finger for {prediction.key_label}: unknown ({prediction.score:.2f})"
    return (
        f"Observed finger for {prediction.key_label}: "
        f"{prediction.handedness} {prediction.finger} "
        f"({prediction.score:.2f}, dt={prediction.delta_ms:+.1f} ms)"
    )


def run(args: argparse.Namespace) -> int:
    try:
        model_path = ensure_model(args.model)
    except BaseException as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        return 1

    cv2.setUseOptimized(True)
    cv2.setNumThreads(1)
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
    inference: HandInferencePipeline | None = None
    window_created = False
    latest_keypress = "none"
    latest_delta_ms: float | None = None
    last_displayed_sequence = -1
    exit_code = 0

    capture.start()
    if not capture.ready.wait(timeout=10.0):
        stop_event.set()
        capture.join(timeout=3.0)
        print("Timed out while opening the camera.", file=sys.stderr)
        return 1
    _, _, _, capture_error = capture.snapshot()
    if capture_error:
        stop_event.set()
        capture.join(timeout=3.0)
        print(capture_error, file=sys.stderr)
        return 1

    try:
        inference = HandInferencePipeline(
            model_path,
            num_hands=args.num_hands,
            max_fps=args.inference_fps,
        )
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
                    inference.engine.register_keypress(event.timestamp_ns, event.label)

            packet, stale_drops = frame_slot.consume_for_render()
            if packet is None or packet.sequence == last_displayed_sequence:
                cv2.waitKey(1)
                continue

            display = resize_for_processing(packet.image, args.processing_height)
            inference.submit(display, packet.timestamp_ns)
            snapshot = inference.engine.snapshot()
            draw_hand_overlay(display, snapshot)
            now_ns = time.perf_counter_ns()
            render_rate.tick(now_ns)

            backend, capture_fps, read_failures, capture_error = capture.snapshot()
            inference_fps, inference_latency_ms, inference_drops, inference_error = inference.stats()
            source_height, source_width = packet.image.shape[:2]
            render_height, render_width = display.shape[:2]
            delta_text = "n/a" if latest_delta_ms is None else f"{latest_delta_ms:+.3f} ms"
            analysis_age_ms = (
                0.0 if snapshot.timestamp_ns == 0 else (now_ns - snapshot.timestamp_ns) / 1_000_000
            )
            draw_osd(
                display,
                [
                    f"Capture: {source_width}x{source_height} {backend} | Process: {render_width}x{render_height}",
                    f"Capture/Render FPS: {capture_fps:5.1f} / {render_rate.value():5.1f}",
                    f"MediaPipe FPS: {inference_fps:5.1f} | latency/age: {inference_latency_ms:5.1f}/{analysis_age_ms:5.1f} ms",
                    f"Latest keypress: {latest_keypress} | key-frame delta: {delta_text}",
                    prediction_text(snapshot),
                    (
                        f"Dropped: stale={stale_drops} | reads={read_failures} "
                        f"| MP={inference_drops} | keys={keyboard_collector.dropped}"
                    ),
                    "Press Escape or close this window to exit.",
                ],
            )
            cv2.imshow(WINDOW_NAME, display)
            last_displayed_sequence = packet.sequence

            if capture_error or inference_error:
                print(capture_error or inference_error, file=sys.stderr)
                exit_code = 1
                stop_event.set()
                break
            if cv2.waitKey(1) & 0xFF == 27:
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
        if final_capture_error:
            print(final_capture_error, file=sys.stderr)
            exit_code = 1
        return exit_code
    except KeyboardInterrupt:
        return 0
    except BaseException as exc:
        print(f"Phase 2 failed: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_event.set()
        keyboard_collector.stop()
        capture.join(timeout=3.0)
        if inference is not None:
            inference.close()
        if window_created:
            cv2.destroyAllWindows()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

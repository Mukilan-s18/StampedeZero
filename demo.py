"""
demo.py — StampedeZero Vision Tracker — Local Test Harness
Branch: mukil  |  Engineer 1 (Vision & Tracking Lead)

Standalone webcam loop to validate crowd_tracker.py before handing off
to Engineer 4's Streamlit integration.

Usage:
    python demo.py                    # use default webcam (index 0)
    python demo.py --source 1         # use webcam index 1
    python demo.py --source video.mp4 # test against a recorded video

Controls:
    Q  → quit
    R  → reset in/out counters
    S  → save a screenshot (PNG) with timestamp
    +  → increase line_y by 10 px
    -  → decrease line_y by 10 px
"""

import argparse
import datetime
import sys
import time

import cv2

import config as cfg
from crowd_tracker import VisionTracker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionTracker demo harness")
    p.add_argument(
        "--source",
        default="0",
        help="Webcam index (int) or video file path (default: 0)",
    )
    p.add_argument(
        "--line-y",
        type=float,
        default=cfg.LINE_Y_FRACTION,
        help="Counting line as fraction of frame height (default: %(default)s)",
    )
    p.add_argument(
        "--skip",
        type=int,
        default=cfg.SKIP_FRAMES,
        help="Run YOLO inference every Nth frame (default: %(default)s)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve source — webcam index or file path
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        sys.exit(1)

    print("=" * 60)
    print("  StampedeZero — VisionTracker Demo")
    print(f"  Source  : {source}")
    print(f"  Line Y  : {args.line_y * 100:.1f}% of frame height")
    print(f"  Skip    : every {args.skip} frames")
    print("  Keys    : Q=quit | R=reset | S=screenshot | +/- adjust line")
    print("=" * 60)

    tracker = VisionTracker(
        line_y_fraction=args.line_y,
        skip_frames=args.skip,
    )

    # FPS tracking
    fps_counter = 0
    fps_start = time.time()
    fps_display = 0.0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("[INFO] End of stream or capture error — exiting.")
            break

        # ── Run the tracker ───────────────────────────────────────────────
        payload = tracker.process_frame(raw_frame)
        display = payload["annotated_frame"]

        # ── FPS overlay ───────────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()

        cv2.putText(
            display,
            f"FPS: {fps_display:.1f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )

        # ── Console print (throttled) ─────────────────────────────────────
        if tracker._frame_count % 30 == 0:
            print(
                f"  Frame {tracker._frame_count:>6} | "
                f"IN={payload['total_in']:>3} | "
                f"OUT={payload['total_out']:>3} | "
                f"NET={payload['net_flow']:>+3} | "
                f"ON_SCREEN={payload['current_on_screen']:>2} | "
                f"FPS={fps_display:.1f}"
            )

        cv2.imshow("StampedeZero — VisionTracker (Q to quit)", display)

        # ── Keyboard controls ─────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:          # Q or ESC → quit
            break

        elif key == ord("r") or key == ord("R"):  # R → reset counters
            tracker.reset()
            print("[INFO] Counters reset.")

        elif key == ord("s") or key == ord("S"):  # S → screenshot
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"screenshot_{ts}.png"
            cv2.imwrite(fname, display)
            print(f"[INFO] Screenshot saved: {fname}")

        elif key == ord("+") or key == ord("="):  # + → line down
            tracker._line_y_fraction = min(0.95, tracker._line_y_fraction + 0.02)
            print(f"[INFO] Line Y → {tracker._line_y_fraction * 100:.1f}%")

        elif key == ord("-"):                      # - → line up
            tracker._line_y_fraction = max(0.05, tracker._line_y_fraction - 0.02)
            print(f"[INFO] Line Y → {tracker._line_y_fraction * 100:.1f}%")

    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal stats:", repr(tracker))


if __name__ == "__main__":
    main()

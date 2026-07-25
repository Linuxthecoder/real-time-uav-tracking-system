"""
Real-Time UAV Tracking System
==============================
Fixed for Python 3.11 / 3.12 / 3.13 / 3.14 virtual environments.

Usage:
    python main.py --source video.mp4
    python main.py --source video.mp4 --output outputs/output.mp4
"""

# ══════════════════════════════════════════════════════════════════
#  AUTO-INSTALL + SELF-RESTART
# ══════════════════════════════════════════════════════════════════
import subprocess, sys, os

_FLAG = "__UAV_READY__"

def _pip(*pkgs, **kw):
    """pip install with visible output so errors are clear."""
    cmd = [sys.executable, "-m", "pip", "install",
           "--no-warn-script-location", *pkgs]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0 and not kw.get("optional"):
        print(f"[WARN] pip returned non-zero for: {pkgs} — continuing anyway")

if _FLAG not in os.environ:
    print("╔══════════════════════════════════════════════════════╗")
    print("║   UAV TRACKER — Installing packages, please wait...  ║")
    print("╚══════════════════════════════════════════════════════╝")

    py = sys.version_info
    print(f"[INFO] Python {py.major}.{py.minor}.{py.micro}  →  {sys.executable}\n")

    # 1. opencv (always safe)
    print("[1/3] Installing opencv-python + scipy...")
    _pip("opencv-python", "scipy", "-q")

    # 2. ultralytics pulls in torch automatically at the right version
    print("[2/3] Installing ultralytics (auto-installs compatible torch)...")
    _pip("ultralytics", "-q")

    # 3. verify torch imports cleanly; if not, try upgrading
    print("[3/3] Verifying torch + numpy compatibility...")
    check = subprocess.run(
        [sys.executable, "-c", "import torch, numpy; print('OK', torch.__version__, numpy.__version__)"],
        capture_output=True, text=True
    )
    if "OK" in check.stdout:
        print(f"      torch + numpy: {check.stdout.strip()}")
    else:
        print(f"      [WARN] {check.stderr.strip()[:200]}")
        print("      Trying: pip install --upgrade torch numpy ultralytics...")
        _pip("--upgrade", "torch", "numpy", "ultralytics", "-q", optional=True)

    print("\n[OK] Setup complete — restarting with clean environment...\n")
    env = os.environ.copy()
    env[_FLAG] = "1"
    sys.exit(subprocess.run([sys.executable] + sys.argv, env=env).returncode)

# ══════════════════════════════════════════════════════════════════
#  REAL IMPORTS
# ══════════════════════════════════════════════════════════════════
import warnings
warnings.filterwarnings("ignore")

import cv2
import argparse
import time

from tracker import UAVTracker
from ui      import HUDRenderer
from utils   import ensure_dir, build_writer, resize_frame


def main():
    ap = argparse.ArgumentParser(description="Real-Time UAV Tracking System")
    ap.add_argument("--source",  required=True,  help="Input video (e.g. video.mp4)")
    ap.add_argument("--output",  default="outputs/output.mp4")
    ap.add_argument("--conf",    type=float, default=0.20)
    ap.add_argument("--width",   type=int,   default=1280)
    ap.add_argument("--classes", nargs="+",  type=int, default=None)
    args = ap.parse_args()

    # ── resolve video path ────────────────────────────────────────
    src = args.source
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        src,
        os.path.join(script_dir, src),
        os.path.join(os.getcwd(), src),
    ]
    resolved = None
    for c in candidates:
        if os.path.exists(c):
            resolved = os.path.abspath(c)
            break

    if resolved is None:
        print(f"\n[ERROR] Video not found: '{src}'")
        print(f"  Script folder: {script_dir}")
        print(f"  Files there  : {os.listdir(script_dir)}")
        print(f"\n  Make sure your video file is in the same folder as main.py")
        print(f"  Then run:  python main.py --source video.mp4\n")
        sys.exit(1)

    ensure_dir(os.path.dirname(args.output) or "outputs")

    cap = cv2.VideoCapture(resolved)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {resolved}")
        sys.exit(1)

    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = build_writer(args.output, src_fps, src_w, src_h)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          REAL-TIME UAV TRACKING SYSTEM v1.0              ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Source  : {os.path.basename(resolved):<46}║")
    print(f"║  Output  : {args.output:<46}║")
    print(f"║  Size    : {src_w}x{src_h} @ {src_fps:.1f} fps{'':<27}║")
    print(f"║  Frames  : {str(total):<46}║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    tracker  = UAVTracker(conf=args.conf, class_filter=args.classes)
    renderer = HUDRenderer(src_w, src_h, src_fps)

    fidx        = 0
    t_prev      = time.time()
    fps_display = src_fps

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        proc    = resize_frame(frame, args.width) if src_w > args.width else frame.copy()
        scale_x = src_w / proc.shape[1]
        scale_y = src_h / proc.shape[0]

        tracks = tracker.update(proc, fidx)

        for t in tracks:
            x1,y1,x2,y2 = t["bbox"]
            t["bbox"]   = (int(x1*scale_x), int(y1*scale_y),
                           int(x2*scale_x), int(y2*scale_y))
            t["center"] = (int(t["center"][0]*scale_x),
                           int(t["center"][1]*scale_y))

        now         = time.time()
        fps_display = 0.85*fps_display + 0.15/max(now-t_prev, 1e-4)
        t_prev      = now

        out_frame = renderer.render(frame.copy(), tracks, fidx, fps_display)
        writer.write(out_frame)
        fidx += 1

        if fidx % 15 == 0:
            pct = fidx / max(total, 1) * 100
            bar = "█"*int(pct/5) + "░"*(20-int(pct/5))
            print(f"\r  [{bar}] {pct:5.1f}%  frame {fidx:>5}/{total}"
                  f"  FPS:{fps_display:4.1f}  targets:{len(tracks)}",
                  end="", flush=True)

    cap.release()
    writer.release()
    print(f"\n\n[DONE] → {os.path.abspath(args.output)}\n")


if __name__ == "__main__":
    main()
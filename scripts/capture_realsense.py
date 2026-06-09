#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from affordance_grasp.io.realsense import capture_realsense_interactive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--stem", default="realsense_frame")
    parser.add_argument("--suffix", default=".npz", choices=[".npz", ".npy"])
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup_frames", type=int, default=30)
    parser.add_argument("--no_rgb_png", action="store_true", default=False)
    parser.add_argument("--no_depth_png", action="store_true", default=False)
    args = parser.parse_args()

    capture_realsense_interactive(
        output_dir=args.output_dir,
        stem=args.stem,
        suffix=args.suffix,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        save_rgb_png=not args.no_rgb_png,
        save_depth_png=not args.no_depth_png,
    )


if __name__ == "__main__":
    main()

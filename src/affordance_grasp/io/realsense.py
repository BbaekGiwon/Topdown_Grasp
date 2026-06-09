from pathlib import Path
import os
import sys

import cv2
import numpy as np

from affordance_grasp.io.dataset_io import ensure_dir, save_json, save_rgbd_bundle


def _candidate_pyrealsense2_paths():
    candidates = []

    env_root = os.environ.get("AFF_GRASP_LIBREALSENSE_ROOT", "").strip()
    if env_root:
        root = Path(env_root).expanduser()
        candidates.extend(
            [
                root / "build" / "Release",
                root / "build",
                root / "wrappers" / "python",
            ]
        )

    default_root = Path("/home/kist/librealsense")
    candidates.extend(
        [
            default_root / "build" / "Release",
            default_root / "build",
            default_root / "wrappers" / "python",
        ]
    )

    ordered = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        ordered.append(candidate)
    return ordered


def _import_pyrealsense2():
    try:
        import pyrealsense2 as rs

        return rs
    except ImportError as first_exc:
        last_exc = first_exc
        for candidate in _candidate_pyrealsense2_paths():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            try:
                import pyrealsense2 as rs

                return rs
            except ImportError as exc:
                last_exc = exc

        searched = [str(path) for path in _candidate_pyrealsense2_paths()]
        raise ImportError(
            "Could not import pyrealsense2. Install it in the active environment, or set "
            "AFF_GRASP_LIBREALSENSE_ROOT to your local librealsense build root. "
            f"Searched paths: {searched}. Current Python: {sys.version.split()[0]}. "
            "If you built pyrealsense2 for a different Python version, rebuild it for this interpreter."
        ) from last_exc


rs = _import_pyrealsense2()


def make_depth_vis(depth_raw):
    depth_vis = cv2.convertScaleAbs(depth_raw, alpha=0.03)
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)


def make_capture_path(output_dir, stem, capture_idx, suffix=".npz"):
    output_dir = ensure_dir(output_dir)
    name = f"{stem}_{capture_idx:03d}{suffix}"
    return output_dir / name


def save_capture_bundle(
    output_path,
    rgb_bgr,
    depth_raw,
    depth_m,
    K,
    save_rgb_png=True,
    save_depth_png=True,
):
    output_path = Path(output_path)
    save_rgbd_bundle(output_path, depth=depth_m, K=K, rgb=rgb_bgr)

    metadata = {
        "rgb_shape": list(rgb_bgr.shape),
        "depth_shape": list(depth_m.shape),
        "depth_dtype": str(depth_m.dtype),
        "rgb_dtype": str(rgb_bgr.dtype),
        "K": K.tolist(),
    }
    save_json(output_path.with_suffix(".json"), metadata)

    if save_rgb_png:
        cv2.imwrite(str(output_path.with_name(output_path.stem + "_rgb.png")), rgb_bgr)
    if save_depth_png:
        cv2.imwrite(
            str(output_path.with_name(output_path.stem + "_depth_vis.png")),
            make_depth_vis(depth_raw),
        )


def capture_realsense_interactive(
    output_dir,
    stem="realsense_frame",
    suffix=".npz",
    width=640,
    height=480,
    fps=30,
    warmup_frames=30,
    save_rgb_png=True,
    save_depth_png=True,
    window_name="RealSense RGBD Capture",
):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        print("[INFO] RealSense started")
        print("[INFO] Press 's' to save the current frame, 'q' to quit")

        for _ in range(warmup_frames):
            frames = pipeline.wait_for_frames()
            align.process(frames)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        capture_idx = 0

        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            rgb_bgr = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale

            intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
            K = np.array(
                [
                    [intr.fx, 0.0, intr.ppx],
                    [0.0, intr.fy, intr.ppy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            preview = np.hstack((rgb_bgr, make_depth_vis(depth_raw)))
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                output_path = make_capture_path(output_dir, stem, capture_idx, suffix=suffix)
                save_capture_bundle(
                    output_path,
                    rgb_bgr,
                    depth_raw,
                    depth_m,
                    K,
                    save_rgb_png=save_rgb_png,
                    save_depth_png=save_depth_png,
                )
                print(f"[INFO] Saved capture: {output_path}")
                capture_idx += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] RealSense stopped")


def capture_realsense_once(
    output_dir,
    stem="realsense_frame",
    suffix=".npz",
    width=640,
    height=480,
    fps=30,
    warmup_frames=30,
    save_rgb_png=True,
    save_depth_png=True,
    capture_idx=0,
):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        print("[INFO] RealSense started")
        for _ in range(warmup_frames):
            frames = pipeline.wait_for_frames()
            align.process(frames)

        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("Failed to capture aligned color/depth frame from RealSense")

        rgb_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        depth_m = depth_raw.astype(np.float32) * depth_scale
        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        K = np.array(
            [
                [intr.fx, 0.0, intr.ppx],
                [0.0, intr.fy, intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        output_path = make_capture_path(output_dir, stem, capture_idx, suffix=suffix)
        save_capture_bundle(
            output_path,
            rgb_bgr,
            depth_raw,
            depth_m,
            K,
            save_rgb_png=save_rgb_png,
            save_depth_png=save_depth_png,
        )
        print(f"[INFO] Saved capture: {output_path}")
        return output_path
    finally:
        pipeline.stop()
        print("[INFO] RealSense stopped")

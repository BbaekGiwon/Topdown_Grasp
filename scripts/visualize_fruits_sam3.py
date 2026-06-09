#!/usr/bin/env python3
"""
SAM3 multi-fruit grounded segmentation — documentation / presentation tool.

Runs in the `sam3` or `pipeline_all` conda environment.

Uses the Meta SAM3 SDK (build_sam3_predictor) with text-prompt grounding.
For each fruit query a fresh session is opened so masks are cleanly isolated
per fruit type.  Two output images are produced per input:

  {output_dir}/{stem}_fruit_all.png   — all fruits overlaid (colour-coded)
  {output_dir}/{stem}_fruit_grid.png  — per-fruit panel grid
  {output_dir}/{stem}_fruit_results.json

Usage
-----
  # Batch directory (recommended):
  conda activate sam3
  python scripts/visualize_fruits_sam3.py \\
      --image_dir data/sam3_fruit_vis/raw \\
      --fruits mango tomato lemon kiwi \\
      --output_dir data/sam3_fruit_vis/output

  # Single image:
  python scripts/visualize_fruits_sam3.py \\
      --image data/sam3_fruit_vis/raw/my_fruit.jpg \\
      --fruits mango tomato lemon kiwi

  # Auto-detect fruit types via Qwen2.5-VL (needs Qwen in the same env):
  python scripts/visualize_fruits_sam3.py \\
      --image_dir data/sam3_fruit_vis/raw --auto

  # Use SAM 3 (not 3.1):
  python scripts/visualize_fruits_sam3.py \\
      --image_dir data/sam3_fruit_vis/raw \\
      --fruits mango tomato lemon kiwi \\
      --sam3_version sam3
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Fruit colour palette (BGR)
_FRUIT_COLORS_BGR = {
    "mango":      (0,   160, 255),
    "tomato":     (50,  50,  220),
    "lemon":      (0,   220, 220),
    "kiwi":       (30,  130, 80),
    "apple":      (50,  50,  220),
    "banana":     (0,   200, 255),
    "orange":     (0,   130, 255),
    "grape":      (200, 0,   130),
    "strawberry": (80,  50,  255),
    "pear":       (50,  200, 160),
    "peach":      (130, 160, 255),
    "watermelon": (50,  100, 200),
    "cherry":     (40,  0,   180),
    "blueberry":  (180, 100, 30),
}

_FALLBACK_PALETTE = [
    (255, 80,  0),   (0,  255, 100), (100, 0,   255),
    (255, 0,   150), (0,  150, 255), (200, 255, 0),
    (200, 0,   200), (0,  200, 200), (255, 200, 0),
]


def _fruit_color(name: str, idx: int) -> tuple:
    return _FRUIT_COLORS_BGR.get(name.lower(),
                                  _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)])


def _mask_to_box(mask_bool: np.ndarray) -> list:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(args) -> tuple:
    """Return (rgb_uint8_HWC, stem)."""
    if args.input:
        npz = np.load(args.input)
        rgb = npz["rgb"]
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb, Path(args.input).stem
    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), Path(args.image).stem


def load_image_file(path: Path) -> tuple:
    """Return (rgb_uint8_HWC, stem) from a plain image path."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), path.stem


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def collect_images(image_dir: str) -> list:
    """Return sorted list of image Paths in a directory."""
    d = Path(image_dir)
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)


# ---------------------------------------------------------------------------
# Qwen auto-detect (optional)
# ---------------------------------------------------------------------------

def detect_fruits_qwen(image_rgb: np.ndarray, model_id: str,
                        device_map: str, torch_dtype_str: str) -> list:
    """Ask Qwen2.5-VL to list all visible fruit types."""
    import re
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype_map = {"auto": "auto", "bfloat16": torch.bfloat16,
                 "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(torch_dtype_str, "auto")

    print(f"  Qwen loading: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, device_map=device_map, torch_dtype=dtype
    )
    model.eval()

    prompt_text = (
        "List every fruit type visible in this image.\n"
        'Output exactly one JSON object: {"fruits": ["apple", "banana", ...]}\n'
        "Use lowercase singular English nouns. JSON only. No markdown."
    )
    chat = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt_text}
    ]}]
    prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image_rgb], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=128)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]
    print(f"  Qwen raw: {raw!r}")

    fruits = []
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            fruits = json.loads(match.group(0)).get("fruits", [])
        except json.JSONDecodeError:
            pass

    del model
    import torch as _torch
    _torch.cuda.empty_cache()
    return [str(f).strip().lower() for f in fruits if f]


# ---------------------------------------------------------------------------
# SAM3 — load / unload
# ---------------------------------------------------------------------------

def sam3_load(version: str = "sam3.1", use_fa3: bool = True,
              checkpoint_path: str = None):
    """
    Build and return a SAM3 predictor.

    The predictor exposes:
      handle_request(dict)        → dict
      handle_stream_request(dict) → generator of dicts
    """
    from sam3 import build_sam3_predictor

    print(f"  SAM3 loading (version={version}, use_fa3={use_fa3}) ...")
    predictor = build_sam3_predictor(
        version=version,
        compile=False,
        use_fa3=use_fa3,
        use_rope_real=True,
        async_loading_frames=False,
        checkpoint_path=checkpoint_path if checkpoint_path else None,
    )
    print("  SAM3 loaded")
    return predictor


# ---------------------------------------------------------------------------
# SAM3 — query one image with multiple fruit text prompts
# ---------------------------------------------------------------------------

def query_image(predictor, image_rgb: np.ndarray, fruit_names: list) -> dict:
    """
    For each fruit name, open a SAM3 session and add a text prompt.
    add_prompt() already returns detection masks for the prompted frame —
    no propagation needed for single-image use.

    Returns
    -------
    {
      "mango":  [{"mask": ndarray(bool), "score": float, "box_xyxy": list}, ...],
      "tomato": [...],
      ...
    }
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # SAM3 expects frames named 00000.jpg, 00001.jpg, …
        frame_path = os.path.join(tmpdir, "00000.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

        results = {}
        for fruit in fruit_names:
            print(f"  Querying: {fruit!r}")

            resp = predictor.handle_request({
                "type": "start_session",
                "resource_path": tmpdir,
            })
            session_id = resp["session_id"]

            # add_prompt runs detection on frame 0 and returns masks directly
            prompt_resp = predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": fruit,
            })

            instances = []
            outputs   = prompt_resp.get("outputs", {})
            obj_ids   = outputs.get("out_obj_ids", [])
            bin_masks = outputs.get("out_binary_masks")   # numpy (N, H, W) bool
            probs     = outputs.get("out_probs", [])

            if bin_masks is not None and len(bin_masks) > 0:
                import numpy as _np
                bin_masks = _np.array(bin_masks)
                for i in range(len(obj_ids)):
                    mask_bool = bin_masks[i].astype(bool)
                    if mask_bool.any():
                        score = float(probs[i]) if i < len(probs) else None
                        instances.append({
                            "mask":     mask_bool,
                            "score":    score,
                            "box_xyxy": _mask_to_box(mask_bool),
                        })

            results[fruit] = instances
            n = len(instances)
            print(f"    → {n} instance(s)" + (" detected" if n else " (not found)"))

            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })

    return results


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _draw_label(canvas_bgr, text, cx, cy, color_bgr,
                font_scale=0.65, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x0 = max(cx - tw // 2, 2)
    y0 = max(cy, th + 6)
    cv2.rectangle(canvas_bgr,
                  (x0 - 4, y0 - th - 4), (x0 + tw + 4, y0 + baseline + 2),
                  color_bgr, -1)
    cv2.putText(canvas_bgr, text, (x0, y0),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Output 1 — all fruits overlaid on one image
# ---------------------------------------------------------------------------

def make_overlay_all(image_rgb: np.ndarray, results: dict) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    for idx, (fruit, instances) in enumerate(results.items()):
        color = _fruit_color(fruit, idx)
        for inst in instances:
            mask = inst["mask"]
            if not mask.any():
                continue
            overlay = canvas.copy()
            overlay[mask] = color
            canvas = cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0.0)
            contours, _ = cv2.findContours(mask.astype(np.uint8),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 2)
            ys, xs = np.where(mask)
            cx, cy = int(xs.mean()), int(ys.mean())
            score = inst.get("score")
            label = f"{fruit} {score:.2f}" if score is not None else fruit
            _draw_label(canvas, label, cx, cy, color)

    return canvas


# ---------------------------------------------------------------------------
# Output 2 — per-fruit panel grid
# ---------------------------------------------------------------------------

def make_grid_panels(image_rgb: np.ndarray, results: dict,
                     panel_w: int = 400) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale   = panel_w / w
    panel_h = int(h * scale)
    base_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    base_sm  = cv2.resize(base_bgr, (panel_w, panel_h))

    panels = []
    for idx, (fruit, instances) in enumerate(results.items()):
        color = _fruit_color(fruit, idx)
        panel = (base_sm * 0.28).astype(np.uint8)

        found = False
        for inst in instances:
            mask = inst["mask"]
            if not mask.any():
                continue
            found = True
            mask_sm = cv2.resize(mask.astype(np.uint8), (panel_w, panel_h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            panel[mask_sm] = base_sm[mask_sm]
            overlay = panel.copy()
            overlay[mask_sm] = color
            panel = cv2.addWeighted(overlay, 0.28, panel, 0.72, 0.0)
            contours, _ = cv2.findContours(mask_sm.astype(np.uint8),
                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(panel, contours, -1, color, 2)

        n = len([i for i in instances if i["mask"].any()])
        status  = f"  x{n}" if found else "  NOT FOUND"
        title   = f" {fruit.upper()}{status}"
        bar_col = color if found else (55, 55, 55)
        title_bar = np.full((40, panel_w, 3), bar_col, dtype=np.uint8)
        cv2.putText(title_bar, title, (6, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(np.vstack([title_bar, panel]))

    if not panels:
        return np.zeros((panel_h + 40, panel_w, 3), dtype=np.uint8)

    ncols = min(4, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    blank = np.zeros_like(panels[0])
    while len(panels) < nrows * ncols:
        panels.append(blank)

    rows = [np.hstack(panels[r * ncols:(r + 1) * ncols]) for r in range(nrows)]
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Save outputs for one image
# ---------------------------------------------------------------------------

def save_outputs(image_rgb, stem, results, out_dir, panel_width):
    all_path  = out_dir / f"{stem}_fruit_all.png"
    grid_path = out_dir / f"{stem}_fruit_grid.png"
    cv2.imwrite(str(all_path),  make_overlay_all(image_rgb, results))
    cv2.imwrite(str(grid_path), make_grid_panels(image_rgb, results, panel_width))

    summary = {
        "stem": stem,
        "results": {
            fruit: [{"box_xyxy": i["box_xyxy"],
                     "mask_pixels": int(i["mask"].sum())}
                    for i in instances]
            for fruit, instances in results.items()
        },
    }
    json_path = out_dir / f"{stem}_fruit_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"    saved: {all_path.name}")
    print(f"    saved: {grid_path.name}")
    print(f"    saved: {json_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",     help="Single image file (.png / .jpg / .webp)")
    src.add_argument("--input",     help="RGB-D NPZ bundle (keys: rgb, depth, K)")
    src.add_argument("--image_dir", help="Directory of images — processed in batch")

    fruit_src = p.add_mutually_exclusive_group(required=True)
    fruit_src.add_argument("--fruits", nargs="+",
                           help="Fruit names, e.g. mango tomato lemon kiwi")
    fruit_src.add_argument("--auto", action="store_true",
                           help="Auto-detect fruit types per image via Qwen2.5-VL")

    p.add_argument("--qwen_model_id",    default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--qwen_device_map",  default="auto")
    p.add_argument("--qwen_torch_dtype", default="bfloat16",
                   choices=["auto", "bfloat16", "float16", "float32"])

    p.add_argument("--sam3_version",    default="sam3.1", choices=["sam3", "sam3.1"])
    p.add_argument("--sam3_use_fa3",    action="store_true", default=True,
                   help="Use Flash Attention 3 (requires Hopper GPU)")
    p.add_argument("--sam3_no_fa3",     dest="sam3_use_fa3", action="store_false",
                   help="Disable Flash Attention 3")
    p.add_argument("--sam3_checkpoint", default=None,
                   help="Local checkpoint path (auto-downloads from HF if omitted)")

    p.add_argument("--panel_width", type=int, default=400)
    p.add_argument("--output_dir",
                   default=str(ROOT / "data" / "sam3_fruit_vis" / "output"))

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect image list
    if args.image_dir:
        image_paths = collect_images(args.image_dir)
        if not image_paths:
            print(f"[ERROR] No images found in {args.image_dir}")
            return
        print(f"Batch mode: {len(image_paths)} image(s) in {args.image_dir}")
    else:
        image_paths = None

    fixed_fruits = [f.lower() for f in args.fruits] if args.fruits else None
    if fixed_fruits:
        print(f"Fruits: {fixed_fruits}")

    # Load SAM3 once
    print("\nLoading SAM3 ...")
    predictor = sam3_load(
        version=args.sam3_version,
        use_fa3=args.sam3_use_fa3,
        checkpoint_path=args.sam3_checkpoint,
    )

    def _process_one(image_rgb, stem, fruits):
        print(f"\n{'─'*55}")
        print(f"Image : {stem}  {image_rgb.shape[1]}×{image_rgb.shape[0]}")
        print(f"Fruits: {fruits}")
        results = query_image(predictor, image_rgb, fruits)
        save_outputs(image_rgb, stem, results, out_dir, args.panel_width)

    if image_paths is not None:
        for img_path in image_paths:
            try:
                image_rgb, stem = load_image_file(img_path)
            except FileNotFoundError as e:
                print(f"[WARN] {e} — skipping")
                continue

            if args.auto:
                print(f"\n  Qwen — detecting fruits in {stem} ...")
                fruits = detect_fruits_qwen(
                    image_rgb, args.qwen_model_id,
                    args.qwen_device_map, args.qwen_torch_dtype,
                )
                print(f"  Detected: {fruits}")
            else:
                fruits = fixed_fruits

            if not fruits:
                print(f"[WARN] No fruits for {stem} — skipping")
                continue
            _process_one(image_rgb, stem, fruits)
    else:
        image_rgb, stem = load_image(args)

        if args.auto:
            print("\nQwen — detecting fruit types ...")
            fruits = detect_fruits_qwen(
                image_rgb, args.qwen_model_id,
                args.qwen_device_map, args.qwen_torch_dtype,
            )
            print(f"  Detected: {fruits}")
        else:
            fruits = fixed_fruits

        if not fruits:
            print("[ERROR] No fruits to query — exiting.")
            return
        _process_one(image_rgb, stem, fruits)

    print(f"\nAll done.  Outputs → {out_dir}")


if __name__ == "__main__":
    main()

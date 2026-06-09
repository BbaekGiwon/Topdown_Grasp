#!/usr/bin/env python3
"""
Qwen2.5-VL + SAM3 segmentation stage.

Runs in the `pipeline_vision` conda environment.

Input:  RGB-D NPZ bundle (keys: rgb, depth, K)  OR  a plain RGB image
Output: binary mask PNG  +  JSON summary

Usage:
    python scripts/run_qwen_sam3_stage.py \
        --input  data/raw/scene.npz \
        --instruction "grab the tennis ball" \
        --output_dir  data/interim/

    python scripts/run_qwen_sam3_stage.py \
        --image  data/raw/scene.png \
        --instruction "grab the cup handle" \
        --output_dir  data/interim/

Output files:
    {output_dir}/{stem}_mask.png          — binary mask (255=object, 0=background)
    {output_dir}/{stem}_overlay.png       — visualisation overlay
    {output_dir}/{stem}_qwen_sam3.json    — Qwen parse + SAM3 info
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image(args) -> tuple[np.ndarray, str]:
    """Return (rgb_uint8_HWC, stem)."""
    if args.input:
        npz = np.load(args.input)
        rgb = npz["rgb"]                    # RealSense 캡처는 BGR (rs.format.bgr8)
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb, Path(args.input).stem
    else:
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {args.image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), Path(args.image).stem


# ---------------------------------------------------------------------------
# Qwen2.5-VL
# ---------------------------------------------------------------------------

def _build_qwen_prompt(instruction: str, image_w: int, image_h: int) -> str:
    return (
        "You are a robotic manipulation assistant.\n\n"
        "Analyze the image and user instruction carefully.\n\n"
        "Output exactly one JSON object with these keys:\n"
        '  "task"        : the manipulation task (short phrase)\n'
        '  "object"      : the target object name (short noun)\n'
        '  "object_part" : the part of the object to grasp (short noun)\n'
        '  "affordance"  : how to use that part (short verb phrase)\n\n'
        "Rules:\n"
        "- Short lowercase English phrases for all text fields.\n"
        "- JSON only. No markdown. No explanation.\n\n"
        f"User instruction: {instruction}"
    )


def run_qwen(image_rgb: np.ndarray, instruction: str, model_id: str,
             device_map: str, torch_dtype_str: str, max_new_tokens: int) -> dict:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    h, w = image_rgb.shape[:2]
    print(f"  Qwen loading: {model_id}")

    dtype_map = {"auto": "auto", "bfloat16": torch.bfloat16,
                 "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(torch_dtype_str, "auto")

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, device_map=device_map, torch_dtype=dtype
    )
    print("  Qwen loaded")

    prompt_text = _build_qwen_prompt(instruction, w, h)
    chat = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt_text}
    ]}]
    prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image_rgb], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]
    print(f"  Qwen raw: {raw!r}")

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {"error": "no JSON", "raw": raw}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return {"error": str(e), "raw": raw}

    def _n(k):
        return re.sub(r"\s+", " ", str(payload.get(k, "")).strip().lower())

    del model
    import torch
    torch.cuda.empty_cache()

    return {
        "task":        _n("task"),
        "object":      _n("object"),
        "object_part": _n("object_part"),
        "affordance":  _n("affordance"),
    }


# ---------------------------------------------------------------------------
# SAM3
# ---------------------------------------------------------------------------

def _build_sam3_queries(object_name: str, part_name: str) -> list[str]:
    if object_name:
        return [object_name]
    return []


def _run_sam3_query(model, processor, image_rgb, vision_embeds,
                    text: str, threshold: float, mask_threshold: float, device) -> Optional[dict]:
    import torch

    text_inputs = processor(text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(vision_embeds=vision_embeds, **text_inputs)

    h, w = image_rgb.shape[:2]
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=[(h, w)],
    )[0]

    masks  = results.get("masks", [])
    scores = results.get("scores", [])
    boxes  = results.get("boxes", [])
    if len(masks) == 0:
        return None

    best_idx = int(np.argmax([float(s) for s in scores]))
    mask_np = masks[best_idx]
    if hasattr(mask_np, "cpu"):
        mask_np = mask_np.cpu()
    box = boxes[best_idx]
    if hasattr(box, "cpu"):
        box = box.cpu()
    return {
        "mask":          np.array(mask_np).astype(bool),
        "score":         float(scores[best_idx]),
        "box_xyxy":      box.tolist() if hasattr(box, "tolist") else list(box),
        "num_instances": len(masks),
        "all_scores":    [float(s) for s in scores],
    }


def run_sam3(image_rgb: np.ndarray, object_name: str, part_name: str,
             model_id: str, threshold: float, mask_threshold: float) -> dict:
    import torch
    from transformers import Sam3Processor, Sam3Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  SAM3 loading: {model_id}")
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    print("  SAM3 loaded")

    img_inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

    queries = _build_sam3_queries(object_name, part_name)
    print(f"  Queries: {queries}")

    best_result = None
    used_query = None
    for q in queries:
        result = _run_sam3_query(model, processor, image_rgb,
                                 vision_embeds, q, threshold, mask_threshold, device)
        if result is not None:
            used_query = q
            best_result = result
            print(f"  OK: {q!r}  score={result['score']:.3f}  mask_px={result['mask'].sum()}")
            break
        print(f"  No result: {q!r}")

    del model
    torch.cuda.empty_cache()

    if best_result is None:
        return {"used_query": None, "queries_tried": queries, "mask": None, "score": None}

    return {
        "used_query":    used_query,
        "queries_tried": queries,
        "mask":          best_result["mask"],
        "score":         best_result["score"],
        "box_xyxy":      best_result["box_xyxy"],
        "num_instances": best_result["num_instances"],
        "all_scores":    best_result["all_scores"],
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_overlay(image_rgb: np.ndarray, qwen: dict, sam3: dict) -> np.ndarray:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask = sam3.get("mask")
    if mask is not None and mask.any():
        overlay = canvas.copy()
        overlay[mask] = (0, 0, 220)
        canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0.0)
        box = sam3.get("box_xyxy")
        if box:
            x0, y0, x1, y1 = [int(v) for v in box]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 220, 0), 2)
    lines = [
        f"object: {qwen.get('object', '')}",
        f"part  : {qwen.get('object_part', '')}",
        f"query : {sam3.get('used_query', 'N/A')}",
        f"score : {sam3.get('score', 0.0):.3f}" if sam3.get("score") else "score : N/A",
    ]
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (12, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 80), 2, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="RGB-D NPZ bundle (keys: rgb, depth, K)")
    src.add_argument("--image", help="Plain RGB/BGR image file (.png/.jpg)")

    p.add_argument("--instruction", required=True, help="Manipulation instruction")

    p.add_argument("--qwen_model_id",       default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--qwen_device_map",     default="auto")
    p.add_argument("--qwen_torch_dtype",    default="bfloat16",
                   choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--qwen_max_new_tokens", type=int, default=256)

    p.add_argument("--sam3_model_id",       default="facebook/sam3")
    p.add_argument("--sam3_threshold",      type=float, default=0.5)
    p.add_argument("--sam3_mask_threshold", type=float, default=0.5)

    p.add_argument("--output_dir", default=str(ROOT / "data" / "interim"))

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image_rgb, stem = load_image(args)
    print(f"Image: {stem}  shape={image_rgb.shape}")

    # Stage 1: Qwen
    print(f"\n[1/2] Qwen  instruction={args.instruction!r}")
    qwen = run_qwen(
        image_rgb=image_rgb,
        instruction=args.instruction,
        model_id=args.qwen_model_id,
        device_map=args.qwen_device_map,
        torch_dtype_str=args.qwen_torch_dtype,
        max_new_tokens=args.qwen_max_new_tokens,
    )
    print(f"  object={qwen.get('object')!r}  part={qwen.get('object_part')!r}")

    # Stage 2: SAM3
    print(f"\n[2/2] SAM3")
    sam3 = run_sam3(
        image_rgb=image_rgb,
        object_name=qwen.get("object", ""),
        part_name=qwen.get("object_part", ""),
        model_id=args.sam3_model_id,
        threshold=args.sam3_threshold,
        mask_threshold=args.sam3_mask_threshold,
    )

    mask = sam3.get("mask")
    if mask is None or not mask.any():
        print("\n[WARN] No mask produced — exiting with error.")
        sys.exit(1)

    # Save outputs
    mask_path = out_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
    print(f"\nSaved mask: {mask_path}")

    overlay_path = out_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), draw_overlay(image_rgb, qwen, sam3))
    print(f"Saved overlay: {overlay_path}")

    summary = {
        "stem":        stem,
        "instruction": args.instruction,
        "qwen": {
            "task":        qwen.get("task", ""),
            "object":      qwen.get("object", ""),
            "object_part": qwen.get("object_part", ""),
            "affordance":  qwen.get("affordance", ""),
        },
        "sam3": {
            "model":         args.sam3_model_id,
            "queries_tried": sam3.get("queries_tried", []),
            "used_query":    sam3.get("used_query"),
            "score":         sam3.get("score"),
            "box_xyxy":      sam3.get("box_xyxy"),
            "num_instances": sam3.get("num_instances"),
            "mask_pixels":   int(mask.sum()),
        },
        "outputs": {
            "mask":    str(mask_path),
            "overlay": str(overlay_path),
        },
    }
    json_path = out_dir / f"{stem}_qwen_sam3.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON:  {json_path}")

    print(f"\nDone.  mask={mask_path}")


if __name__ == "__main__":
    main()

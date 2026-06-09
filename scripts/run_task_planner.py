#!/usr/bin/env python3
"""
Task planner: Qwen2.5-VL 로 자연어 지시문을 pick task 리스트로 파싱.

이미지는 선택 사항 — 텍스트만으로도 파싱 가능.

Usage:
    python scripts/run_task_planner.py \\
        --instruction "망고 2개랑 오렌지 1개 집어줘" \\
        --output_dir data/interim/ \\
        --stem task

    python scripts/run_task_planner.py \\
        --instruction "grab 2 mangoes and 1 orange" \\
        --input data/raw/scene.npz \\
        --output_dir data/interim/ \\
        --stem task

Output:
    {output_dir}/{stem}_task_plan.json
    {
      "instruction": "...",
      "plan": [
        {"object": "mango",  "count": 2},
        {"object": "orange", "count": 1}
      ]
    }
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

_TASK_PLAN_PROMPT = """\
You are a robotic task planner. The robot can pick and place objects one at a time.
Parse the user's instruction into an ordered list of pick tasks.

Output exactly one JSON array. Each element must have:
  "object": the object name in English (lowercase noun, e.g. "mango", "orange", "apple")
  "count":  how many times to pick this object (positive integer, default 1 if unspecified)

Rules:
- Output a JSON array ONLY. No markdown. No explanation. No extra text.
- Translate object names to English if they are in another language.
- Preserve the order from the instruction.

User instruction: {instruction}"""


# ---------------------------------------------------------------------------
# Image loader (optional)
# ---------------------------------------------------------------------------

def _load_image(args) -> np.ndarray | None:
    if getattr(args, 'input', None):
        npz = np.load(args.input)
        rgb = npz["rgb"]
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if getattr(args, 'image', None):
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise FileNotFoundError(f"Cannot read: {args.image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return None


# ---------------------------------------------------------------------------
# Qwen task planner
# ---------------------------------------------------------------------------

def run_qwen_task_planner(instruction: str, image_rgb: np.ndarray | None,
                          model_id: str, device_map: str,
                          torch_dtype_str: str, max_new_tokens: int) -> list[dict]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype_map = {"auto": "auto", "bfloat16": torch.bfloat16,
                 "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(torch_dtype_str, "auto")

    print(f"  Qwen loading: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, device_map=device_map, torch_dtype=dtype)
    print("  Qwen loaded")

    prompt_text = _TASK_PLAN_PROMPT.format(instruction=instruction)

    if image_rgb is not None:
        chat = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt_text}
        ]}]
        prompt = processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image_rgb],
                           return_tensors="pt").to(model.device)
    else:
        chat = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        prompt = processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    print(f"  Qwen raw: {raw!r}")

    del model
    torch.cuda.empty_cache()

    # Parse JSON array
    match = re.search(r'\[.*?\]', raw, flags=re.DOTALL)
    if not match:
        print(f"[ERROR] No JSON array in Qwen output: {raw!r}")
        sys.exit(1)
    try:
        plan_raw = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse error: {e}\nRaw: {raw!r}")
        sys.exit(1)

    if not isinstance(plan_raw, list) or len(plan_raw) == 0:
        print(f"[ERROR] Empty or invalid plan: {plan_raw!r}")
        sys.exit(1)

    plan = []
    for item in plan_raw:
        obj   = re.sub(r"\s+", " ", str(item.get("object", "")).strip().lower())
        count = max(1, int(item.get("count", 1)))
        if obj:
            plan.append({"object": obj, "count": count})
    return plan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", help="RGB-D NPZ (선택 사항, Qwen 컨텍스트용)")
    src.add_argument("--image", help="RGB 이미지 파일 (선택 사항, Qwen 컨텍스트용)")

    p.add_argument("--instruction", required=True,
                   help="자연어 지시문 (예: '망고 2개랑 오렌지 1개 집어줘')")

    p.add_argument("--qwen_model_id",       default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--qwen_device_map",     default="auto")
    p.add_argument("--qwen_torch_dtype",    default="bfloat16",
                   choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--qwen_max_new_tokens", type=int, default=128)

    p.add_argument("--output_dir", default=str(ROOT / "data" / "interim"))
    p.add_argument("--stem",       default="task",
                   help="출력 파일 stem (default: task → task_task_plan.json)")

    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load image (optional)
    image_rgb = None
    try:
        image_rgb = _load_image(args)
    except FileNotFoundError as e:
        print(f"[WARN] {e} — 이미지 없이 진행")

    print(f"\n[Task Planner]  instruction={args.instruction!r}")
    if image_rgb is not None:
        print(f"  image shape: {image_rgb.shape}")
    else:
        print("  image: 없음 (텍스트 전용)")

    plan = run_qwen_task_planner(
        instruction=args.instruction,
        image_rgb=image_rgb,
        model_id=args.qwen_model_id,
        device_map=args.qwen_device_map,
        torch_dtype_str=args.qwen_torch_dtype,
        max_new_tokens=args.qwen_max_new_tokens,
    )

    total = sum(item["count"] for item in plan)
    print(f"\n[Task Plan]  총 {total}회 pick")
    for i, item in enumerate(plan):
        print(f"  {i+1}. {item['object']} × {item['count']}")

    out_path = out_dir / f"{args.stem}_task_plan.json"
    with open(out_path, "w") as f:
        json.dump({"instruction": args.instruction, "plan": plan}, f,
                  indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

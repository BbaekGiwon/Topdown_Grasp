# Qwen-VLPart-AnyGrasp Pipeline

## Goal

Define the current open-vocabulary grasping pipeline as:

1. `RGB + open-vocabulary instruction -> Qwen2.5-VL`
2. `Qwen2.5-VL -> task / object / object_part / affordance` extraction
3. `VLPart -> object bounding box`
4. `VLPart -> part segmentation inside the object ROI`
5. `masked depth -> point cloud crop`
6. `AnyGrasp -> grasp pose proposals`
7. `postprocess -> best grasp selection`

## Why This Structure

The design separates:

- language understanding: Qwen2.5-VL
- open-vocabulary detection and part segmentation: VLPart
- grasp synthesis from local geometry: AnyGrasp

This separation makes each stage easier to inspect, replace, and debug.

## Recommended Package Layout

```text
src/affordance_grasp/
  perception/
    open_vocab_parser.py
    vlpart_wrapper.py
  grasp/
    anygrasp_wrapper.py
  pipeline/
    open_vocab_pipeline.py
scripts/
  run_open_vocab_pipeline.py
```

## Stage Contracts

### 1. Qwen2.5-VL

Input:

- RGB image
- user instruction

Output schema:

```json
{
  "task": "grasp",
  "object": "mug",
  "object_part": "handle",
  "affordance": "hold"
}
```

Design notes:

- Force JSON-only output.
- Keep labels short and normalized.
- Treat Qwen as a semantic planner, not the final segmentation model.

### 2. VLPart Object Localization

Input:

- full RGB
- object label from Qwen

Output:

- object bounding boxes
- optional object masks or scores

Design notes:

- Use bounding box selection before part segmentation.
- Keep the crop stage explicit because it stabilizes the part prompt.

### 3. VLPart Part Segmentation

Input:

- cropped object RGB
- part label from Qwen

Output:

- part mask in crop coordinates

Post-step:

- lift the crop mask back to full-image coordinates

### 4. AnyGrasp

Input:

- depth
- intrinsics `K`
- full RGB or cropped RGB
- binary part mask

Output:

- grasp candidates in camera frame
- scores

Design notes:

- Build point cloud only from the part mask.
- If AnyGrasp expects a 3D crop, convert the 2D mask to a masked point cloud first.

## Practical Mapping From Current Repo

Target:

- `perception/open_vocab_parser.py`: semantic extraction only
- `perception/vlpart_wrapper.py`: object box + part mask
- `pipeline/open_vocab_pipeline.py`: orchestration
- `grasp/anygrasp_wrapper.py`: AnyGrasp bridge

## Main Implementation Rule

Do not let Qwen directly decide geometry. Qwen should only provide semantic targets. Geometry should be owned by VLPart and AnyGrasp.

## Suggested First Milestone

1. Keep existing RGB-D IO and visualization utilities.
2. Keep Qwen JSON extraction explicit and inspectable.
3. Use VLPart for object localization and part grounding.
4. Use AnyGrasp for geometry-driven proposal generation.
5. Save intermediate artifacts and summaries at each stage.

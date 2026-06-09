# Affordance Grasp
by HARILAB GWBaek, 2026.03.16 Update

Integrated pipeline workspace for:

- RealSense RGB-D capture
- Qwen2.5-VL semantic parsing
- VLPart object localization and part segmentation
- AnyGrasp grasp proposal generation
- Camera-to-robot-base grasp transform

## Layout

- `third_party/`: external repos kept as separately managed code
- `src/affordance_grasp/`: reusable local pipeline code
- `scripts/`: thin entrypoints for each pipeline step
- `configs/`: camera, robot, calibration settings
- `data/`: captured inputs, intermediate artifacts, final outputs

## Pipeline

1. Capture `rgb`, `depth`, `K`
2. Parse the instruction with Qwen2.5-VL
3. Ground object and part regions with VLPart
4. Run AnyGrasp on the selected part mask
5. Transform grasp to robot base frame

For the architecture details, see:

- `docs/qwen_vlpart_anygrasp_architecture.md`
- `src/affordance_grasp/pipeline/open_vocab_pipeline.py`
- `scripts/run_open_vocab_pipeline.py`

## Capture Format

- Use `.npz` as the default raw RGB-D bundle format.
- Prefer `.npz` over dict-style `.npy` to avoid NumPy pickle compatibility issues across environments.

## Setup

This project is currently used with a Conda environment defined in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate aff_grasp
```

Because the repository does not yet define a packaged install, run scripts from the repository root and expose `src/` on `PYTHONPATH`.

```bash
cd /path/to/Affordance_grasp
export PYTHONPATH=$PWD/src
```

## External Dependencies

Required for the current pipeline:

- Python 3.9
- `numpy`, `opencv-python`
- `torch`, `transformers`
- `pyrealsense2` for RealSense capture
- VLPart code and weights under `third_party/VLPart`
- AnyGrasp SDK and checkpoints under `third_party/anygrasp_sdk`

Notes:

- Qwen/VLPart/AnyGrasp inference are expected to run on a CUDA-enabled GPU environment.
- RealSense capture requires the camera to be connected and the librealsense runtime to be available.

## Basic Workflow

1. Capture an RGB-D scene with RealSense and save it under `data/raw/`.
2. Run the open-vocabulary pipeline with a natural-language grasp instruction.
3. Inspect the saved Qwen parse, VLPart overlay, and part mask under `data/interim/`.
4. Review AnyGrasp summaries and pose outputs under `data/outputs/`.

## Example Commands

Capture one RGB-D frame:

```bash
python scripts/capture_realsense.py --stem scene_000 --suffix .npz
```

Run the open-vocabulary pipeline from saved RGB-D input:

```bash
python scripts/run_open_vocab_pipeline.py \
  --input data/raw/scene_000.npz \
  --instruction "grasp the handle of the mug" \
  --run_grasp
```

Optional stage-wise usage:

```bash
python scripts/run_qwen_vlpart_only.py \
  --input data/raw/scene_000.npz \
  --instruction "grasp the handle of the mug"

python scripts/run_anygrasp_only.py \
  --rgb_input path/to/aligned_scene_000_rgb.png \
  --depth_input data/raw/scene_000.npz \
  --mask_input data/interim/scene_000_vlpart_part_mask.png
```

## Notes

- Keep modifications to `third_party/` minimal.
- Put orchestration logic in `src/` and expose it through `scripts/`.
- Store scene-level artifacts under `data/raw`, `data/interim`, and `data/outputs`.

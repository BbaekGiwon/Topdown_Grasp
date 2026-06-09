# Third-Party Dependencies

This directory contains external repositories used by the affordance grasp pipeline.

## Included Repositories

### `VLPart`

- Purpose: open-vocabulary object localization and part grounding
- Used by: `scripts/run_open_vocab_pipeline.py`, `scripts/run_qwen_vlpart_only.py`
- Upstream docs: `third_party/VLPart/README.md`

### `anygrasp_sdk`

- Purpose: grasp candidate generation from RGB-D input
- Used by: `scripts/run_open_vocab_pipeline.py`, `scripts/run_anygrasp_only.py`
- Local integration points: `src/affordance_grasp/grasp/anygrasp_wrapper.py`

## Local Usage Notes

- Keep direct edits inside `third_party/` minimal.
- Prefer adding project-specific wrappers and orchestration code under `src/affordance_grasp/` and `scripts/`.
- Model checkpoints, generated outputs, cache files, and other local artifacts under `third_party/` are excluded by `.gitignore`.

## Recommended Tracking Policy

- Track source code and configuration files needed to run the pipeline.
- Do not track checkpoints, inference outputs, temporary files, or `__pycache__`.
- If a local modification to an external repository is necessary, document it in the commit message or add a short note near the modified file.

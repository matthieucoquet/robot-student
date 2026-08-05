# Repository Guidelines

- `robot-student` implements high-throughput PPO with PyTorch, TensorDict, and Genesis. Optimize rollout collection, environment stepping, return computation, and PPO updates.
- `src/robot_student/` is the library; `experiment/{ant,g1}/` contains experiment code and MJCF assets; ignored outputs belong in `result/`.
- Use `uv` and Python `>=3.13,<3.14`. Install with `uv sync --locked`.
- Before handoff, run `uv run ruff format --check` and `uv run ruff check`. Format with `uv run ruff format`; run focused tests with `uv run pytest <test-path>` when available.
- Do not run full training or evaluation for routine verification. They generally require CUDA, may use Weights & Biases, and write artifacts.
- Preserve tensor shapes, dtypes, device placement, `TensorDict` batch semantics, and inference/no-grad boundaries.
- In hot paths, avoid unnecessary synchronization, CPU transfers, allocations, and per-environment Python loops. Prefer batched operations and preallocated buffers.
- Follow Ruff's 140-character line length. Use descriptive names without abbreviations; standard terms such as PPO, KL, TD, DOF, and MLP are fine.
- Keep changes focused, preserve unrelated work, and keep documentation aligned with behavior.
- Do not preserve backward compatibility.

# Repository Guidelines

## Purpose

This Python library implements high-throughput PPO training with Genesis, with DeepMimic, BeyondMimic, and potential Newton support planned later. Prioritize training throughput: avoid unnecessary synchronization, device transfers, allocations, and Python overhead in performance-critical paths.

## Layout

- `src/robot_student/` contains PPO, rollout storage, Genesis integration, environments, models, and utilities.
- `experiment/` contains Python-configured training experiments; `experiment/ant/` is the current example.
- `result/` contains generated training output and is ignored.

## Development

The project uses `uv` and requires Python `>=3.13,<3.14`.

- `uv sync --locked` installs the locked environment.
- `uv run python -m experiment.ant.train` launches the CUDA Ant experiment with its viewer.
- `uv run ruff format` formats the code.
- `uv run ruff format --check` and `uv run ruff check` run the CI checks.

## Conventions

Follow PEP 8, use descriptive names without abbreviations, and follow Ruff's configuration in `pyproject.toml`. Keep changes clean and minimal, preserve user changes and `uv.lock`, and keep documentation aligned with actual behavior.

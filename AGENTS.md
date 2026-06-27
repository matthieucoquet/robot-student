# Repository Guidelines

## Project Goals

The goal of this library is to implement Proximal Policy Optimization (PPO)
for efficient training with the Genesis physics platform, with potential
support for Newton later. The implementation roadmap is:

1. Implement PPO.
2. Build DeepMimic and BeyondMimic capabilities on top of the PPO
   implementation.

Implementation choices must prioritize training throughput. The PPO loop and
its integration with Genesis should run as fast as possible, avoiding
unnecessary synchronization, data transfers, allocations, and Python overhead
in performance-critical paths.

## Project Structure & Module Organization

This is a small Python package managed with `uv`.

- `src/robot_student/` contains the importable package code.
- `pyproject.toml` defines package metadata, Python version requirements, build backend, and the `robot-student` script.
- `README.md` is currently empty; update it when adding user-facing behavior.

Library code is inside `src/robot_student/`. Configuration of training is also done in python in the `experiment` folder.

## Build, Test, and Development Commands

- `uv sync` installs the project environment from `pyproject.toml` and `uv.lock`.
- `uv run robot-student` runs the configured console script.
- `uv run ruff format` formats Python code.
- `uv run ruff check` lints Python code.

The project requires Python `>=3.13`, as declared in `pyproject.toml`.

## Coding Style & Naming Conventions

- Use standard PEP8 Python conventions
- Don't use abreviations in names.

Ruff is configured in `pyproject.toml` for formatting and linting. Run `uv run ruff format` before committing, and use `uv run ruff check` to catch lint issues.

## Agent-Specific Instructions

Before editing, inspect existing files and preserve the minimal package structure. Do not remove user changes or generated lock files unless explicitly asked. Keep documentation synchronized with actual configured tools.

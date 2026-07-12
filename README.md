# robot-student

Really early PPO training experiments for robotic reinforcement learning in Genesis.

## Setup

Install dependencies with uv:

```sh
uv sync
```

## Train Ant

Launch the Ant PPO experiment with:

```sh
uv run python -m experiment.ant.train
```

The current Ant experiment uses Genesis with CUDA and opens the viewer.

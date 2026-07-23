# robot-student

Really early PPO training experiments for robotic reinforcement learning in Genesis.

## Setup

Install the dependencies with uv:

```sh
uv sync
```

## Weights & Biases

When using Weights & Biases for the metrics storage, first provide `WANDB_API_KEY` through the environment variables, then login to W&B:

```sh
uv run wandb login
```

## Train Ant

Launch the Ant PPO experiment with:

```sh
uv run python -m experiment.ant.train
```

## Evaluate Ant

Set `WEIGHTS_AND_BIASES_RUN_ID` near the top of `experiment/ant/evaluate.py` to the W&B run to evaluate.

Launch the evaluation with:

```sh
uv run python -m experiment.ant.evaluate
```

## Acknowledgements

This project builds on ideas from two excellent projects: [MimicKit](https://github.com/xbpeng/MimicKit) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl).

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

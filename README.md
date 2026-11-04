# robot-student

Really early PPO training experiments for robotic reinforcement learning in Genesis.

## Setup

Initialize the submodules, then install the dependencies with uv:

```sh
git submodule update --init
uv sync
```

To also install the optional GMR motion-retargeting dependency:

```sh
uv sync --extra gmr
```

## Weights & Biases

When using Weights & Biases for the metrics storage, first provide `WANDB_API_KEY` through the environment variables, then login to W&B:

```sh
uv run wandb login
```

## Train

Launch the G1 DeepMimic experiment with:

```sh
uv run python -m experiment.g1.train_deepmimic
```

## Evaluate Policy

Edit the run_id from the training run in `experiment/g1/evaluate.py`.

Launch the evaluation with:

```sh
uv run python -m experiment.g1.evaluate
```

## Acknowledgements

This project builds on ideas from two excellent projects: [MimicKit](https://github.com/xbpeng/MimicKit) and [RSL-RL](https://github.com/leggedrobotics/rsl_rl).

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

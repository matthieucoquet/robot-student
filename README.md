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


## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

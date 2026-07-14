FROM python:3.13-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace/robot-student

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src/ ./src/
COPY experiment/ ./experiment/

RUN uv sync --locked --no-dev

CMD ["uv", "run", "python", "-m", "experiment.ant.train"]

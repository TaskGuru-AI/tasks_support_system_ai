FROM python:3.12-slim AS base
ENV PIP_DEFAULT_TIMEOUT=100

# hadolint global ignore=DL3008,DL3009
# install libgomp1
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app

RUN pip install --no-cache-dir poetry==1.8.5

FROM base AS deps
COPY pyproject.toml poetry.lock README.md ./
COPY .streamlit ./.streamlit
COPY setup ./setup

RUN set -e && \
    poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

RUN poetry run python -m setup --environment local_docker

FROM deps AS final
COPY tasks_support_system_ai/ ./tasks_support_system_ai/
RUN poetry install --only-root --no-interaction --no-ansi


ENV PYTHONUNBUFFERED=1
ENV IS_DOCKER=1

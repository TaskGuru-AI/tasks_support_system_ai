FROM python:3.12-slim AS base

ENV PIP_DEFAULT_TIMEOUT=100
ENV POETRY_HOME=/app/.poetry
ENV POETRY_CACHE_DIR=/app/.cache
# ENV POETRY_VIRTUALENVS_PATH=/app/.virtualenvs
ENV NLTK_DATA=/app/nltk_data
ENV POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1=12.2.0-14 \
    && apt-get install -y --no-install-recommends curl=7.88.1-10+deb12u8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create all directories and set permissions as root
RUN mkdir -p /app/data /app/logs \
    /app/.poetry /app/.cache \
    && chmod -R 777 /app

RUN pip install --no-cache-dir poetry==1.8.5

# Install dependencies as root
FROM base AS deps
COPY pyproject.toml poetry.lock README.md ./
COPY .streamlit ./.streamlit
COPY scripts ./scripts

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

USER 1002:1002

RUN poetry run python scripts/prefetch_data.py

FROM deps AS final
COPY tasks_support_system_ai/ ./tasks_support_system_ai/
RUN poetry install --only-root --no-interaction --no-ansi

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV IS_DOCKER=1

RUN chown -R 1002:1002 /app \
    && chmod -R 777 /app/logs \
    && chmod -R 777 /app/data

# # Create a non-root user (optional if using user from docker-compose)
RUN groupadd -g 1002 github \
    && useradd -u 1002 -g github -s /bin/bash -m github

FROM python:3.12-slim AS base

# install libgomp1
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    cargo && \
    cargo install just && \
    apt-get remove -y cargo && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y curl
RUN apt-get install just
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN pip install poetry

FROM base AS deps
COPY pyproject.toml poetry.lock README.md ./
RUN set -e && \
    poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

FROM deps AS final
COPY tasks_support_system_ai/ ./tasks_support_system_ai/
RUN poetry install --only-root --no-interaction --no-ansi


ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

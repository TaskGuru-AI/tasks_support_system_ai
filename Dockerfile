FROM python:3.12-slim AS base

ARG APP_USER=github

# install libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y curl

WORKDIR /app
RUN mkdir -p /app/data && chown -R $APP_USER:$APP_USER /app/data

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

FROM python:3.12-slim AS base

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g appuser -s /bin/sh -m appuser

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app /app/data /.cache/pypoetry && \
    chown -R appuser:appuser /app /.cache

WORKDIR /app
ENV PYTHONUNBUFFERED=1

USER appuser
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.in-project true

FROM base AS deps
COPY --chown=appuser:appuser pyproject.toml poetry.lock README.md ./
RUN poetry install --no-root --no-interaction --no-ansi

FROM deps AS final
COPY --chown=appuser:appuser tasks_support_system_ai/ ./tasks_support_system_ai/
RUN poetry install --only-root --no-interaction --no-ansi

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

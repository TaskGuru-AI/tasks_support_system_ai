FROM python:3.12-slim AS base
ENV PIP_DEFAULT_TIMEOUT=100

# install libgomp1
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1=12.2.0-14 \
    && apt-get install -y --no-install-recommends curl=7.88.1-10+deb12u8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/data && chmod -R 777 /app/data

RUN pip install --no-cache-dir poetry==1.8.5

FROM base AS deps
COPY pyproject.toml poetry.lock README.md ./
COPY .streamlit ./.streamlit
COPY scripts ./scripts

RUN set -e && \
    poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

RUN poetry run python scripts/prefetch_data.py

FROM deps AS final
COPY tasks_support_system_ai/ ./tasks_support_system_ai/
RUN poetry install --only-root --no-interaction --no-ansi


ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV IS_DOCKER=1

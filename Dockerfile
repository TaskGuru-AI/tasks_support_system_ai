FROM python:3.12-slim AS base

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install just binary directly
RUN JUST_VERSION=$(curl -s "https://api.github.com/repos/casey/just/releases/latest" | grep -Po '"tag_name": "\K[^"]*') && \
    curl -L "https://github.com/casey/just/releases/download/${JUST_VERSION}/just-${JUST_VERSION}-x86_64-unknown-linux-musl.tar.gz" | tar xz -C /usr/local/bin just


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

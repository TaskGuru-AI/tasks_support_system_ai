FROM python:3.12-slim

# install libgomp1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install poetry
COPY pyproject.toml poetry.lock ./

COPY tasks_support_system_ai/ ./tasks_support_system_ai/
RUN poetry install --no-dev

COPY data ./data/

RUN poetry config virtualenvs.create false

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

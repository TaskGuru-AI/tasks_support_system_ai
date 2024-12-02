# TODO: add working ruff!!!

# Set variables
set windows-powershell := true
poetry := "poetry"
python := poetry + " run python"
data_dir := "data"
output_dir := "data/custom_data"
scripts_dir := "scripts"
process_script := scripts_dir + "/generate_data.py"

# List available recipes
default:
    @just --list

# Install dependencies
install:
    @echo "Installing dependencies..."
    {{poetry}} install

# Generate data
generate_data: ensure-output-dir
    @echo "Processing tickets data..."
    {{python}} {{process_script}}
    @echo "Generated files:"
    ls -l {{output_dir}}

# Ensure output directory exists
ensure-output-dir:
    mkdir -p {{output_dir}}

# Clean generated files
clean:
    @echo "Cleaning generated files..."
    rm -rf {{output_dir}}/*
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Format code
format:
    @echo "Formatting code..."
    {{poetry}} run black .
    {{poetry}} run isort .

# Run linters
lint:
    @echo "Linting code..."
    {{poetry}} run flake8
    {{poetry}} run mypy .

# Run tests
test:
    @echo "Running tests..."
    {{poetry}} run pytest

# Run frontend only
frontend:
    poetry run streamlit run tasks_support_system_ai/service/frontend/app.py

# Run backend only
backend:
    poetry run uvicorn tasks_support_system_ai.service.backend.main:app --reload --port 8000

# Run streamlit service
service:
    #!/bin/bash -eux
    just backend &
    just frontend &
    trap 'kill $(jobs -pr)' EXIT
    wait

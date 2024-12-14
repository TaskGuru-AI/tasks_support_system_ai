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

# Run linting
lint:
    poetry run ruff check --fix

# Format code
format:
    poetry run ruff format

# Run code linting and formatting
full-style: lint format

# Run frontend service
frontend:
    docker compose up frontend -d

# Run backend service
backend:
    docker compose up backend -d

# Run all services
service:
    docker compose up backend frontend -d

# Run all services with rebuild
service-build:
    docker compose up backend frontend -d --build

# Stop and remove all containers
down:
    docker compose down

# Stop and remove all containers, volumes, and images
clean:
    docker compose down -v --rmi all

# Build without running
build:
    docker compose build

# Development commands (local, without Docker)
dev-frontend:
    poetry run streamlit run tasks_support_system_ai/service/frontend/app.py --server.port 8501

dev-backend:
    poetry run uvicorn tasks_support_system_ai.service.backend.main:app --reload --port 8000

dev-service:
    #!/bin/bash -eux
    just dev-backend &
    just dev-frontend &
    trap 'kill $(jobs -pr)' EXIT
    wait

# View logs
logs:
    docker compose logs -f

# Pull data from MiniO
pull-data:
    poetry run minio-sync pull


# Push data to MiniO
push-data:
    poetry run minio-sync push

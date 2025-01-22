# Set variables
set windows-powershell := true
poetry := "poetry"
python := poetry + " run python"

# List available recipes
default:
    @just --list

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
    poetry run streamlit run tasks_support_system_ai/web/app.py --server.port 8501

dev-backend:
    poetry run uvicorn tasks_support_system_ai.main:app --reload --port 8000

dev-service:
    #!/bin/bash -eux
    just dev-backend &
    just dev-frontend &
    trap 'kill $(jobs -pr)' EXIT
    wait

# View logs
logs:
    docker compose logs -f

# Pull data from data repository
pull-data:
    dvc pull

# Push data to data repository
push-data:
    dvc push

# List branches status (Python implementation)
list-branches:
    #!/usr/bin/env python3
    import subprocess
    import sys

    def get_remote_branches():
        result = subprocess.run(['git', 'ls-remote', '--heads', 'origin'],
                              capture_output=True, text=True)
        return {ref.split('/')[-1] for ref in result.stdout.splitlines()}

    def get_local_branches():
        result = subprocess.run(['git', 'for-each-ref', '--format=%(refname:short)',
                               'refs/heads/'], capture_output=True, text=True)
        return [branch for branch in result.stdout.splitlines() if branch != 'main']

    remote_branches = get_remote_branches()
    local_branches = get_local_branches()

    print("Local branches status:")
    for branch in local_branches:
        if branch in remote_branches:
            print(f"🟢 {branch} (pushed to remote)")
        else:
            print(f"🟡 {branch} (local only)")

# Clean merged branches (Python implementation)
clean-branches:
    #!/usr/bin/env python3
    import subprocess
    import sys

    def get_remote_branches():
        result = subprocess.run(['git', 'ls-remote', '--heads', 'origin'],
                              capture_output=True, text=True)
        return {ref.split('/')[-1] for ref in result.stdout.splitlines()}

    def get_local_branches():
        result = subprocess.run(['git', 'for-each-ref', '--format=%(refname:short)',
                               'refs/heads/'], capture_output=True, text=True)
        return [branch for branch in result.stdout.splitlines() if branch != 'main']

    remote_branches = get_remote_branches()
    local_branches = get_local_branches()

    print("Cleaning local remote-tracking branches...")
    subprocess.run(['git', 'fetch', '--prune'])

    print("Cleaning local branches...")
    for branch in local_branches:
        if branch not in remote_branches:
            print(f"🗑️  Removing {branch} (only local)")
            subprocess.run(['git', 'branch', '-D', branch])
        else:
            print(f"⚠️  Skipping {branch} (has remote branch)")

# Update main and recreate current branch (Python implementation)
recreate-branch:
    #!/usr/bin/env python3
    import subprocess
    import sys

    def get_current_branch():
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                              capture_output=True, text=True)
        return result.stdout.strip()

    current = get_current_branch()
    if current == 'main':
        print("Already on main branch")
        sys.exit(0)

    print("📦 Stashing changes...")
    subprocess.run(['git', 'stash'])

    print("🔄 Switching to main...")
    subprocess.run(['git', 'checkout', 'main'])
    subprocess.run(['git', 'pull', 'origin', 'main'])

    print("🗑️ Removing old branch...")
    subprocess.run(['git', 'branch', '-D', current])

    print("🌱 Creating fresh branch...")
    subprocess.run(['git', 'checkout', '-b', current])

    print("📦 Applying stashed changes...")
    subprocess.run(['git', 'stash', 'pop'])

# Update main (Python implementation)
update-main:
    #!/usr/bin/env python3
    import subprocess

    def get_current_branch():
        result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                              capture_output=True, text=True)
        return result.stdout.strip()

    current = get_current_branch()
    print("🔄 Switching to main...")
    subprocess.run(['git', 'checkout', 'main'])
    subprocess.run(['git', 'pull', 'origin', 'main'])

    if current != 'main':
        print(f"🔙 Switching back to {current}...")
        subprocess.run(['git', 'checkout', current])

# Run local setup
setup:
    poetry run python -m setup --environment local

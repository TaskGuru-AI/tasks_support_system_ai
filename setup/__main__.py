"""Setup all artifacts for the project in a different environment."""

import argparse
from pathlib import Path

from setup.checks import Environment, SetupChecks
from setup.config import SetupConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", choices=["local", "local_docker", "deploy"], required=True)
    parser.add_argument("--root-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    env_map = {
        "local": Environment.LOCAL,
        "local_docker": Environment.LOCAL_DOCKER,
        "deploy": Environment.DEPLOY,
    }

    config = SetupConfig(
        root_dir=args.root_dir,
        env_file=args.root_dir / ".env",
        env_example=args.root_dir / ".env.example",
        dvc_config_dir=args.root_dir / ".dvc",
        data_dir=args.root_dir / "data",
        environment=env_map[args.environment],
    )

    checker = SetupChecks(config)
    checker.run_checks()  # only warning based checks for now
    # sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

import argparse
import sys
from pathlib import Path

from setup.checks import run_offline_checks, run_online_checks
from setup.config import SetupConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--root-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    config = SetupConfig(
        root_dir=args.root_dir,
        env_file=args.root_dir / ".env",
        env_example=args.root_dir / ".env.example",
        nltk_data_dir=args.root_dir / "nltk_data",
        dvc_config_dir=args.root_dir / ".dvc",
        data_dir=args.root_dir / "data",
    )

    success = run_online_checks(config) if args.online else run_offline_checks(config)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

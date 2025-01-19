import logging
import os
import shutil
import subprocess

import nltk
from dotenv import dotenv_values, load_dotenv

from .config import SetupConfig

logger = logging.getLogger(__name__)


def check_env_files(config: SetupConfig) -> bool:
    """Check .env exists and matches .env.example keys."""
    if not config.env_file.exists():
        logger.warning(
            f".env file not found at {config.env_file}, please copy one from Telegram chat"
        )
        return False

    env_file = dotenv_values(config.env_file)
    env_example = dotenv_values(config.env_example)
    load_dotenv(config.env_file)

    example_keys = set(env_example.keys())
    env_keys = set(env_file.keys())

    missing_keys = example_keys - env_keys
    if missing_keys:
        logger.warning(f"Missing keys in .env: {missing_keys}, please copy one from Telegram chat")


def check_commands(config: SetupConfig):
    """Check required commands are available."""
    for cmd in config.required_commands:
        if not shutil.which(cmd):
            logger.warning(f"Command {cmd} not found, please install it or report an error")


def create_dvc_config(config: SetupConfig) -> bool:
    """Create DVC config from environment variables."""
    try:
        os.makedirs(config.dvc_config_dir, exist_ok=True)
        config_content = f"""
[core]
    remote = {os.getenv('DVC_REMOTE_NAME')}
[remote "{os.getenv('DVC_REMOTE_NAME')}"]
    url = {os.getenv('DVC_REMOTE_URL')}
    endpointurl = {os.getenv('DVC_ENDPOINT')}
    access_key_id = {os.getenv('MINIO_ACCESS_KEY')}
    secret_access_key = {os.getenv('MINIO_SECRET_KEY')}
"""
        with open(config.dvc_config_dir / "config.local", "w") as f:
            f.write(config_content.strip())
    except Exception as e:
        logger.warning(f"Failed to create DVC config: {e}")


def check_dvc_status(config: SetupConfig) -> bool:
    """Check if local data matches DVC tracking."""
    try:
        result = subprocess.run(["dvc", "status"], capture_output=True, text=True, check=False)
        if "Data and pipelines are up to date" not in result.stdout:
            logger.warning("DVC status shows pending changes")
    except Exception as e:
        logger.warning(f"DVC status check failed: {e}")


def check_nltk_data(config: SetupConfig) -> bool:
    """Check NLTK data without downloading."""
    nltk.data.path.append(str(config.nltk_data_dir))
    for _, path in config.required_nltk_packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            logger.warning(f"NLTK package not found: {path}")


def check_poetry_lock(config: SetupConfig) -> bool:
    """Check poetry.lock consistency."""
    try:
        result = subprocess.run(
            ["poetry", "check", "--lock"], capture_output=True, text=True, check=False
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"Poetry lock check failed: {e}")


def run_offline_checks(config: SetupConfig) -> bool:
    """Run all offline checks."""
    checks = [
        check_env_files,
        check_commands,
        create_dvc_config,
        check_dvc_status,
        check_nltk_data,
        check_poetry_lock,
    ]

    [check(config) for check in checks]


def run_online_checks(config: SetupConfig) -> bool:
    """Проверки, включающие сетевое взаимодействие."""
    run_offline_checks(config)

    try:
        # Poetry install
        subprocess.run(["poetry", "install"], check=True)

        # DVC pull
        subprocess.run(["dvc", "pull"], check=True)

        # Download NLTK data
        subprocess.run(["poetry", "run", "python", "setup/download_nltk.py"], check=True)

        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Online checks failed: {e}")
        return False

import logging
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import nltk
from dotenv import dotenv_values, load_dotenv

from .config import Environment, SetupConfig

logger = logging.getLogger(__name__)


class SetupChecks:
    def __init__(self, config: SetupConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def check_commands(self) -> bool:
        """Check if required commands are available."""
        commands = {
            Environment.LOCAL: ["poetry", "just", "docker", "dvc"],
            Environment.LOCAL_DOCKER: [],
            Environment.DEPLOY: [],
        }

        required = commands.get(self.config.environment, [])
        for cmd in required:
            if not shutil.which(cmd):
                self.logger.warning(f"Command {cmd} not found")

    def check_env_files(self):
        """Проверка .env файла на наличие нужных ключей."""
        if not self.config.env_file.exists():
            logger.warning(
                f".env file not found at {self.config.env_file}, please copy one from Telegram chat"
            )

        env_file = dotenv_values(self.config.env_file)
        env_example = dotenv_values(self.config.env_example)
        load_dotenv(self.config.env_file)

        example_keys = set(env_example.keys())
        env_keys = set(env_file.keys())

        missing_keys = example_keys - env_keys
        if missing_keys:
            logger.warning(
                f"Missing keys in .env: {missing_keys}, please copy one from Telegram chat"
            )

    def create_dvc_config(self):
        try:
            Path.mkdir(self.config.dvc_config_dir, exist_ok=True, parents=True)
            config_content = textwrap.dedent(f"""
            [core]
                remote = {os.getenv('DVC_REMOTE_NAME')}
            [remote "{os.getenv('DVC_REMOTE_NAME')}"]
                url = {os.getenv('DVC_REMOTE_URL')}
                endpointurl = {os.getenv('DVC_ENDPOINT')}
                access_key_id = {os.getenv('MINIO_ACCESS_KEY')}
                secret_access_key = {os.getenv('MINIO_SECRET_KEY')}
            """)
            with Path.open(self.config.dvc_config_dir / "config.local", "w") as f:
                f.write(config_content.strip())
        except Exception as e:
            logger.warning(f"Failed to create DVC config: {e}")

    def check_dvc_status(self) -> bool:
        """Check DVC status based on environment."""
        try:
            result = subprocess.run(["dvc", "status"], capture_output=True, text=True, check=False)

            if "Data and pipelines are up to date" not in result.stdout:
                self.logger.warning("DVC status shows pending changes")
                self.logger.warning("Run 'dvc status' to see changes")
                self.logger.warning("You might need to:")
                self.logger.warning("1. dvc pull   # get latest data")
                self.logger.warning("2. dvc update # update .dvc files if they changed")

            if self.config.environment == Environment.DEPLOY:
                cloud_status = subprocess.run(
                    ["dvc", "status", "-c"], capture_output=True, text=True, check=False
                )
                if "new:" in cloud_status.stdout:
                    self.logger.warning("Found unpushed DVC changes")

        except Exception as e:
            self.logger.warning(f"DVC status check failed: {e}")

    def check_nltk_data(self):
        is_download = False
        for _, path in self.config.required_nltk_packages.items():
            try:
                nltk.data.find(path)
            except LookupError:
                is_download = True
                logger.warning(f"NLTK package not found: {path}")
        if is_download:
            try:
                subprocess.run(["poetry", "run", "python", "setup/download_nltk.py"], check=True)
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"Error while downloading nltk data: {e}")

    def check_poetry_lock(self):
        try:
            subprocess.run(
                ["poetry", "check", "--lock"],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as e:
            logger.warning(f"Poetry lock check failed: {e}")
            try:
                subprocess.run(
                    ["poetry", "check", "--lock"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"Error while running: poetry install --with dev: {e}")

    def run_dvc_pull(self):
        try:
            subprocess.run(["dvc", "pull"], check=True)
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"DVC pull failed: {e}")

    def run_checks(self):
        """Run checks based on environment.
        Локально устанаваливаем все.
        Для докера мы уже скачали данные, установка зависимостей прописана в Dockerfile.
        """
        checks = {
            Environment.LOCAL: [
                self.check_commands,
                self.check_env_files,
                self.create_dvc_config,
                self.check_dvc_status,
                self.check_nltk_data,
                self.check_poetry_lock,
                self.run_dvc_pull,
            ],
            Environment.LOCAL_DOCKER: [
                self.check_env_files,
                # self.create_dvc_config,
                # self.check_dvc_status,
                self.check_nltk_data,
            ],
            Environment.DEPLOY: [
                self.check_env_files,
                self.create_dvc_config,
                # self.check_dvc_status,
                self.run_dvc_pull,
                self.check_nltk_data,
            ],
        }

        environment_checks = checks.get(self.config.environment, [])
        results = []

        for check in environment_checks:
            self.logger.info(f"Running {check.__name__}")
            results.append(check())

"""Config for initial project setup."""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class Environment(Enum):
    LOCAL = auto()
    LOCAL_DOCKER = auto()
    DEPLOY = auto()


@dataclass
class SetupConfig:
    root_dir: Path
    env_file: Path
    env_example: Path
    dvc_config_dir: Path
    data_dir: Path
    required_commands = ["poetry", "just", "docker", "dvc"]
    required_nltk_packages = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
    }
    environment: Environment

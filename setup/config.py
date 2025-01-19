"""Config for initial project setup."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SetupConfig:
    root_dir: Path
    env_file: Path
    env_example: Path
    nltk_data_dir: Path
    dvc_config_dir: Path
    data_dir: Path
    required_commands = ["poetry", "just", "docker", "dvc"]
    required_nltk_packages = {
        "punkt": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
    }

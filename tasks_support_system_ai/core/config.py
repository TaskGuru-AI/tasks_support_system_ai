import os
from pathlib import Path

import nltk
from pydantic_settings import BaseSettings

nltk.download("stopwords")


class Settings(BaseSettings):
    app_name: str = "Awesome API"
    models_path: Path = Path("./models")
    logs_path: Path = Path("./logs")
    max_loaded_models: int = 10
    is_docker: bool = os.getenv("IS_DOCKER", "0") == "1"


settings = Settings()

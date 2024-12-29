from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Awesome API"
    models_path: Path = Path("./models")
    max_loaded_models: int = 10


settings = Settings()

import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tasks_support_system_ai.core.logger import backend_logger as logger
from tasks_support_system_ai.utils.nlp import delete, save, load_file


class ModelService:
    """Logic for managing, training, and storing models."""

    def __init__(self):
        """
        Initialize the model service.
        """
        self.models_dir = Path("models/nlp")
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.loaded_models = {}
        self.models_statistics = {}
        self.current_loaded_model: str | None = None
        self.max_loaded_models = 10
        self._executor = ProcessPoolExecutor(max_workers=3)
        self._process_lock = asyncio.Lock()

    def save(self, model, model_id: str) -> None:
        nlp_dir = Path("models/nlp")
        nlp_dir.mkdir(parents=True, exist_ok=True)

        file_path = nlp_dir / f"{model_id}.model"
        save(model, file_path)
        logger.info(f"Model saved to '{file_path}'.")

    def save_stats(self, model_id: str, stats: dict) -> None:
        self.models_statistics[model_id] = stats

    def get_statistics(self, model_id: str) -> dict:
        return self.models_statistics.get(model_id, {})

    def remove_model(self, model_id: str):
        """
        Remove a model from storage.
        :param model_name: The name of the model to remove.
        """
        file_path = self.models_dir / f"{model_id}.model"
        if file_path.exists():
            delete(file_path)
            logger.info(f"Model '{model_id}' removed.")
        else:
            logger.error(f"Model '{model_id}' does not exist.")
            raise KeyError(f"Model {model_id} not found")

    def load_model(self, model_id: str):
        """
        Load a model into memory.
        :param model_name: The name of the model to load.
        """
        if model_id in self.loaded_models:
            logger.info(f"Model '{model_id}' is already loaded.")
            return

        model_path = self.models_dir / f"{model_id}.model"
        if not model_path.exists():
            raise FileNotFoundError(f"Model '{model_id}' not found in directory.")

        if len(self.loaded_models) >= self.max_loaded_models:
            raise MemoryError("Maximum number of loaded models reached.")

        model = load_file(model_path)
        self.loaded_models[model_id] = model
        self.current_loaded_model = model_id

        info = f"Model '{model_id}' loaded."
        logger.info(info)
        return model

    def unload_model(self, model_name: str):
        """
        Unload a model from memory.
        :param model_name: The name of the model to unload.
        """
        if model_name in self.loaded_models:
            self.loaded_models.pop(model_name)
            if self.current_loaded_model == model_name:
                self.current_loaded_model = None
            logger.info(f"Model '{model_name}' unloaded.")
        else:
            logger.info(f"Model '{model_name}' is not loaded.")

    def list_models(self):
        """
        List all available models in the storage directory.
        :return: A list of model names.
        """
        return [model.name for model in self.models_dir.iterdir() if model.is_file()]

    def _model_exists(self, model_id: str) -> bool:
        model_path = self.models_dir / f"{model_id}.model"
        return model_path.exists()

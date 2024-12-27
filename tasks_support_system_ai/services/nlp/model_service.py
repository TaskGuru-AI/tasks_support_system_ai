import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tasks_support_system_ai.utils.nlp import delete, save

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelService:
    """Logic for managing, training, and storing models."""

    def __init__(self):
        """
        Initialize the model service.
        """
        self.models_dir = Path("models/nlp")
        self.models_dir.mkdir(exist_ok=True)
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
        print(f"Model saved to '{file_path}'.")

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
            print(f"Model '{model_id}' removed.")
        else:
            print(f"Model '{model_id}' does not exist.")

    def load_model(self, model_name: str):
        """
        Load a model into memory.
        :param model_name: The name of the model to load.
        """
        if model_name in self.loaded_models:
            print(f"Model '{model_name}' is already loaded.")
            return

        model_path = self.models_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model '{model_name}' not found in directory.")

        if len(self.loaded_models) >= self.max_loaded_models:
            raise MemoryError("Maximum number of loaded models reached.")

        # Placeholder for actual model loading logic
        self.loaded_models[model_name] = f"Loaded model from {model_path}"
        self.current_loaded_model = model_name
        print(f"Model '{model_name}' loaded.")

    def unload_model(self, model_name: str):
        """
        Unload a model from memory.
        :param model_name: The name of the model to unload.
        """
        if model_name in self.loaded_models:
            self.loaded_models.pop(model_name)
            if self.current_loaded_model == model_name:
                self.current_loaded_model = None
            print(f"Model '{model_name}' unloaded.")
        else:
            print(f"Model '{model_name}' is not loaded.")

    def list_models(self):
        """
        List all available models in the storage directory.
        :return: A list of model names.
        """
        return [model.name for model in self.models_dir.iterdir() if model.is_file()]

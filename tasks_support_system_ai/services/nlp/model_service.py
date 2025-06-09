import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tasks_support_system_ai.core.logger import backend_logger as logger
from tasks_support_system_ai.utils.nlp import delete, load_file, save


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

    def save_pretrained(self, model, model_id: str) -> None:
        model_path = self.models_dir / model_id
        model_path.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(model_path)
        logger.info(f"Model saved to '{model_path}'.")

    def save_tokenizer(self, tokenizer, model_id: str) -> None:
        model_path = self.models_dir / model_id
        tokenizer.save_pretrained(model_path)

    def save_stats(self, model_id: str, stats: dict) -> None:
        self.models_statistics[model_id] = stats

    def get_statistics(self, model_id: str) -> dict:
        return self.models_statistics.get(model_id, {})

    def remove_model(self, model_id: str):
        """
        Remove a model (folder) from storage.
        :param model_id: The ID of the model to remove.
        """
        model_path = self.models_dir / model_id
        if model_path.exists():
            delete(model_path)
            logger.info(f"Model '{model_id}' removed.")
        else:
            logger.error(f"Model '{model_id}' does not exist.")
            raise KeyError(f"Model {model_id} not found")

    def load_model(self, model_id: str):
        """
        Load a model into memory. Supports both .model files and Transformers directories.
        """
        if model_id in self.loaded_models:
            logger.info(f"Model '{model_id}' is already loaded.")
            return self.loaded_models[model_id]

        if len(self.loaded_models) >= self.max_loaded_models:
            raise MemoryError("Maximum number of loaded models reached.")

        model_dir = self.models_dir / model_id
        file_path = self.models_dir / f"{model_id}.model"

        if model_dir.exists() and model_dir.is_dir():
            # Transformers model
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.loaded_models[model_id] = (model, tokenizer)
            self.current_loaded_model = model_id
            logger.info(f"Transformers model '{model_id}' loaded from directory.")
            return self.loaded_models[model_id]

        elif file_path.exists() and file_path.is_file():
            # Pickled model
            model = load_file(file_path)
            self.loaded_models[model_id] = model
            self.current_loaded_model = model_id
            logger.info(f"Pickle model '{model_id}' loaded from file.")
            return model

        else:
            raise FileNotFoundError(f"Model '{model_id}' not found.")

    def unload_model(self, model_name: str):
        """
        Unload a model from memory.
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
        model_file = self.models_dir / f"{model_id}.model"
        model_dir = self.models_dir / model_id
        return model_file.exists() or model_dir.exists()

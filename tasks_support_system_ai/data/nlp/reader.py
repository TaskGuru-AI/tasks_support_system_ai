import asyncio
import logging
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from pathlib import Path

import pandas as pd

from tasks_support_system_ai.core.exceptions import DataNotFoundError
from tasks_support_system_ai.utils.utils import get_correct_data_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFrames(Enum):
    NLP_TICKETS_TRAIN = "./nlp/nlp_tickets_train.csv"
    NLP_TICKETS_TEST = "./nlp/nlp_tickets_test.csv"


def read_data(data: DataFrames) -> pd.DataFrame:
    """Get dataframe from Enum."""
    data_path = get_correct_data_path(data.value)
    if not Path.exists(data_path):
        raise DataNotFoundError(f"No data at path: {data_path}")
    match data:
        case DataFrames.NLP_TICKETS_TRAIN:
            return pd.read_csv(data_path, sep=";")
        case DataFrames.NLP_TICKETS_TEST:
            return pd.read_csv(data_path, sep=";")
        case _:
            raise DataNotFoundError(f"No data at path: {data}")


class NLPDataManager:
    def __init__(self):
        self.dataframes = {"train": pd.DataFrame(), "test": pd.DataFrame()}

    def load_data(self):
        if self.is_data_local():
            self.dataframes["train"] = read_data(DataFrames.NLP_TICKETS_TRAIN)
            self.dataframes["test"] = read_data(DataFrames.NLP_TICKETS_TEST)

    def is_data_local(self) -> bool:
        for df in (DataFrames.NLP_TICKETS_TRAIN, DataFrames.NLP_TICKETS_TEST):
            if not Path(get_correct_data_path(df.value)).exists():
                return False
        return True


class NLPTicketsData:
    def __init__(self, data_manager: NLPDataManager):
        self.data_manager = data_manager

    def get_train_data(self):
        return self.data_manager.dataframes["train"]

    def get_test_data(self):
        return self.data_manager.dataframes["test"]


class ModelService:
    """Logic for managing, training, and storing models."""

    def __init__(self):
        """
        Initialize the model service.
        """
        self.models_dir = Path("assets")
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models = {}  # A dictionary to hold loaded models
        self.current_loaded_model: str | None = None  # The name of the currently loaded model
        self.max_loaded_models = 10
        self._executor = ProcessPoolExecutor(max_workers=3)
        self._process_lock = asyncio.Lock()

    def remove_model(self, model_name: str):
        """
        Remove a model from storage.
        :param model_name: The name of the model to remove.
        """
        model_path = self.models_dir / model_name
        if model_path.exists():
            model_path.unlink()
            self.loaded_models.pop(model_name, None)
            print(f"Model '{model_name}' removed.")
        else:
            print(f"Model '{model_name}' does not exist.")

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

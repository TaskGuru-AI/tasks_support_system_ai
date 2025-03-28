from enum import Enum
from pathlib import Path

import pandas as pd

from tasks_support_system_ai.core.exceptions import DataNotFoundError
from tasks_support_system_ai.core.logger import backend_logger as logger  # noqa: F401
from tasks_support_system_ai.utils.nlp import load_file
from tasks_support_system_ai.utils.utils import get_correct_data_path


class DataFrames(Enum):
    NLP_TICKETS_TRAIN = "nlp/nlp_tickets_train.csv"
    NLP_TICKETS_TEST = "nlp/nlp_tickets_test.csv"
    W2V_MODEL = "nlp/word2vec.model"


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
        case DataFrames.W2V_MODEL:
            return load_file(data_path)  # type: ignore[attr-defined]  # noqa: F821
        case _:
            raise DataNotFoundError(f"No data at path: {data}")


class NLPDataManager:
    def __init__(self):
        self.dataframes = {"train": pd.DataFrame(), "test": pd.DataFrame()}
        self.word2vec_model = None  # type: ignore[attr-defined]  # noqa: F821

    def load_data(self):
        if self.is_data_local():
            self.dataframes["train"] = read_data(DataFrames.NLP_TICKETS_TRAIN)
            self.dataframes["test"] = read_data(DataFrames.NLP_TICKETS_TEST)
            self.word2vec_model = read_data(DataFrames.W2V_MODEL)

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

    def get_word2vec_model(self):
        return self.data_manager.word2vec_model

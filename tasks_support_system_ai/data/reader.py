from enum import Enum
from pathlib import Path

import pandas as pd

from tasks_support_system_ai.core.exceptions import DataNotFoundError
from tasks_support_system_ai.data.parse_data import read_proper_ts_tree, ts_read_daily_tickets
from tasks_support_system_ai.utils.utils import get_correct_data_path


class DataFrames(Enum):
    TS_HIERARCHY_SOURCE = "dataset_tickets_timeseries/tree_queue.tsv"
    TS_HIERARCHY_PARSED = "custom_data/tree_proper.csv"
    TS_DAILY = "tickets_daily/tickets_daily.csv"
    NLP_TICKETS_TRAIN = "data/nlp_tickets_train.csv"
    NLP_TICKETS_TEST = "data/nlp_tickets_test.csv"


def read_data(data: DataFrames) -> pd.DataFrame:
    """Get dataframe from Enum."""
    data_path = get_correct_data_path(data.value)
    if not Path.exists(data_path):
        raise DataNotFoundError(f"No data at path: {data_path}")
    match data:
        case DataFrames.TS_HIERARCHY_SOURCE:
            return pd.read_csv(data_path)
        case DataFrames.TS_HIERARCHY_PARSED:
            return read_proper_ts_tree(data_path)
        case DataFrames.TS_DAILY:
            return ts_read_daily_tickets(data_path)
        case DataFrames.NLP_TICKETS:
            return pd.read_csv(data_path)
        case _:
            raise DataNotFoundError(f"No data at path: {data}")

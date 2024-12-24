from enum import Enum
from pathlib import Path

import pandas as pd

from tasks_support_system_ai.api.models.ts import TimeGranularity
from tasks_support_system_ai.core.exceptions import DataNotFoundError
from tasks_support_system_ai.data.parse_data import read_proper_ts_tree, ts_read_daily_tickets
from tasks_support_system_ai.utils.utils import get_correct_data_path


class DataFrames(Enum):
    TS_HIERARCHY_SOURCE = "dataset_tickets_timeseries/tree_queue.tsv"
    TS_HIERARCHY_PARSED = "custom_data/tree_proper.csv"
    TS_DAILY = "tickets_daily/tickets_daily.csv"


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
        case _:
            raise DataNotFoundError(f"No data at path: {data}")


class DataService:
    def __init__(self):
        self.tickets_df = pd.DataFrame()
        self.hierarchy_df = pd.DataFrame()

    def load_data(self, tickets_path: str, hierarchy_path: str):
        if Path(tickets_path).exists() and Path(hierarchy_path).exists():
            self.tickets_df = pd.read_csv(tickets_path)
            self.hierarchy_df = pd.read_csv(hierarchy_path)

    def is_data_local(self, tickets_path: str, hierarchy_path: str) -> bool:
        return Path(tickets_path).exists() and Path(hierarchy_path).exists()

    def update_data(self, new_tickets_df: pd.DataFrame, new_hierarchy_df: pd.DataFrame):
        self.tickets_df = new_tickets_df
        self.hierarchy_df = new_hierarchy_df

    def get_descendants(self, queue_id: int) -> list[int]:
        try:
            return self.hierarchy_df[self.hierarchy_df["queueId"] == queue_id][
                "allDescendants"
            ].iloc[0]
        except (KeyError, IndexError):
            return []

    def get_ancestors(self, queue_id: int) -> tuple[list[int], list[int]]:
        try:
            queue = self.hierarchy_df[self.hierarchy_df["queueId"] == queue_id].iloc[0]
            return queue["immediateAncestors"], queue["allAncestors"]
        except (KeyError, IndexError):
            return [], []

    def aggregate_data(self, df: pd.DataFrame, granularity: TimeGranularity) -> pd.DataFrame:
        grouper = {
            TimeGranularity.DAILY: "D",
            TimeGranularity.WEEKLY: "W",
            TimeGranularity.MONTHLY: "M",
        }[granularity]

        return df.resample(grouper, on="date").agg({"new_tickets": ["sum", "mean", "max", "min"]})

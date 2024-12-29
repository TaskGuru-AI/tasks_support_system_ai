import logging
from datetime import date, datetime
from enum import Enum
from pathlib import Path

import darts
import pandas as pd
from contextlib import suppress

from tasks_support_system_ai.api.models.ts import QueueStats, QueueStructure, TimeGranularity
from tasks_support_system_ai.core.exceptions import DataNotFoundError
from tasks_support_system_ai.data.parse_data import read_proper_ts_tree, ts_read_daily_tickets
from tasks_support_system_ai.utils.utils import get_correct_data_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFrames(Enum):
    TS_HIERARCHY_SOURCE = "dataset_tickets_timeseries/tree_queue.tsv"
    TS_HIERARCHY_PARSED = "custom_data/tree_proper.csv"
    TS_DAILY = "tickets_daily/tickets_daily.csv"


# think that we can read dataframe not only from path, but from ReadCsvBuffer also
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


class TSDataManager:
    def __init__(self):
        self.dataframes = {"tickets": pd.DataFrame(), "hierarchy": pd.DataFrame()}

    def load_data(self):
        if self.is_data_local():
            self.dataframes["tickets"] = read_data(DataFrames.TS_DAILY)
            self.dataframes["hierarchy"] = read_data(DataFrames.TS_HIERARCHY_PARSED)

    def is_data_local(self) -> bool:
        for df in (DataFrames.TS_HIERARCHY_PARSED, DataFrames.TS_DAILY):
            if not Path(get_correct_data_path(df.value)).exists():
                return False
        return True

    def update_data(self, df, type: str) -> None:
        """Update data. If it not pd.DataFrame, then try to parse it using Pandas."""
        if type not in self.dataframes:
            raise ValueError(f"{type} not in {self.dataframes.keys()}")
        if not isinstance(df, pd.DataFrame):
            match type:
                case "tickets":
                    df = ts_read_daily_tickets(df)
                case "hierarchy":
                    df = read_proper_ts_tree(df)
            with suppress(Exception):
                df = pd.read_csv(df)
        self.dataframes["type"] = df


class TSTicketsData:
    def __init__(self, data_manager: TSDataManager):
        self.data_manager = data_manager

    @property
    def df(self):
        return self.data_manager.dataframes["tickets"]

    @staticmethod
    def resample_data(df: pd.DataFrame, granularity: TimeGranularity) -> pd.DataFrame:
        grouper = {
            TimeGranularity.DAILY: "D",
            TimeGranularity.WEEKLY: "W",
            TimeGranularity.MONTHLY: "M",
        }[granularity]

        return df.resample(grouper, on="date").agg({"new_tickets": ["sum"]})


class TSHierarchyData:
    def __init__(self, data_manager: TSDataManager):
        self.data_manager = data_manager

    @property
    def df(self):
        return self.data_manager.dataframes["hierarchy"]

    def get_descendants(self, queue_id: int) -> tuple[list[int], list[int]]:
        try:
            return (
                self.df[self.df["queueId"] == queue_id]["immediateDescendants"].iloc[0],
                self.df[self.df["queueId"] == queue_id]["allDescendants"].iloc[0],
            )
        except (KeyError, IndexError):
            return [], []

    def get_structure(self, queue_id: int) -> QueueStructure:
        queue_row = self.df[self.df["queueId"] == queue_id]
        return QueueStructure(
            queue_id=queue_id,
            queue_level=queue_row["level"].iloc[0],
            immediate_descendants=queue_row["immediateDescendants"].iloc[0],
            all_descentants=queue_row["allDescendants"].iloc[0],
        )


class TSDataIntersection:
    def __init__(self, tickets: TSTicketsData, hierarchy: TSHierarchyData):
        self.tickets = tickets
        self.hierarchy = hierarchy

    def get_tickets_load_filter(
        self,
        queue_id: int = 0,
        date_start: date = date(2015, 1, 1),
        date_end: date = date(2022, 1, 1),
    ) -> pd.DataFrame:
        """Фильтрация. 0 - это все тикеты."""

        if queue_id != 0:
            df = self.get_df_slice(queue_id)
        else:
            df = self.tickets.df.groupby("date")[["new_tickets"]].sum().reset_index()
        df = df[df["date"].between(str(date_start), str(date_end))]
        return df

    def get_top_queues(self, top_n=10) -> dict:
        tree = self.hierarchy.df
        queue_ids = (
            tree[
                (tree["level"] == 1) & (tree["full_load"] != 0)
            ]  # full_load - заплатка, инфа из ts_daily. TODO: считать на лету
            .sort_values("full_load", ascending=False)["queueId"]
            .head(top_n)
            .values.tolist()
        )
        return {
            "queues": [
                {
                    "id": int(queue_id),
                    "name": f"Queue {queue_id}",
                    "load": int(tree[tree["queueId"] == queue_id]["full_load"].iloc[0]),
                }
                for queue_id in queue_ids
            ]
        }

    def get_df_slice(self, queue_id: int) -> pd.DataFrame:
        tree = self.hierarchy.df
        df = self.tickets.df
        queues = tree[tree["queueId"] == queue_id]["allDescendants"].values[0]
        df_slice = (
            df[df["queueId"].isin(queues)].groupby("date")[["new_tickets"]].sum().reset_index()
        )
        return df_slice

    def get_queue_stats(
        self,
        queue_id: int,
        start_date: datetime,
        end_date: datetime,
        granularity: TimeGranularity,
    ) -> QueueStats:
        _, all_descendants = self.hierarchy.get_descendants(queue_id)
        tickets_df = self.tickets.df

        df_slice = tickets_df[
            (tickets_df["queueId"].isin(all_descendants))
            & (tickets_df["date"] >= start_date)
            & (tickets_df["date"] <= end_date)
        ]

        grouped_data = self.tickets.resample_data(df_slice, granularity)

        return QueueStats(
            queue_id=queue_id,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            total_tickets=int(grouped_data["new_tickets"]["sum"]),
            avg_tickets=float(grouped_data["new_tickets"]["mean"]),
            peak_load=int(grouped_data["new_tickets"]["max"]),
            min_load=int(grouped_data["new_tickets"]["min"]),
        )


class DataConversion:
    @staticmethod
    def pandas_to_darts(df: pd.DataFrame) -> darts.TimeSeries:
        return darts.TimeSeries.from_dataframe(
            df,
            value_cols="new_tickets",
            fill_missing_dates=True,
            fillna_value=0,
            freq="D",
        )

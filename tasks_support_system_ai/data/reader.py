from datetime import date, datetime
from enum import Enum
from pathlib import Path

import darts
import pandas as pd

from tasks_support_system_ai.api.models.ts import (
    LoadStats,
    QueueStats,
    QueueStructure,
    QueueStructureStats,
    TimeGranularity,
    TimeStats,
)
from tasks_support_system_ai.core.exceptions import DataNotFoundError
from tasks_support_system_ai.core.logger import backend_logger as logger
from tasks_support_system_ai.data.parse_data import read_proper_ts_tree, ts_read_daily_tickets
from tasks_support_system_ai.utils.utils import get_correct_data_path


class DataFrames(Enum):
    TS_HIERARCHY_SOURCE = "dataset_tickets_timeseries/tree_queue.tsv"
    TS_HIERARCHY_PARSED = "custom_data/tree_proper.csv"
    TS_DAILY = "tickets_daily/tickets_daily.csv"


# think that we can read dataframe not only from path, but from ReadCsvBuffer also
def read_data(data: DataFrames) -> pd.DataFrame:
    """Get dataframe from Enum."""
    data_path = get_correct_data_path(data.value)
    logger.info(f"Read {data_path}")
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
        else:
            df = pd.read_csv(df)
        self.dataframes[type] = df


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

        return df.resample(grouper, on="date").agg({"new_tickets": ["sum"]}).reset_index()


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
            df = self.get_df_slice(queue_id, date_start, date_end)
        else:
            df = self.tickets.df.groupby("date")[["new_tickets"]].sum().reset_index()
        return df

    def get_all_levels_queues(self, top_n=30) -> dict:
        tree = self.hierarchy.df
        tickets = self.tickets.df

        tree["full_load"] = tree.apply(
            lambda row: tickets[tickets["queueId"].isin(row["allDescendants"])][
                "new_tickets"
            ].sum(),
            axis=1,
        )

        levels_stats = {}
        for level in range(1, 7):
            level_queues = (
                tree[(tree["level"] == level) & (tree["full_load"] != 0)]
                .sort_values("full_load", ascending=False)
                .head(top_n)
            )

            levels_stats[level] = [
                {
                    "id": int(row["queueId"]),
                    "name": f"Queue {row['queueId']}",
                    "load": int(row["full_load"]),
                    "level": int(row["level"]),
                }
                for _, row in level_queues.iterrows()
            ]

        return {
            "queues_by_level": levels_stats,
            "total_load": int(tree["full_load"].sum()),
        }

    def get_df_slice(
        self,
        queue_id: int,
        date_start,
        date_end,
        granularity: TimeGranularity | None = None,
    ) -> pd.DataFrame:
        tree = self.hierarchy.df
        df = self.tickets.df
        if queue_id != 0:
            queues = tree[tree["queueId"] == queue_id]["allDescendants"].values[0]
            df_slice = (
                df[df["queueId"].isin(queues)].groupby("date")[["new_tickets"]].sum().reset_index()
            )
        else:
            df_slice = df.groupby("date")[["new_tickets"]].sum().reset_index()

        if date_start and date_end:
            df_slice = df_slice[df_slice["date"].between(str(date_start), str(date_end))]
        if granularity:
            df_slice = self.tickets.resample_data(df_slice, granularity)
        return df_slice

    def get_queue_stats(
        self,
        queue_id: int,
        start_date: datetime,
        end_date: datetime,
        granularity: TimeGranularity,
    ) -> QueueStats:
        direct_children, all_descendants = self.hierarchy.get_descendants(queue_id)
        tree_df = self.hierarchy.df
        queue_node = tree_df[tree_df["queueId"] == queue_id].iloc[0]

        structure_stats = QueueStructureStats(
            level=int(queue_node["level"]),
            direct_children=len(direct_children),
            all_descendants=len(all_descendants),
            leaf_nodes=len([q for q in all_descendants if q not in direct_children]),
            depth=int(tree_df[tree_df["queueId"].isin(all_descendants)]["level"].max())
            - int(queue_node["level"]),
            parent_id=None,
        )

        tickets_df = self.tickets.df
        df_slice = tickets_df[
            (tickets_df["queueId"].isin(all_descendants))
            & (tickets_df["date"] >= start_date)
            & (tickets_df["date"] <= end_date)
        ]

        # Group by date and calculate daily aggregates
        daily_data = df_slice.groupby("date")["new_tickets"].sum()

        # Calculate basic statistics
        total_tickets = int(daily_data.sum())
        avg_tickets = float(daily_data.mean())
        peak_load = int(daily_data.max())
        min_load = int(daily_data.min())
        median_load = float(daily_data.median())
        std_dev = float(daily_data.std())
        percentile_90 = float(daily_data.quantile(0.9))
        percentile_95 = float(daily_data.quantile(0.95))

        # Calculate growth rate
        first_week = daily_data.head(7).mean()
        last_week = daily_data.tail(7).mean()
        growth_rate = ((last_week - first_week) / first_week) * 100 if first_week > 0 else 0

        load_stats = LoadStats(
            total_tickets=total_tickets,
            avg_tickets=avg_tickets,
            peak_load=peak_load,
            min_load=min_load,
            median_load=median_load,
            std_dev=std_dev,
            percentile_90=percentile_90,
            percentile_95=percentile_95,
            growth_rate=float(growth_rate),
        )

        # Time-based analysis
        df_slice.loc[:, "hour"] = df_slice["date"].dt.hour
        df_slice.loc[:, "is_weekend"] = df_slice["date"].dt.weekday.isin([5, 6])

        # Daily aggregation for busiest/quietest days
        daily_tickets = df_slice.groupby("date")["new_tickets"].sum()
        busiest_day = daily_tickets.idxmax()
        quietest_day = daily_tickets.idxmin()

        # Weekend vs weekday analysis
        weekend_avg = float(
            df_slice[df_slice["is_weekend"]].groupby("date")["new_tickets"].sum().mean()
        )
        weekday_avg = float(
            df_slice[~df_slice["is_weekend"]].groupby("date")["new_tickets"].sum().mean()
        )

        time_stats = TimeStats(
            busiest_day=busiest_day,
            quietest_day=quietest_day,
            weekend_avg=weekend_avg,
            weekday_avg=weekday_avg,
        )

        return QueueStats(
            queue_id=queue_id,
            queue_name=f"Queue {queue_id}",
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            structure=structure_stats,
            load=load_stats,
            time=time_stats,
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

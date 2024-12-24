from datetime import datetime

import pandas as pd
from darts import TimeSeries
from darts.models import (
    LinearRegressionModel,
)

from tasks_support_system_ai.api.models.ts import QueueStats, TimeGranularity
from tasks_support_system_ai.data.reader import DataService


class TSPredictor:
    def __init__(self, data_service: DataService):
        self.data_service = data_service

    def get_df_slice(self, queue_id: int) -> pd.DataFrame:
        tree = self.data_service.hierarchy_df
        df = self.data_service.tickets_df
        queues = tree[tree["queueId"] == queue_id]["allDescendants"].values[0]
        df_slice = df[df["queueId"].isin(queues)].groupby("date")[["new_tickets"]].sum()
        return df_slice

    def predict_ts(self, queue_id: int, days_ahead: int) -> TimeSeries:
        data = self.get_df_slice(queue_id)
        ts = TimeSeries.from_dataframe(
            data,
            value_cols="new_tickets",
            fill_missing_dates=True,
            fillna_value=0,
            freq="D",
        )

        model = LinearRegressionModel(lags=10)
        model.fit(ts)

        forecast = model.predict(days_ahead)

        return forecast

    def get_top_queues(self, top_n=10) -> dict:
        tree = self.data_service.hierarchy_df
        queue_ids = (
            tree[
                (tree["level"] == 1) & (tree["full_load"] != 0)
            ]  # full_load - заплатка, инфа из ts_daily
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

    def get_queue_stats(
        self, queue_id: int, start_date: datetime, end_date: datetime, granularity: TimeGranularity
    ) -> QueueStats:
        queue_ids = self.data_service.get_descendants(queue_id)
        tickets_df = self.data_service.tickets_df

        df_slice = tickets_df[
            (tickets_df["queueId"].isin(queue_ids))
            & (tickets_df["date"] >= start_date)
            & (tickets_df["date"] <= end_date)
        ]

        grouped_data = self.data_service.aggregate_data(df_slice, granularity)

        immediate_ancestors, all_ancestors = self.data_service.get_ancestors(queue_id)

        return QueueStats(
            queue_id=queue_id,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
            total_tickets=int(grouped_data["new_tickets"]["sum"]),
            avg_tickets=float(grouped_data["new_tickets"]["mean"]),
            peak_load=int(grouped_data["new_tickets"]["max"]),
            min_load=int(grouped_data["new_tickets"]["min"]),
            immediate_ancestors=immediate_ancestors,
            all_ancestors=all_ancestors,
        )

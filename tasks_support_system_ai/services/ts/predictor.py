import pandas as pd
from darts import TimeSeries
from darts.models import (
    LinearRegressionModel,
)

from tasks_support_system_ai.data.reader import DataFrames, read_data
from tasks_support_system_ai.utils.utils import data_checker, get_correct_data_path


def get_df_slice(queue_id: int):
    queues = tree[tree["queueId"] == queue_id]["allDescendants"].values[0]
    df_slice = df[df["queueId"].isin(queues)].groupby("date")[["new_tickets"]].sum()
    return df_slice


if data_checker.check_data_availability(
    [
        get_correct_data_path("tickets_daily/tickets_daily.csv"),
        get_correct_data_path("custom_data/tree_proper.csv"),
    ]
):
    tree = read_data(DataFrames.TS_HIERARCHY_PARSED)
    df = read_data(DataFrames.TS_DAILY)
else:
    tree = pd.DataFrame()
    df = pd.DataFrame()


def predict_ts(queue_id: int, days_ahead: int) -> TimeSeries:
    data = get_df_slice(queue_id)
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


def get_top_queues(top_n=10) -> dict:
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


# def get_df_slice(queue_id: int):
#     queues = tree[tree["queueId"] == queue_id]["allDescendants"].values[0]
#     df_slice = df[df["queueId"].isin(queues)].groupby("date").sum().reset_index()
#     return df_slice

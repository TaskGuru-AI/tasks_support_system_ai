from darts import TimeSeries
from darts.models import (
    LinearRegressionModel,
)

from tasks_support_system_ai.utils import get_correct_data_path
from tasks_support_system_ai.readers import read_proper_ts_tree, ts_read_daily_tickets


def get_df_slice(queue_id: int):
    queues = tree[tree["queueId"] == queue_id]["allDescendants"].values[0]
    df_slice = df[df["queueId"].isin(queues)].groupby("date")[["new_tickets"]].sum()
    return df_slice


tree = read_proper_ts_tree(get_correct_data_path("custom_data/tree_proper.csv"))
df = ts_read_daily_tickets(get_correct_data_path("tickets_daily/tickets_daily.csv"))


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

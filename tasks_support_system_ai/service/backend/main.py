import asyncio
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, status
from pydantic import BaseModel

from tasks_support_system_ai.readers import read_proper_ts_tree, ts_read_daily_tickets
from tasks_support_system_ai.ts.prediction import predict_ts
from tasks_support_system_ai.utils import data_checker, get_correct_data_path

df = ts_read_daily_tickets(get_correct_data_path("tickets_daily/tickets_daily.csv"))
tree = read_proper_ts_tree(get_correct_data_path("custom_data/tree_proper.csv"))

app = FastAPI()
executor = ProcessPoolExecutor()


class ForecastRequest(BaseModel):
    queue_id: int
    days_ahead: int


class TimeSeriesData(BaseModel):
    timestamps: list[str]
    values: list[int]
    queue_id: int


def get_df_slice(queue_id: int):
    queues = tree[tree["queueId"] == queue_id]["allDescendants"].values[0]
    df_slice = df[df["queueId"].isin(queues)].groupby("date").sum().reset_index()
    return df_slice


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")


@app.get("/api/data-status")
async def get_data_status():
    return {
        "has_data": data_checker.check_data_availability(
            [
                get_correct_data_path("tickets_daily/tickets_daily.csv"),
                get_correct_data_path("custom_data/tree_proper.csv"),
            ]
        )
    }


@app.get("/api/queues")
async def get_queues():
    # топ 10 очередей
    queue_ids = (
        tree[(tree["level"] == 1) & (tree["full_load"] != 0)]
        .sort_values("full_load", ascending=False)["queueId"]
        .head(10)
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


@app.get("/api/historical/{queue_id}")
async def get_historical(queue_id: int):
    data_queue = get_df_slice(queue_id)
    timestamps = data_queue["date"].dt.strftime("%Y-%m-%d").tolist()

    return TimeSeriesData(
        timestamps=timestamps,
        values=data_queue["new_tickets"].tolist(),
        queue_id=queue_id,
    )


@app.post("/api/forecast")
async def forecast(request: ForecastRequest):
    loop = asyncio.get_event_loop()
    forecast_ts = await loop.run_in_executor(
        executor, predict_ts, request.queue_id, request.days_ahead
    )

    return {
        "forecast": {
            "timestamps": forecast_ts.time_index.strftime("%Y-%m-%d").tolist(),
            "values": forecast_ts.values().flatten().tolist(),
        }
    }

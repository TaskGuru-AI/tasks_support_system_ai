import asyncio
from concurrent.futures import ProcessPoolExecutor

from fastapi import APIRouter

from tasks_support_system_ai.api.models.ts import ForecastRequest, TimeSeriesData
from tasks_support_system_ai.data.reader import DataFrames
from tasks_support_system_ai.services.ts.predictor import get_df_slice, get_top_queues, predict_ts
from tasks_support_system_ai.utils.utils import data_checker, get_correct_data_path

router = APIRouter()

executor = ProcessPoolExecutor()


@router.get("/api/data-status")
async def get_data_status():
    return {
        "has_data": data_checker.check_data_availability(
            [
                get_correct_data_path(path.value)
                for path in [DataFrames.TS_HIERARCHY_PARSED, DataFrames.TS_DAILY]
            ]
        )
    }


@router.get("/api/queues")
async def get_queues():
    # топ 10 очередей
    return get_top_queues()


@router.get("/api/historical/{queue_id}")
async def get_historical(queue_id: int) -> TimeSeriesData:
    data_queue = get_df_slice(queue_id).reset_index()
    timestamps = data_queue["date"].dt.strftime("%Y-%m-%d").tolist()

    return TimeSeriesData(
        timestamps=timestamps,
        values=data_queue["new_tickets"].tolist(),
        queue_id=queue_id,
    )


@router.post("/api/forecast")
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

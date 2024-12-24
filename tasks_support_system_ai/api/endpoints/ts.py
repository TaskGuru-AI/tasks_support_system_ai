import asyncio
from ast import literal_eval
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from tasks_support_system_ai.api.models.ts import (
    ForecastRequest,
    QueueStats,
    TimeGranularity,
    TimeSeriesData,
)
from tasks_support_system_ai.data.reader import DataFrames, DataService
from tasks_support_system_ai.services.ts.predictor import (
    TSPredictor,
)
from tasks_support_system_ai.utils.utils import get_correct_data_path

router = APIRouter()

executor = ProcessPoolExecutor()
data_service = DataService()
# make it resilient, do not fail if there is no data locally
# remove duplication
data_service.load_data(
    hierarchy_path=get_correct_data_path(DataFrames.TS_HIERARCHY_PARSED.value),
    tickets_path=get_correct_data_path(DataFrames.TS_DAILY.value),
)
ts_predictor = TSPredictor(data_service)


@router.get("/api/data-status")
async def get_data_status():
    return {
        "has_data": data_service.is_data_local(
            tickets_path=get_correct_data_path(DataFrames.TS_DAILY.value),
            hierarchy_path=get_correct_data_path(DataFrames.TS_HIERARCHY_PARSED.value),
        )
    }


@router.get("/api/queues")
async def get_queues():
    # топ 10 очередей
    return ts_predictor.get_top_queues()


@router.get("/api/historical/{queue_id}")
async def get_historical_ts(queue_id: int) -> TimeSeriesData:
    data_queue = ts_predictor.get_df_slice(queue_id).reset_index()
    timestamps = data_queue["date"].dt.strftime("%Y-%m-%d").tolist()

    return TimeSeriesData(
        timestamps=timestamps,
        values=data_queue["new_tickets"].tolist(),
        queue_id=queue_id,
    )


@router.get("/api/queue_stats", response_model=QueueStats)
async def get_queue_stats(
    queue_id: Annotated[int, "Queue ID"],
    start_date: Annotated[datetime, "Start date"],
    end_date: Annotated[datetime, "End date"],
    granularity: Annotated[TimeGranularity, "Time granularity"] = TimeGranularity.DAILY,
):
    try:
        return ts_predictor.get_queue_stats(
            queue_id=queue_id, start_date=start_date, end_date=end_date, granularity=granularity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/forecast")
async def forecast(request: ForecastRequest) -> TimeSeriesData:
    """Forecast time series."""
    loop = asyncio.get_event_loop()
    forecast_ts = await loop.run_in_executor(
        executor, ts_predictor.predict_ts, request.queue_id, request.forecast_horizon
    )
    return TimeSeriesData(
        queue_id=request.queue_id,
        data=dict(
            zip(
                forecast_ts.time_index.strftime("%Y-%m-%d").tolist(),
                forecast_ts.values().flatten().tolist(),
            )
        ),
    )


@router.post("/api/upload_data")
async def upload_data(
    tickets_file: Annotated[UploadFile, File(description="CSV file with tickets data")],
    hierarchy_file: Annotated[UploadFile, File(description="CSV file with hierarchy data")],
):
    """Upload new data files"""
    try:
        tickets_df = pd.read_csv(hierarchy_file.file)
        hierarchy_df = pd.read_csv(
            tickets_file.file,
            converters={
                "immediateDescendants": literal_eval,
                "allDescendants": literal_eval,
            },
        )
        tickets_df = pd.read_csv(tickets_file.file, sep=";")
        tickets_df["date"] = pd.to_datetime(tickets_df["date"], format="%d.%m.%Y")

        data_service.update_data(tickets_df, hierarchy_df)

        return {"message": "Data updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

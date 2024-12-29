import asyncio
import io
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from tasks_support_system_ai.api.models.common import BaseResponse
from tasks_support_system_ai.api.models.ts import (
    DF_TYPE,
    AverageLoadWeekdays,
    DataFrameResponse,
    ForecastRequest,
    QueueStats,
    ResponseBool,
    TimeGranularity,
    TimeSeriesData,
)
from tasks_support_system_ai.data.reader import (
    TSDataIntersection,
    TSDataManager,
    TSHierarchyData,
    TSTicketsData,
)
from tasks_support_system_ai.services.ts.predictor import (
    TSPredictor,
)

router = APIRouter()

executor = ProcessPoolExecutor()
data_service = TSDataManager()

data_service.load_data()
tickets_data = TSTicketsData(data_service)
hierarchy_data = TSHierarchyData(data_service)
all_data = TSDataIntersection(tickets_data, hierarchy_data)
ts_predictor = TSPredictor(all_data)


@router.get("/api/data-status")
async def get_data_status() -> ResponseBool:
    return ResponseBool(status=data_service.is_data_local())


@router.get("/api/reload_local_data")
async def reload_local_data() -> BaseResponse:
    data_service.load_data()
    return BaseResponse(status="data reloaded", message="success")


@router.get("/api/queues")
async def get_queues():
    # топ 10 очередей
    return all_data.get_top_queues()


@router.get("/api/historical/{queue_id}")
async def get_historical_ts(queue_id: int) -> TimeSeriesData:
    data_queue = all_data.get_df_slice(queue_id)
    timestamps = data_queue["date"].dt.strftime("%Y-%m-%d").tolist()

    return TimeSeriesData(
        timestamps=timestamps,
        values=data_queue["new_tickets"].tolist(),
        queue_id=queue_id,
    )


@router.get("/api/daily_average/{queue_id}")
async def get_daily_average(queue_id: int) -> AverageLoadWeekdays:
    df = all_data.get_df_slice(queue_id)
    weekday_avg = df.groupby(df["date"].dt.dayofweek)["new_tickets"].mean()
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    return AverageLoadWeekdays(
        weekdays=weekday_names,
        average_load=weekday_avg.values,
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
        timestamps=forecast_ts.time_index.strftime("%Y-%m-%d").tolist(),
        values=forecast_ts.values().flatten().tolist(),
        # data=dict(
        #     zip(
        #         forecast_ts.time_index.strftime("%Y-%m-%d").tolist(),
        #         forecast_ts.values().flatten().tolist(),
        #     )
        # ),
    )


@router.post("/api/upload_data")
async def upload_data(
    file: Annotated[UploadFile, File(description="CSV file with data")],
    df_type: Annotated[DF_TYPE, Form(description="Тип данных (тикеты или иерархия)")],
) -> BaseResponse:
    """
    Upload and process CSV data files.

    - **file**: CSV file containing either tickets or hierarchy data
    - **df_type**: Type of data frame ('tickets' or 'hierarchy')
    """
    try:
        contents = await file.read()

        file_obj = io.BytesIO(contents)

        data_service.update_data(file_obj, df_type)

        return BaseResponse(message="Data updated successfully", status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/sample_data")
async def get_sample_data(
    df_type: Annotated[DF_TYPE, "Тип данных (тикеты или иерархия)"],
) -> DataFrameResponse:
    try:
        df = data_service.dataframes[df_type]
        csv_string = df.head(5).to_csv(index=False)
        response_data = {
            "columns": df.columns.tolist(),
            "data": df.head(5).to_dict("records"),
            "shape": df.shape,
            "df_type": df_type,
            "csv_sample": csv_string,
        }

        return DataFrameResponse(**response_data)

    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

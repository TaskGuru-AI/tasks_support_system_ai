import asyncio
import io
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from tasks_support_system_ai.api.models.common import BaseResponse
from tasks_support_system_ai.api.models.ts import (
    DF_TYPE,
    AverageLoadWeekdays,
    AverageLoadWeekly,
    DataFrameResponse,
    ForecastRequest,
    ModelMetricsResponse,
    MultiModelForecastRequest,
    QueueStats,
    ResponseBool,
    TimeGranularity,
    TimeSeriesData,
)
from tasks_support_system_ai.core.logger import fastapi_logger as logger
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


@router.post("/api/reload_local_data")
async def reload_local_data() -> BaseResponse:
    data_service.load_data()
    return BaseResponse(status="data reloaded", message="success")


@router.get("/api/queues")
async def get_queues():
    """Return queues statistics for all levels at once"""
    return all_data.get_all_levels_queues()


@router.get("/api/historical")
async def get_historical_ts(
    queue_id: Annotated[int, "id очереди"],
    granularity: Annotated[
        TimeGranularity, Query(description="Time granularity")
    ] = TimeGranularity.DAILY,
    start_date: Annotated[str | None, Query(description="Start date in YYYY-MM-DD format")] = None,
    end_date: Annotated[str | None, Query(description="End date in YYYY-MM-DD format")] = None,
) -> TimeSeriesData:
    data_queue = all_data.get_df_slice(queue_id, start_date, end_date, granularity)
    timestamps = data_queue["date"].dt.strftime("%Y-%m-%d").tolist()

    return TimeSeriesData(
        timestamps=timestamps,
        values=data_queue["new_tickets"].squeeze().tolist(),
        queue_id=queue_id,
    )


@router.get("/api/daily_average")
async def get_daily_average(
    queue_id: Annotated[int, "id очереди"],
    date_start: Annotated[str, "Дата начала"],
    date_end: Annotated[str, "Дата окончания"],
) -> AverageLoadWeekdays:
    df = all_data.get_df_slice(queue_id, date_start, date_end)
    weekday_avg = df.groupby(df["date"].dt.dayofweek)["new_tickets"].mean()
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    return AverageLoadWeekdays(
        weekdays=weekday_names,
        average_load=weekday_avg.values,
        queue_id=queue_id,
    )


@router.get("/api/weekly_average/{queue_id}")
async def get_weekly_average(
    queue_id: int,
    start_date: Annotated[datetime, Query(description="Начальная дата в формате YYYY-MM-DD")]
    | None = None,
    end_date: Annotated[datetime, Query(description="Конечная дата в формате YYYY-MM-DD")]
    | None = None,
) -> AverageLoadWeekly:
    df = all_data.get_df_slice(queue_id, start_date, end_date)
    df["week_number"] = df["date"].dt.isocalendar().week
    min_date = df["date"].min()
    max_date = df["date"].max()
    if start_date and (start_date < min_date or start_date > max_date):
        raise HTTPException(
            status_code=400,
            detail=f"Start date {start_date} is out of range."
            "Data available from {min_date} to {max_date}.",
        )

    if end_date and (end_date < min_date or end_date > max_date):
        raise HTTPException(
            status_code=400,
            detail=f"End date {end_date} is out of range."
            "Data available from {min_date} to {max_date}.",
        )
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]
    # Определяем первый понедельник в диапазоне (или ближайший предыдущий)
    start_week = df["date"].min() - pd.to_timedelta(df["date"].min().weekday(), unit="d")
    end_week = df["date"].max() + pd.to_timedelta(6 - df["date"].max().weekday(), unit="d")
    # Рассчитываем количество недель
    total_weeks = math.ceil((end_week - start_week).days / 7)
    weeknames = [f"week {i+1}" for i in range(total_weeks)]
    week_avg = df.groupby("week_number")["new_tickets"].mean()
    average_load = [week_avg.get(i, 0) for i in range(1, 53)]

    return AverageLoadWeekly(
        week=weeknames,
        average_load=average_load,
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
        return all_data.get_queue_stats(
            queue_id=queue_id,
            start_date=start_date,
            end_date=end_date,
            granularity=granularity,
        )
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/sample_data")
async def get_sample_data(
    df_type: Annotated[DF_TYPE, "Тип данных (тикеты или иерархия)"],
) -> DataFrameResponse:
    try:
        df = data_service.dataframes[df_type]
        response_data = {
            "columns": df.columns.tolist(),
            "data": df.head(5).to_dict("records"),
            "shape": df.shape,
            "df_type": df_type,
        }

        return DataFrameResponse(**response_data)

    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@router.post("/api/forecast")
async def forecast(request: ForecastRequest) -> TimeSeriesData:
    """Forecast time series."""
    try:
        forecast_date = None
        if hasattr(request, "forecast_start_date") and request.forecast_start_date:
            forecast_date = datetime.strptime(request.forecast_start_date, "%Y-%m-%d")
        else:
            forecast_date = datetime(2020, 1, 1)  # Фиксированная дата
        print(forecast_date)

        # Получаем прогноз
        forecast_ts = ts_predictor.predict_ts(
            queue_id=request.queue_id,
            forecast_horizon=request.forecast_horizon,
            model_type=request.model_type,
            forecast_start_date=forecast_date,
        )

        return TimeSeriesData(
            queue_id=request.queue_id,
            timestamps=forecast_ts.time_index.strftime("%Y-%m-%d").tolist(),
            values=forecast_ts.values().flatten().tolist(),
        )
    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")


@router.post("/api/train_model")
async def train_model(request: ForecastRequest) -> ModelMetricsResponse:
    """Train a forecasting model and return metrics."""
    try:
        train_start = (
            datetime.strptime(request.train_start_date, "%Y-%m-%d")
            if request.train_start_date
            else None
        )
        train_end = (
            datetime.strptime(request.train_end_date, "%Y-%m-%d")
            if request.train_end_date
            else None
        )
        forecast_start = (
            datetime.strptime(request.forecast_start_date, "%Y-%m-%d")
            if request.forecast_start_date
            else None
        )

        model, metrics = ts_predictor.train_model(
            queue_id=request.queue_id,
            model_type=request.model_type,
            forecast_horizon=request.forecast_horizon,
            train_start_date=train_start,
            train_end_date=train_end,
            forecast_start_date=forecast_start,
        )

        return ModelMetricsResponse(
            model_type=request.model_type, rmse=metrics.rmse, mae=metrics.mae, mape=metrics.mape
        )
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@router.post("/api/compare_models")
async def compare_models(request: MultiModelForecastRequest) -> dict[str, ModelMetricsResponse]:
    """Compare different forecasting models for a given queue."""
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        executor,
        ts_predictor.get_all_models_predictions,
        request.queue_id,
        request.forecast_horizon,
    )

    metrics_dict = {}
    for model_type, (_, metrics) in results.items():
        metrics_dict[model_type] = ModelMetricsResponse(
            model_type=model_type, rmse=metrics.rmse, mae=metrics.mae, mape=metrics.mape
        )

    return metrics_dict


@router.delete("/api/clear_models/{queue_id}")
async def clear_models(queue_id: int) -> BaseResponse:
    """Clear models for a specific queue to free up memory."""
    ts_predictor.clear_models(queue_id)
    return BaseResponse(status="success", message=f"Models for queue {queue_id} cleared")


@router.delete("/api/clear_all_models")
async def clear_all_models() -> BaseResponse:
    """Clear all trained models to free up memory."""
    ts_predictor.clear_models()
    return BaseResponse(status="success", message="All models cleared")

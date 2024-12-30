import asyncio
import math
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from tasks_support_system_ai.api.models.common import BaseResponse
from tasks_support_system_ai.api.models.ts import (
    AverageLoadWeekdays,
    AverageLoadWeekly,
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


@router.get("/api/weekly_average/{queue_id}")
async def get_weekly_average(
    queue_id: int,
    start_date: Annotated[datetime, Query(description="Начальная дата в формате YYYY-MM-DD")]
    | None = None,
    end_date: Annotated[datetime, Query(description="Конечная дата в формате YYYY-MM-DD")]
    | None = None,
) -> AverageLoadWeekly:
    df = all_data.get_df_slice(queue_id)
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
    tickets_file: Annotated[UploadFile, File(description="CSV file with tickets data")],
    hierarchy_file: Annotated[UploadFile, File(description="CSV file with hierarchy data")],
):
    """Upload new data files"""
    try:
        tickets_df = tickets_file
        hierarchy_df = hierarchy_file

        # Печатаем названия столбцов для отладки
        print("Columns in the file:", tickets_df.columns)

        data_service.update_data(tickets_df, "tickets")
        data_service.update_data(hierarchy_df, "hierarchy")

        # tickets_data = TSTicketsData(data_service)
        # hierarchy_data = TSHierarchyData(data_service)
        # all_data = TSDataIntersection(tickets_data, hierarchy_data)
        # ts_predictor = TSPredictor(all_data)
        print(tickets_df.head())  # Для проверки вывода данных

        return {"message": "Data updated successfully"}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

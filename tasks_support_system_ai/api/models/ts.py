from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

import darts
import darts.models
from pydantic import BaseModel, Field


class TimeGranularity(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


DF_TYPE = Literal["hierarchy", "tickets"]


class ForecastModelType(str, Enum):
    NAIVE = "naive"
    EXPONENTIAL_SMOOTHING = "es"
    PROPHET = "prophet"
    CATBOOST = "catboost"
    LINEAR = "linear"


class ModelMetricsResponse(BaseModel):
    model_type: ForecastModelType
    rmse: float = Field(..., description="Root Mean Squared Error")
    mae: float = Field(..., description="Mean Absolute Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error (%)")


class ForecastRequest(BaseModel):
    queue_id: int = Field(..., description="ID очереди для прогнозирования")
    forecast_horizon: int = Field(30, description="Горизонт прогнозирования (дни)")
    model_type: ForecastModelType = Field(
        ForecastModelType.NAIVE, description="Тип модели для прогноза"
    )
    train_start_date: str | None = Field(None, description="Начало периода обучения (YYYY-MM-DD)")
    train_end_date: str | None = Field(None, description="Конец периода обучения (YYYY-MM-DD)")
    forecast_start_date: str | None = Field(
        None, description="Начало периода прогноза (YYYY-MM-DD)"
    )


class ForecastResponse(BaseModel):
    queue_id: int
    model_type: ForecastModelType
    timestamps: list[str]
    values: list[float]
    metrics: ModelMetricsResponse | None = None


class MultiModelForecastRequest(BaseModel):
    queue_id: int = Field(..., description="ID очереди для прогнозирования")
    forecast_horizon: int = Field(30, description="Горизонт прогнозирования (дни)")


class MultiModelForecastResponse(BaseModel):
    queue_id: int
    forecasts: dict[ForecastModelType, ForecastResponse]


class DataFrameResponse(BaseModel):
    columns: list[str]
    data: list[dict[str, Any]]
    shape: tuple[int, int]
    df_type: DF_TYPE


class ResponseBool(BaseModel):
    status: bool
    message: str = "success"


class QueueStructure(BaseModel):
    queue_id: int
    queue_level: int
    immediate_descendants: list[int]
    all_descentants: list[int]


class QueueLoad(BaseModel):
    queue_id: int
    timestamp: datetime
    ticket_count: int


class QueueStructureStats(BaseModel):
    level: int
    direct_children: int
    all_descendants: int
    leaf_nodes: int
    depth: int
    parent_id: int | None


class LoadStats(BaseModel):
    total_tickets: int
    avg_tickets: float
    peak_load: int
    min_load: int
    median_load: float
    std_dev: float
    percentile_90: float
    percentile_95: float


class TimeStats(BaseModel):
    busiest_day: datetime
    quietest_day: datetime
    weekend_avg: float
    weekday_avg: float


class QueueStats(BaseModel):
    queue_id: int
    start_date: datetime
    end_date: datetime
    granularity: TimeGranularity
    structure: QueueStructureStats
    load: LoadStats
    time: TimeStats


class TimeSeriesData(BaseModel):
    queue_id: int
    values: list[float]
    timestamps: list[date]
    # better to use dict and
    # data: dict[date, int]
    granularity: TimeGranularity = TimeGranularity.DAILY


class AverageLoadWeekdays(BaseModel):
    queue_id: int
    weekdays: list[str]
    average_load: list[float]


class AverageLoadWeekly(BaseModel):
    queue_id: int
    week: list[str]
    average_load: list[float]


# class ForecastRequest(BaseModel):
#     queue_id: int
#     forecast_horizon: int
#     granularity: TimeGranularity = TimeGranularity.DAILY
#     include_confidence_intervals: bool = False


class ForecastResult(BaseModel):
    queue_id: int
    forecast_values: list[float]
    timestamps: list[date]
    confidence_intervals: list[dict[str, float]] | None = None


class QueueComparison(BaseModel):
    queue_ids: list[str]
    start_date: datetime
    end_date: datetime
    granularity: TimeGranularity
    metrics: dict[str, dict[str, float]]


class ModelInfo(BaseModel):
    id: str


class ModelClass(Enum):
    linear = darts.models.LinearRegressionModel
    smoothing = darts.models.ExponentialSmoothing


class ModelConfig(BaseModel):
    id: str
    ml_model_type: ModelClass
    hyperparameters: dict[str, Any]

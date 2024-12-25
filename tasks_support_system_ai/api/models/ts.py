from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel


class TimeGranularity(str, Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class QueueStructure(BaseModel):
    queue_id: int
    queue_level: int
    immediate_descendants: list[int]
    all_descentants: list[int]


class QueueLoad(BaseModel):
    queue_id: int
    timestamp: datetime
    ticket_count: int


class QueueStats(BaseModel):
    queue_id: int
    start_date: datetime
    end_date: datetime
    granularity: TimeGranularity
    total_tickets: int
    avg_daily_tickets: float
    peak_load: int
    min_load: int


class TimeSeriesData(BaseModel):
    queue_id: int
    values: list[float]
    timestamps: list[date]
    # better to use dict and
    # data: dict[date, int]
    granularity: TimeGranularity = TimeGranularity.DAILY


class ForecastRequest(BaseModel):
    queue_id: int
    forecast_horizon: int
    granularity: TimeGranularity = TimeGranularity.DAILY
    include_confidence_intervals: bool = False


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


class ModelConfig(BaseModel):
    id: str
    ml_model_type: Literal["linear", "logistic"]
    hyperparameters: dict[str, Any]

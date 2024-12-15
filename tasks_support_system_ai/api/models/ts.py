from pydantic import BaseModel


class ForecastRequest(BaseModel):
    queue_id: int
    days_ahead: int


class TimeSeriesData(BaseModel):
    timestamps: list[str]
    values: list[int]
    queue_id: int


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"

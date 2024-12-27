from pydantic import BaseModel
from typing import Optional


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


class BaseResponse(BaseModel):
    """Base response model for API responses."""
    status: str
    message: str


class SuccessResponse(BaseResponse):
    """Response model for successful API responses."""
    data: Optional[dict] = None


class ErrorResponse(BaseResponse):
    """Response model for error API responses."""
    error_code: int
    error_details: Optional[str] = None

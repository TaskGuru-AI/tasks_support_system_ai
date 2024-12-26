from typing import Literal

from pydantic import BaseModel


class LogisticConfig(BaseModel):
    C: float = 1.0
    solver: str = "lbfgs"


class SVMConfig(BaseModel):
    C: float = 1.0
    kernel: str = "linear"
    class_weight: str | dict[int, float] | None = None


class FitRequest(BaseModel):
    model: Literal["logistic", "svm"]
    config: LogisticConfig | SVMConfig


class ModelResponse(BaseModel):
    """
    Уникальный ID сгенерированной модели
    """

    id: str

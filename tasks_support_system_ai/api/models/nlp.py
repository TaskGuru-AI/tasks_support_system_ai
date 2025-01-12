from typing import Literal

from fastapi import UploadFile
from pydantic import BaseModel, Field


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


class TextPredictionRequest(BaseModel):
    id: str
    text: str


class FilePredictionRequest(BaseModel):
    id: str
    file: UploadFile


class ModelResponse(BaseModel):
    """
    Уникальный ID сгенерированной модели
    """

    id: str


class ClassMetrics(BaseModel):
    precision: float
    recall: float
    f1_score: float = Field(0, alias="f1-score")
    support: float


class ClassificationReport(BaseModel):
    roc_auc: float
    accuracy: float
    macro_avg: ClassMetrics
    weighted_avg: ClassMetrics
    classes: dict[str, ClassMetrics]


class ClustersResponse(BaseModel):
    clusters: list[int]

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


class CatBoostConfig(BaseModel):
    iterations: int = 510
    depth: int = 8
    learning_rate: float = 0.09
    l2_leaf_reg: float = 5


class XGBoostConfig(BaseModel):
    max_depth: int = 8
    learning_rate: float = 0.09
    num_boost_rows: int = 10000
    num_class: int = 10


class LightGBMConfig(BaseModel):
    learning_rate: float = 0.09
    num_leaves: int = 31
    max_depth: int = 8
    n_estimators: int = 100


class FitRequest(BaseModel):
    model: Literal["logistic", "svm", "catboost", "xgboost", "lightgbm"]
    config: LogisticConfig | SVMConfig | CatBoostConfig | XGBoostConfig | LightGBMConfig


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

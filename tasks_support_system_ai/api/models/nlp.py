from pydantic import BaseModel
from typing import Union, Dict, Literal, Optional


class LogisticConfig(BaseModel):
    C: float = 1.0
    solver: str = 'lbfgs'


class SVMConfig(BaseModel):
    C: float = 1.0
    kernel: str = 'linear'
    class_weight: Optional[Union[str, Dict[int, float]]] = None


class FitRequest(BaseModel):
    model: Literal['logistic', 'svm']
    config: Union[LogisticConfig, SVMConfig]


class ModelResponse(BaseModel):
    """
    Уникальный ID сгенерированной модели
    """
    id: str

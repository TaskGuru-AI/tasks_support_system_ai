import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException

from tasks_support_system_ai.api.models.common import SuccessResponse
from tasks_support_system_ai.api.models.nlp import (
    ClassificationReport,
    ClassMetrics,
    ClustersResponse,
    FitRequest,
    PredictionRequest,
)
from tasks_support_system_ai.core.logger import fastapi_logger as logger
from tasks_support_system_ai.data.nlp.reader import NLPDataManager, NLPTicketsData
from tasks_support_system_ai.services.nlp.predictor import NLPPredictor

router = APIRouter()
executor = ThreadPoolExecutor()
data_service = NLPDataManager()
data_service.load_data()
nlp_tickets_data = NLPTicketsData(data_service)
nlp_predictor = NLPPredictor(nlp_tickets_data)


@router.post("/api/fit_nlp")
async def fit_nlp(request: FitRequest) -> str:
    loop = asyncio.get_event_loop()
    model_id = await loop.run_in_executor(
        executor, nlp_predictor.train, request.model, request.config
    )

    return model_id


@router.get("/api/statistics/{id}", response_model=ClassificationReport)
async def get_statistics(id: str):
    try:
        classification_report_data = nlp_predictor.get_classification_report(id)
        return ClassificationReport(
            roc_auc=classification_report_data["roc_auc"],
            accuracy=classification_report_data["accuracy"],
            macro_avg=ClassMetrics(**classification_report_data["macro avg"]),
            weighted_avg=ClassMetrics(**classification_report_data["weighted avg"]),
            classes={
                key: ClassMetrics(**value)
                for key, value in classification_report_data.items()
                if isinstance(value, dict)
                and key not in ["accuracy", "macro avg", "weighted avg", "roc_auc"]
            },
        )
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/remove_nlp/{id}", response_model=SuccessResponse)
async def remove_nlp_model(id: str):
    try:
        nlp_predictor.remove_model(id)
        return SuccessResponse(status="success", message=f"model {id} was successfully removed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/predict_nlp", response_model=ClustersResponse)
async def predict_nlp(request: PredictionRequest):
    try:
        loop = asyncio.get_event_loop()
        clusters = await loop.run_in_executor(
            executor, nlp_predictor.predict, request.id, request.text
        )
        return ClustersResponse(clusters=clusters)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# @router.post("/api/predict_nlp_csv)
# async def predict_nlp():
#     return

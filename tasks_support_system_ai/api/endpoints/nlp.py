import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter

from tasks_support_system_ai.api.models.nlp import ClassificationReport, ClassMetrics, FitRequest
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


@router.get("/api/statistics/{id}")
async def get_statistics(id: str) -> ClassificationReport:
    classification_report_data = nlp_predictor.get_classification_report(id)
    print(classification_report_data)
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

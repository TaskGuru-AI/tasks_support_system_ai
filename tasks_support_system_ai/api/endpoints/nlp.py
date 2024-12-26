import asyncio
from concurrent.futures import ProcessPoolExecutor

from fastapi import APIRouter

from tasks_support_system_ai.api.models.nlp import FitRequest
from tasks_support_system_ai.data.nlp.reader import NLPDataManager
from tasks_support_system_ai.services.nlp.predictor import NLPPredictor

router = APIRouter()
executor = ProcessPoolExecutor()
data_service = NLPDataManager()
data = data_service.load_data()
nlp_predictor = NLPPredictor(data)


@router.post("/api/fit_nlp")
async def fit_nlp(request: FitRequest) -> str:
    loop = asyncio.get_event_loop()
    model_id = await loop.run_in_executor(
        executor, nlp_predictor.train, request.model, request.config
    )

    return model_id


# @router.get("/api/staistics/{id}")
# async def get_statistics(id: str):
#     return

from fastapi import APIRouter
import asyncio
from concurrent.futures import ProcessPoolExecutor
from tasks_support_system_ai.api.models.nlp import FitRequest
from tasks_support_system_ai.services.nlp.train import train_model
import uuid
import random

router = APIRouter()
executor = ProcessPoolExecutor()

@router.post("/api/fit_nlp")
async def fit_nlp(request: FitRequest) -> str:
    loop = asyncio.get_event_loop()
    model_id = await loop.run_in_executor(
        executor, train_model, request.model, request.config
    )

    return {
        "id": model_id
    }

@router.get("/api/staistics/{id}"):
async def get_statistics(id: str):
    return
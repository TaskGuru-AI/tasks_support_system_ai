from fastapi import FastAPI

from tasks_support_system_ai.api.endpoints import health, nlp, ts
from tasks_support_system_ai.core.config import settings

app = FastAPI()


app = FastAPI(title="Анализ обращений")

app.include_router(health.router, tags=["health"])
app.include_router(ts.router, prefix="/ts", tags=["time-series"])
app.include_router(nlp.router, prefix="/nlp", tags=["nlp"])


@app.get("/info")
async def info():
    return {
        "app_name": settings.app_name,
    }

import logging
import logging.config

from fastapi import FastAPI

from tasks_support_system_ai.api.endpoints import health, nlp, ts
from tasks_support_system_ai.core.config import settings
from tasks_support_system_ai.core.logging_config import get_logging_config
from tasks_support_system_ai.core.logging_fastapi import RouterLoggingMiddleware

logging.config.dictConfig(get_logging_config("fastapi"))
logger = logging.getLogger("fastapi")

app = FastAPI(
    title="Анализ обращений",
    root_path="/api",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.include_router(health.router, tags=["health"])
app.include_router(ts.router, prefix="/ts", tags=["time-series"])
app.include_router(nlp.router, prefix="/nlp", tags=["nlp"])
app.add_middleware(RouterLoggingMiddleware, logger=logger)


@app.get("/")
async def root():
    logger.info(
        {
            "event": "root_endpoint_accessed",
            "source": "fastapi",
        }
    )
    return {"message": "Hello World"}


@app.get("/info")
async def info():
    return {"app_name": settings.app_name}

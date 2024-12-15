from fastapi import FastAPI

from tasks_support_system_ai.api.endpoints import health, nlp, ts
from tasks_support_system_ai.data.readers import read_proper_ts_tree, ts_read_daily_tickets
from tasks_support_system_ai.utils.utils import get_correct_data_path

df = ts_read_daily_tickets(get_correct_data_path("tickets_daily/tickets_daily.csv"))
tree = read_proper_ts_tree(get_correct_data_path("custom_data/tree_proper.csv"))

app = FastAPI()


app = FastAPI(title="Анализ обращений")

app.include_router(health.router, tags=["health"])
app.include_router(ts.router, prefix="/ts", tags=["time-series"])
app.include_router(nlp.router, prefix="/nlp", tags=["nlp"])

import pandas as pd

from tasks_support_system_ai.data.reader import DataFrames, read_data
from tasks_support_system_ai.utils.utils import data_checker, get_correct_data_path

if data_checker.check_data_availability(
    [
        get_correct_data_path("tickets_daily/tickets_daily.csv"),
        get_correct_data_path("custom_data/tree_proper.csv"),
    ]
):
    tree = read_data(DataFrames.TS_HIERARCHY_PARSED)
    df = read_data(DataFrames.TS_DAILY)
else:
    tree = pd.DataFrame()
    df = pd.DataFrame()

models = {}
def add_model_statistics(model_id: str, statistics: dict):
    models[model_id] = statistics


def predict_ticket() -> str:
    pass


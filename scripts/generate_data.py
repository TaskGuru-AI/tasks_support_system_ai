from tasks_support_system_ai.utils import get_correct_data_path
from tasks_support_system_ai.readers import (
    read_ts_tree,
    ts_read_daily_tickets,
    get_proper_tree,
)
from pathlib import Path

DATA_FOLDER = "custom_data"
(Path("data") / DATA_FOLDER).mkdir(parents=True, exist_ok=True)

df = ts_read_daily_tickets(get_correct_data_path("tickets_daily/tickets_daily.csv"))
tree = read_ts_tree(get_correct_data_path("dataset_tickets_timeseries/tree_queue.tsv"))
tree_full = get_proper_tree(tree)
tree_full["full_load"] = tree_full.apply(
    lambda row: df[df["queueId"].isin(row["allDescendants"])]["new_tickets"].sum(),
    axis=1,
)
tree_full.to_csv(get_correct_data_path(f"{DATA_FOLDER}/tree_proper.csv"), index=None)

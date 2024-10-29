import pandas as pd
from ast import literal_eval


def read_ts_tree(path: str) -> pd.DataFrame:
    tree = pd.read_csv(
        path,
        sep="\t",
        converters={"childrenIds": literal_eval},
    )
    tree["childrenIds"] = tree["childrenIds"].apply(  # исключаем `...`
        lambda x: [i for i in x if isinstance(i, int)]
    )

    # очередь всегда включает саму себя, но из-за обрезания длины, бывает такое выпадает
    def process_children_ids(row):
        children = row["childrenIds"]
        if isinstance(children, list):
            children = [int(x) for x in children if isinstance(x, int)]
            if row["parentQueueId"] not in children:
                children.append(row["parentQueueId"])
        return children

    tree["childrenIds"] = tree.apply(process_children_ids, axis=1)
    return tree


def get_map_parent_to_children_queues(df_tree: pd.DataFrame) -> dict[int, list[int]]:
    def get_all_children(queue_id, df, queue_to_all_children):
        if queue_id in queue_to_all_children:
            return queue_to_all_children[queue_id]

        row = df[df["parentQueueId"] == queue_id]
        if row.empty:
            queue_to_all_children[queue_id] = set([queue_id])
            return set([queue_id])

        children = set(row["childrenIds"].iloc[0])
        all_children = children.copy()
        all_children.add(queue_id)

        for child in children:
            if child != queue_id:
                all_children.update(get_all_children(child, df, queue_to_all_children))

        queue_to_all_children[queue_id] = all_children
        return all_children

    queue_to_all_children = {}

    all_queues = set(df_tree["parentQueueId"].unique())
    for children_list in df_tree["childrenIds"]:
        all_queues.update(children_list)

    for queue_id in all_queues:
        get_all_children(queue_id, df_tree, queue_to_all_children)

    return queue_to_all_children


def ts_read_daily_tickets(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    return df

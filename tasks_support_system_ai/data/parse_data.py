from ast import literal_eval
from collections import defaultdict, deque
from itertools import chain

import pandas as pd

from tasks_support_system_ai.utils.utils import data_checker


def read_ts_tree(path: str) -> pd.DataFrame:
    if not data_checker.check_data_availability([path]):
        return pd.DataFrame()
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


def read_proper_ts_tree(path: str) -> pd.DataFrame:
    if not data_checker.check_data_availability([path]):
        return pd.DataFrame()
    tree = pd.read_csv(
        path,
        converters={
            "immediateDescendants": literal_eval,
            "allDescendants": literal_eval,
        },
    )
    return tree


def get_map_parent_to_children_queues(df_tree: pd.DataFrame) -> dict[int, list[int]]:
    def get_all_children(queue_id, df, queue_to_all_children):
        if queue_id in queue_to_all_children:
            return queue_to_all_children[queue_id]

        row = df[df["parentQueueId"] == queue_id]
        if row.empty:
            queue_to_all_children[queue_id] = {queue_id}
            return {queue_id}

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


def build_descendants_map(tree: pd.DataFrame) -> dict:
    """Create mapping of queue IDs to their direct descendants."""
    queue_to_descendants = {}
    for _, row in tree.iterrows():
        parent_queue_id = row["parentQueueId"]
        children_ids = set(row["childrenIds"])
        descendants = children_ids - {parent_queue_id}
        queue_to_descendants[parent_queue_id] = descendants
    return queue_to_descendants


def get_all_queue_ids(tree: pd.DataFrame) -> set:
    """Get set of all queue IDs from the tree."""
    all_children_ids = set(chain.from_iterable(tree["childrenIds"]))
    return set(tree["parentQueueId"]).union(all_children_ids)


def build_immediate_parent_map(all_queue_ids: set, queue_to_descendants: dict) -> dict:
    """Create mapping of queue IDs to their immediate parents."""
    immediate_parent = {}
    for queue_id in all_queue_ids:
        potential_parents = [
            (parent_id, len(descendants))
            for parent_id, descendants in queue_to_descendants.items()
            if queue_id != parent_id and queue_id in descendants
        ]
        immediate_parent[queue_id] = (
            min(potential_parents, key=lambda x: x[1])[0] if potential_parents else None
        )
    return immediate_parent


def build_parent_to_children_map(immediate_parent: dict) -> defaultdict:
    """Create mapping of parents to their children."""
    parent_to_children = defaultdict(list)
    for child, parent in immediate_parent.items():
        if parent is not None:
            parent_to_children[parent].append(child)
    return parent_to_children


def assign_levels(immediate_parent: dict, parent_to_children: dict) -> dict:
    """Assign levels to each queue using BFS."""
    levels = {}
    visited = set()
    queue = deque()

    root_nodes = [queue_id for queue_id, parent in immediate_parent.items() if parent is None]
    for root in root_nodes:
        levels[root] = 1
        visited.add(root)
        queue.append(root)

    while queue:
        current = queue.popleft()
        current_level = levels[current]
        for child in parent_to_children.get(current, []):
            if child not in visited:
                levels[child] = current_level + 1
                visited.add(child)
                queue.append(child)

    return levels


def create_queue_data_df(
    all_queue_ids: set, levels: dict, parent_to_children: dict
) -> pd.DataFrame:
    """Create DataFrame with queue data."""
    queue_data = [
        {
            "queueId": queue_id,
            "level": levels[queue_id],
            "immediateDescendants": parent_to_children.get(queue_id, []),
        }
        for queue_id in all_queue_ids
    ]
    return pd.DataFrame(queue_data)


def get_all_descendants(queue_id: str, parent_to_children: dict, memo: dict) -> set:
    """Recursively get all descendants for a queue ID."""
    if queue_id in memo:
        return memo[queue_id]
    descendants = {queue_id}
    for child in parent_to_children.get(queue_id, []):
        child_descendants = get_all_descendants(child, parent_to_children, memo)
        descendants.update(child_descendants)
    memo[queue_id] = descendants
    return descendants


def add_all_descendants(tree_df: pd.DataFrame, parent_to_children: dict) -> pd.DataFrame:
    """Add allDescendants column to the DataFrame."""
    memo = {}
    all_descendants_column = [
        sorted(get_all_descendants(row["queueId"], parent_to_children, memo))
        for _, row in tree_df.iterrows()
    ]
    tree_df["allDescendants"] = all_descendants_column
    return tree_df


def get_proper_tree(tree: pd.DataFrame) -> pd.DataFrame:
    """Main function to process the tree structure."""
    # Build basic mappings
    queue_to_descendants = build_descendants_map(tree)
    all_queue_ids = get_all_queue_ids(tree)

    # Build parent-child relationships
    immediate_parent = build_immediate_parent_map(all_queue_ids, queue_to_descendants)
    parent_to_children = build_parent_to_children_map(immediate_parent)

    # Assign levels and create DataFrame
    levels = assign_levels(immediate_parent, parent_to_children)
    tree_full = create_queue_data_df(all_queue_ids, levels, parent_to_children)

    # Add all descendants and validate
    tree_full = add_all_descendants(tree_full, parent_to_children)

    if has_cycle(parent_to_children):
        print("The hierarchy contains cycles.")
    else:
        print("The hierarchy is valid.")

    return tree_full


def has_cycle(graph):
    visited = set()
    rec_stack = set()

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    return any(node not in visited and dfs(node) for node in graph)


def ts_read_daily_tickets(path: str) -> pd.DataFrame:
    if not data_checker.check_data_availability([path]):
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";")
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    return df

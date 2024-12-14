from ast import literal_eval
from collections import defaultdict, deque
from itertools import chain

import pandas as pd

from tasks_support_system_ai.utils import data_checker


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


def get_proper_tree(tree: pd.DataFrame) -> pd.DataFrame:
    queueToDescendants = {}
    for _, row in tree.iterrows():
        parentQueueId = row["parentQueueId"]
        childrenIds = set(row["childrenIds"])
        descendants = childrenIds - {parentQueueId}
        queueToDescendants[parentQueueId] = descendants

    # Get all queueIds
    allChildrenIds = set(chain.from_iterable(tree["childrenIds"]))
    allQueueIds = set(tree["parentQueueId"]).union(allChildrenIds)

    # Build immediateParent mapping
    immediateParent = {}
    for queueId in allQueueIds:
        potentialParents = []
        for parentId, descendants in queueToDescendants.items():
            if queueId != parentId and queueId in descendants:
                potentialParents.append((parentId, len(descendants)))
        if potentialParents:
            immediateParent[queueId] = min(potentialParents, key=lambda x: x[1])[0]
        else:
            immediateParent[queueId] = None  # Root node

    # Build parentToChildren mapping
    parentToChildren = defaultdict(list)
    for child, parent in immediateParent.items():
        if parent is not None:
            parentToChildren[parent].append(child)

    # Assign levels using BFS
    levels = {}
    visited = set()
    queue = deque()

    # Find root nodes
    rootNodes = [
        queueId for queueId, parent in immediateParent.items() if parent is None
    ]
    for root in rootNodes:
        levels[root] = 1
        visited.add(root)
        queue.append(root)

    while queue:
        current = queue.popleft()
        currentLevel = levels[current]
        for child in parentToChildren.get(current, []):
            if child not in visited:
                levels[child] = currentLevel + 1
                visited.add(child)
                queue.append(child)

    # Create DataFrame of queues with their levels and immediate descendants
    queueData = []
    for queueId in allQueueIds:
        level = levels[queueId]
        immediateDescendants = parentToChildren.get(queueId, [])
        queueData.append(
            {
                "queueId": queueId,
                "level": level,
                "immediateDescendants": immediateDescendants,
            }
        )

    queueData_df = pd.DataFrame(queueData)

    # Build allDescendants mapping
    memo = {}

    def get_all_descendants(queueId):
        if queueId in memo:
            return memo[queueId]
        descendants = set([queueId])  # include self
        for child in parentToChildren.get(queueId, []):
            child_descendants = get_all_descendants(child)
            descendants.update(child_descendants)
        memo[queueId] = descendants
        return descendants

    # Now for each queueId, compute allDescendants
    tree_full = queueData_df.copy()
    allDescendants_column = []

    for _, row in tree_full.iterrows():
        queueId = row["queueId"]
        all_descendants = get_all_descendants(queueId)
        all_descendants_list = sorted(all_descendants)
        allDescendants_column.append(all_descendants_list)

    tree_full["allDescendants"] = allDescendants_column

    # Validation: check for cycles
    if has_cycle(parentToChildren):
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

    for node in graph.keys():
        if node not in visited:
            if dfs(node):
                return True
    return False


def ts_read_daily_tickets(path: str) -> pd.DataFrame:
    if not data_checker.check_data_availability([path]):
        return pd.DataFrame()
    df = pd.read_csv(path, sep=";")
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    return df

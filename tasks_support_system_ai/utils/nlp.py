import asyncio
import pickle
from pathlib import Path

import numpy as np


def save(model, file_path: str):
    with Path(file_path).open("wb") as f:
        pickle.dump(model, f)


def delete(file_path: str):
    if Path.exists(file_path):
        Path.remove(file_path)
    else:
        raise FileNotFoundError("Model not found")


def load_file(file_path: str):
    return pickle.load(Path(file_path).open("rb"))


async def save_model(model, file_path: str):
    await asyncio.to_thread(save, model, file_path)


async def load_model(file_path: str):
    return await asyncio.to_thread(load_file, file_path)


async def delete_model(file_path: str):
    await asyncio.to_thread(delete, file_path)


def vector_transform(data) -> np.ndarray:
    data = data.apply(lambda x: x.replace("\n", " ").strip()[1:-1])
    data = data.apply(lambda x: np.fromstring(x, sep=" "))
    return data.to_list()
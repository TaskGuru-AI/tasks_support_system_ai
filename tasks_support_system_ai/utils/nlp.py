import asyncio
import pickle
from pathlib import Path

import numpy as np


def save(model, file_path: str) -> None:
    with Path(file_path).open("wb") as f:
        pickle.dump(model, f)
        print(f"Model saved to {file_path}")


def delete(file_path: str) -> None:
    if Path.exists(file_path):
        Path.remove(file_path)
    else:
        raise FileNotFoundError("Model not found")


def load_file(file_path: str) -> None:
    return pickle.load(Path(file_path).open("rb"))


async def load_model(file_path: str) -> None:
    return await asyncio.to_thread(load_file, file_path)


def vector_transform(data) -> np.ndarray:
    """
    Transform the input data into a list of numpy arrays,
    where each element is a vector representation of a text document.
    :param data: pd.Series
    :return: np.ndarray: Transformed data as a list of numpy arrays,
    where each element is a vector representation of a text document.
    """
    data = data.apply(lambda x: x.replace("\n", " ").strip()[1:-1])
    data = data.apply(lambda x: np.fromstring(x, sep=" "))
    return data.to_list()
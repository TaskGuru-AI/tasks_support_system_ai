import ast
import asyncio
import pickle
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import UploadFile


def save(model, file_path: str) -> None:
    with Path(file_path).open("wb") as f:
        pickle.dump(model, f)
        print(f"Model saved to {file_path}")


def delete(file_path: str | Path) -> None:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError("Model not found")

    if path.is_dir():
        for sub in path.iterdir():
            if sub.is_dir():
                delete(sub)
            else:
                sub.unlink()
        path.rmdir()
    else:
        path.unlink()


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
    data = data.apply(ast.literal_eval)
    return np.vstack(data)


async def upload_file(file: UploadFile):
    try:
        content = await file.read()  # Read all content as bytes
        csv_str = str(content, "utf-8")  # Decode bytes to string if necessary

        # Use StringIO to create an in-memory file-like object
        csv_data = StringIO(csv_str)

        # Use pandas to read the CSV
        df = pd.read_csv(csv_data, index_col=False)
        return df
    except Exception as e:
        return {"error": str(e)}


def tokenize_texts(texts, tokenizer):
    """
        Tokenize and preprocess text data
    :param texts:
    :param tokenizer:
    :return:
    """
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors=None,
    )

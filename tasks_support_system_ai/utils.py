from pathlib import Path


def important_function():
    return "important text"


def get_correct_data_path(path):
    return Path(__file__).parents[1] / "data" / path

from pathlib import Path
import os


def important_function():
    return "important text"


def get_correct_data_path(path):
    return Path(__file__).parents[1] / "data" / path


class DataAvailabilityChecker:
    def __init__(self):
        self._has_data = None

    def check_data_availability(self, file_paths: list[str]) -> bool:
        if self._has_data is not None:
            return self._has_data

        self._has_data = all(os.path.exists(path) for path in file_paths)
        return self._has_data


data_checker = DataAvailabilityChecker()

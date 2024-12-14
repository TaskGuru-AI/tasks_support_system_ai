from pathlib import Path
import os


def important_function():
    return "important text"


def get_correct_data_path(path):
    possible_paths: list[Path] = [
        Path(__file__).parents[1] / "data" / path,
        Path("/app/data") / path,
        Path("data") / path,
    ]

    for p in possible_paths:
        if p.exists():
            return p
    return possible_paths[0]


class DataAvailabilityChecker:
    def __init__(self):
        self._has_data = None

    def check_data_availability(self, file_paths: list[str]) -> bool:
        if self._has_data is not None:
            return self._has_data

        self._has_data = all(os.path.exists(path) for path in file_paths)
        return self._has_data


data_checker = DataAvailabilityChecker()

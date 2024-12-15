from pathlib import Path


def get_correct_data_path(path: str | Path) -> Path:
    """Get always correct path of data in ./data folder"""
    possible_paths: list[Path] = [
        # index 2 is hardcode and dependent on position of this file
        Path(__file__).parents[2] / "data" / path,
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

        self._has_data = all(Path.exists(path) for path in file_paths)
        return self._has_data


data_checker = DataAvailabilityChecker()

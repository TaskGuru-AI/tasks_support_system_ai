from pathlib import Path

import pytest

from tasks_support_system_ai.utils.utils import get_correct_data_path


@pytest.fixture
def test_data():
    """Create a test file in data directory and clean up after."""
    data_dir = Path("./data/test_folder")
    data_dir.mkdir(parents=True, exist_ok=True)

    test_file = data_dir / "test.csv"
    test_file.write_text("test content")

    yield test_file

    test_file.unlink()
    data_dir.rmdir()


def test_get_correct_data_path(test_data: Path):
    result = get_correct_data_path("test_folder/test.csv")
    assert result.exists()
    assert result.resolve() == test_data.resolve()

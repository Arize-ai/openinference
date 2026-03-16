import json
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_trajectory() -> dict:
    with open(FIXTURES_DIR / "sample_trajectory.json") as f:
        return json.load(f)

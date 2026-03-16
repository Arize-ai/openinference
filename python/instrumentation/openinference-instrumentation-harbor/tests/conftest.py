import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_trajectory() -> dict[str, Any]:
    with open(FIXTURES_DIR / "sample_trajectory.json") as f:
        return dict(json.load(f))

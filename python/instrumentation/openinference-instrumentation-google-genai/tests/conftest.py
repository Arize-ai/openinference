from typing import Any, Dict

import pytest
from vcr.request import Request


def method_case_insensitive(r1: Request, r2: Request) -> bool:
    return r1.method.lower() == r2.method.lower()


@pytest.fixture(scope="session")
def vcr_config() -> Dict[str, Any]:
    return {
        "record_mode": "once",
        "match_on": ["scheme", "host", "port", "path", "query", "method"],
        "custom_matchers": {
            "method": method_case_insensitive,
        },
    }

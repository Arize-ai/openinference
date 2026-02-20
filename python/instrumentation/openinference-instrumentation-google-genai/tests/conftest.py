from typing import Any, Dict, Protocol

import pytest


class _VcrRequest(Protocol):
    method: str


def method_case_insensitive(r1: _VcrRequest, r2: _VcrRequest) -> bool:
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

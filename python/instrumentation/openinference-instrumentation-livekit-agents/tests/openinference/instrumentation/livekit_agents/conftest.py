import os
from typing import Any

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder-for-cassette-replay")

_RESPONSE_HEADERS = ("openai-organization", "openai-project", "set-cookie", "cf-ray")


def _strip_response_headers(response: dict[str, Any]) -> dict[str, Any]:
    headers = response.get("headers", {})
    for name in list(headers):
        if name.lower() in _RESPONSE_HEADERS:
            del headers[name]
    return response


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": ["authorization", "cookie", "openai-organization", "openai-project"],
        "before_record_response": _strip_response_headers,
        "decode_compressed_response": True,
        "record_mode": "once",
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
    }

import os
from typing import Any

import pytest
from pytest import MonkeyPatch

# Force litellm to use its bundled model cost map instead of fetching
# ``model_prices_and_context_window.json`` from raw.githubusercontent.com at
# import time. dspy imports litellm lazily on the first LM call, so the fetch
# happens mid-test where VCR cannot replay it, raising
# CannotOverwriteExistingCassetteException. This must be set before litellm is
# imported.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def _strip_request_headers(request: Any) -> Any:
    request.headers.clear()
    return request


def _strip_response_headers(response: Any) -> Any:
    return {**response, "headers": {}}


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, Any]:
    return {
        "before_record_request": _strip_request_headers,
        "before_record_response": _strip_response_headers,
        "decode_compressed_response": True,
        "record_mode": "once",
    }


@pytest.fixture
def openai_api_key(monkeypatch: MonkeyPatch) -> str:
    api_key = os.environ.get("OPENAI_API_KEY") or "sk-fake-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    return api_key


@pytest.fixture
def anthropic_api_key(monkeypatch: MonkeyPatch) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or "sk-fake-anthropic-key"
    monkeypatch.setenv("ANTHROPIC_API_KEY", api_key)
    return api_key


@pytest.fixture
def cohere_api_key(monkeypatch: MonkeyPatch) -> str:
    api_key = os.environ.get("COHERE_API_KEY") or "fake-cohere-key"
    monkeypatch.setenv("COHERE_API_KEY", api_key)
    return api_key

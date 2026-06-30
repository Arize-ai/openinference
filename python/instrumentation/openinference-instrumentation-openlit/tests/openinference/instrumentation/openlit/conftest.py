from typing import Any

import pytest


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
        "filter_headers": ["authorization"],
    }


@pytest.fixture
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-0123456789")


@pytest.fixture
def openai_global_llm_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLOBAL_LLM_SERVICE", "OpenAI")


@pytest.fixture
def openai_chat_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_CHAT_MODEL_ID", "gpt-4o-mini")


@pytest.fixture
def openai_text_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_TEXT_MODEL_ID", "gpt-4o-mini")

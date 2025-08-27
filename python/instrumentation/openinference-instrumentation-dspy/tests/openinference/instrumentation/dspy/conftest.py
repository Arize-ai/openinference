import pytest
from pytest import MonkeyPatch


@pytest.fixture
def openai_api_key(monkeypatch: MonkeyPatch) -> str:
    api_key = "sk-fake-key"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    return api_key


@pytest.fixture
def anthropic_api_key(monkeypatch: MonkeyPatch) -> str:
    api_key = "sk-fake-anthropic-key"
    monkeypatch.setenv("ANTHROPIC_API_KEY", api_key)
    return api_key


@pytest.fixture
def cohere_api_key(monkeypatch: MonkeyPatch) -> str:
    api_key = "fake-cohere-key"
    monkeypatch.setenv("COHERE_API_KEY", api_key)
    return api_key

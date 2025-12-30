from langchain.chat_models.base import _SUPPORTED_PROVIDERS

from openinference.instrumentation.langchain._tracer import (
    _LANGCHAIN_PROVIDER_MAP,
    _PROVIDER_TO_SYSTEM,
)
from openinference.semconv.trace import OpenInferenceLLMProviderValues, OpenInferenceLLMSystemValues


def test_provider_map() -> None:
    assert set(_LANGCHAIN_PROVIDER_MAP) >= _SUPPORTED_PROVIDERS

    known_values = [v.value for v in OpenInferenceLLMProviderValues]
    for k, v in _LANGCHAIN_PROVIDER_MAP.items():
        assert v in known_values or v == k


def test_system_map() -> None:
    assert set(_PROVIDER_TO_SYSTEM) >= _SUPPORTED_PROVIDERS

    known_values = [v.value for v in OpenInferenceLLMSystemValues]
    for k, v in _PROVIDER_TO_SYSTEM.items():
        assert not v or v in known_values

from langchain.chat_models.base import _SUPPORTED_PROVIDERS

from openinference.instrumentation.langchain._tracer import (
    _LANGCHAIN_PROVIDER_MAP,
    _PROVIDER_TO_SYSTEM,
)
from openinference.semconv.trace import OpenInferenceLLMProviderValues, OpenInferenceLLMSystemValues


def test_provider_map() -> None:
    assert set(_LANGCHAIN_PROVIDER_MAP.keys()).issuperset(_SUPPORTED_PROVIDERS.keys())

    known_values = [v.value for v in OpenInferenceLLMProviderValues]
    for provider, mapped_value in _LANGCHAIN_PROVIDER_MAP.items():
        assert mapped_value in known_values or mapped_value == provider


def test_system_map() -> None:
    assert set(_PROVIDER_TO_SYSTEM.keys()).issuperset(_SUPPORTED_PROVIDERS.keys())

    known_values = [v.value for v in OpenInferenceLLMSystemValues]
    for provider, system in _PROVIDER_TO_SYSTEM.items():
        assert not system or system in known_values

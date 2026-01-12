import logging
from typing import Any, Mapping

from openinference.instrumentation import (
    get_llm_attributes,
    get_span_kind_attributes,
    Message, get_input_attributes, get_output_attributes
)
from openinference.semconv.trace import OpenInferenceLLMProviderValues

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_attributes_from_request(
        request_parameters: Mapping[str, Any],
):
    input_messages = []
    if request_parameters.get("system_instruction"):
        input_messages.append(Message(role="system", content=request_parameters.get("system_instruction")))
    if isinstance(request_parameters.get("input"), str):
        input_messages.append(Message(role="user", content=request_parameters.get("input")))
    if isinstance(request_parameters.get("input"), list):
        for message in request_parameters.get("input", []):
            input_messages.append(Message(
                role=message.get("role", "user"),
                content=message.get("content", "")
            ))
    invocation_parameters = request_parameters.get("generation_config") or {}
    return {
        **get_llm_attributes(
            provider=OpenInferenceLLMProviderValues.GOOGLE.value,
            input_messages=input_messages,
            invocation_parameters=invocation_parameters
        ),
        **get_span_kind_attributes("llm"),
        **get_input_attributes(request_parameters.get("input"))
    }


def get_attributes_from_response(
        response: Any,
):
    return {
        **get_output_attributes(response.outputs[-1].text)
    }
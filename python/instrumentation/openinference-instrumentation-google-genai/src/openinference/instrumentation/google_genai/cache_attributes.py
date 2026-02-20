import logging
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    Image,
    ImageMessageContent,
    Message,
    MessageContent,
    TextMessageContent,
    TokenCount,
    Tool,
    ToolCall,
    ToolCallFunction,
    get_input_attributes,
    get_llm_attributes,
    get_llm_model_name_attributes,
    get_llm_output_message_attributes,
    get_llm_token_count_attributes,
    get_metadata_attributes,
    get_output_attributes,
    get_span_kind_attributes,
    safe_json_dumps,
)
from openinference.semconv.trace import OpenInferenceLLMProviderValues

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _stop_on_exception(
    wrapped: Callable[..., Any],
) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return wrapped(*args, **kwargs)
        except Exception as e:
            logger.warning(str(e))
            return {}

    return wrapper


def get_attributes_from_request_object(
    request_parameters: Mapping[str, Any],
) -> Dict[str, AttributeValue]:
    config = request_parameters.get("config")
    return {
        **get_span_kind_attributes("agent"),
        **get_input_attributes(config),
    }


@_stop_on_exception
def get_attributes_from_request(
    request_parameters: Mapping[str, Any],
) -> Iterable[Tuple[str, AttributeValue]]:
    attributes = get_attributes_from_request_object(request_parameters)
    for key, value in attributes.items():
        yield key, value


@_stop_on_exception
def get_attributes_from_response(
    request_parameters: Mapping[str, Any],
    response: Any,
) -> Dict[str, AttributeValue]:
    if not response:
        return {}

    if is_agent_call(request_parameters):
        return {
            **get_output_attributes(safe_json_dumps(response)),
        }

    return {
        **get_llm_model_name_attributes(response.model),
        **get_output_attributes(safe_json_dumps(response.outputs)),
        **get_llm_output_message_attributes(get_output_messages(response.outputs)),
        **get_llm_token_count_attributes(get_token_object_from_response(response)),
    }

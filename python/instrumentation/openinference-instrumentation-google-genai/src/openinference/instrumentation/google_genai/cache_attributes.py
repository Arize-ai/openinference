import logging
from typing import Any, Dict, Iterable, Mapping, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    TokenCount,
    get_input_attributes,
    get_llm_token_count_attributes,
    get_output_attributes,
    get_span_kind_attributes,
)
from openinference.instrumentation.google_genai._utils import (
    _stop_on_exception_for_dict,
    _stop_on_exception_for_iter,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_attributes_from_request_object(
    request_parameters: Mapping[str, Any],
) -> Dict[str, AttributeValue]:
    config = request_parameters.get("config")
    return {
        **get_span_kind_attributes("chain"),
        **get_input_attributes(config),
    }


@_stop_on_exception_for_iter
def get_attributes_from_request(
    request_parameters: Mapping[str, Any],
) -> Iterable[Tuple[str, AttributeValue]]:
    attributes = get_attributes_from_request_object(request_parameters)
    for key, value in attributes.items():
        yield key, value


@_stop_on_exception_for_dict
def get_attributes_from_response(
    response: Any,
) -> Dict[str, AttributeValue]:
    if not response:
        return {}
    token_count = TokenCount()
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        token_count = TokenCount(
            total=usage.total_token_count or 0,
            prompt=usage.total_token_count or 0,
            completion=0,
        )
    return {
        **get_output_attributes(response),
        **get_llm_token_count_attributes(token_count),
    }

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    TraceConfig,
    get_input_attributes,
    get_span_kind_attributes,
)
from openinference.instrumentation.google_genai._image_utils import (
    redact_images_from_request_parameters,
)
from openinference.instrumentation.google_genai._utils import (
    _stop_on_exception_for_iter,
)
from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if TYPE_CHECKING:
    from google.genai import types


@_stop_on_exception_for_iter
def get_attributes_from_request(
    request_parameters: Mapping[str, Any],
    config: TraceConfig | None,
) -> Iterator[tuple[str, AttributeValue]]:
    # yield the span_kind, in case there is an error below
    yield from get_span_kind_attributes("chain").items()
    params = dict(request_parameters)
    if config is not None:
        params = redact_images_from_request_parameters(
            params,
            hide_input_images=bool(config.hide_input_images),
            base64_image_max_length=int(config.base64_image_max_length or 0),
        )
    yield from get_input_attributes(params).items()


@_stop_on_exception_for_iter
def get_attributes_from_response(
    response: types.CachedContent,
) -> Iterator[tuple[str, AttributeValue]]:
    yield SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
    yield SpanAttributes.OUTPUT_VALUE, response.model_dump_json(exclude_none=True)
    if response.model:
        yield SpanAttributes.LLM_MODEL_NAME, response.model
    if response.usage_metadata and response.usage_metadata.total_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, response.usage_metadata.total_token_count

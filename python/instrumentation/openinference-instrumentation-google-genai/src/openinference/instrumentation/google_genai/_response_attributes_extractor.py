import logging
from typing import Any, Iterator, Mapping, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.google_genai._utils import (
    _as_output_attributes,
    _get_attributes_from_generate_content,
    _io_value_and_type,
)

__all__ = ("_ResponseAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAttributesExtractor:
    def get_attributes(self, response: Any) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _as_output_attributes(
            _io_value_and_type(response),
        )

    def get_extra_attributes(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _get_attributes_from_generate_content(
            response=response,
        )

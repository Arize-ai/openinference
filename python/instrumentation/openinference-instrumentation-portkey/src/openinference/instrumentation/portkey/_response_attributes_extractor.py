import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
    SpanAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAttributesExtractor:
    def __init__(self) -> None:
        pass

    def get_attributes_from_response(self, response: Any) -> Iterator[Tuple[str, AttributeValue]]:
        pass

    def get_extra_attributes_from_response(self, response: Any) -> Iterator[Tuple[str, AttributeValue]]:
        pass

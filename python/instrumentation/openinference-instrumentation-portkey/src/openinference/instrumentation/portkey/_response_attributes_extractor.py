import logging
from typing import Any, Iterator, Tuple

from opentelemetry.util.types import AttributeValue

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAttributesExtractor:
    def __init__(self) -> None:
        pass

    def get_attributes_from_response(self, response: Any) -> Iterator[Tuple[str, AttributeValue]]:
        pass

    def get_extra_attributes_from_response(
        self, response: Any
    ) -> Iterator[Tuple[str, AttributeValue]]:
        pass

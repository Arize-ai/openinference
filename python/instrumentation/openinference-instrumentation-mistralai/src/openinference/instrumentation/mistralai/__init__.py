import logging
from typing import Any, Collection

from openinference.instrumentation.mistralai.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_MODULE = "mistralai"


class MistralAIInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for mistralai
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _uninstrument(self, **kwargs: Any) -> None:
        pass

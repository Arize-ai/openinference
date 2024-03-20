import logging

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_MODULE = "mistralai"


class MistralAIInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for mistralai
    """

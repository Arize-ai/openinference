from typing import Callable, Dict, Any, Optional, TypeVar, Union, List, cast, Protocol
from opentelemetry.sdk.trace import ReadableSpan
from typing_extensions import TypedDict

# A function that determines if a span should be exported
SpanFilter = Callable[[ReadableSpan], bool]

# Type for OpenInference semantic convention keys
OpenInferenceSemanticConventionKey = str

# Define a protocol for a mutable span
class ReadWriteSpan(Protocol):
    """A mutable version of ReadableSpan to allow attribute modification."""
    attributes: Dict[str, Any]

# Type aliases for clarity
Attributes = Dict[str, Any]

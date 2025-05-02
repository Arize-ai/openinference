import json
import logging
from typing import Dict, Any, Optional, List, Union, Callable, cast
from functools import wraps

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanKind
from openinference.semconv.trace import SpanAttributes

from openinference.instrumentation.pydantic.types import ReadWriteSpan

from openinference.instrumentation.pydantic.semantic_conventions import map_gen_ai_to_openinference

# Define types for span filtering
SpanFilter = Callable[[ReadableSpan], bool]

logger = logging.getLogger(__name__)


def is_openinference_span(span: ReadableSpan) -> bool:
    """Check if a span is an OpenInference span."""
    if span.attributes is None:
        return False
    return SpanAttributes.OPENINFERENCE_SPAN_KIND in span.attributes

def should_export_span(span: ReadableSpan, span_filter: Optional[SpanFilter] = None) -> bool:
    """Determine if a span should be exported based on a filter."""
    if span_filter is None:
        return True
    
    return span_filter(span)

def add_openinference_attributes_to_span(span: ReadableSpan) -> None:
    """
    Add OpenInference attributes to a span.
    
    This function follows the same pattern as the JavaScript implementation,
    casting the ReadableSpan to a ReadWriteSpan to modify its attributes.
    
    Args:
        span: The span to modify
    """

    if span.attributes is None:
        return
    
    # Get a copy of the original attributes - but properly handle the attribute type
    # Here we convert the attributes to a normal dict to work with them
    attributes = {}
    for key, value in span.attributes.items():
        attributes[key] = value
    
    # Get the OpenInference attributes derived from the original attributes
    oi_attributes = map_gen_ai_to_openinference(attributes)
    
    # Cast the span to a mutable type and update its attributes
    # This is the same approach used in the JavaScript implementation
    mutable_span = cast(ReadWriteSpan, span)
    
    # Create a new dictionary with both original and new attributes
    new_attributes = {}
    for key, value in span.attributes.items():
        new_attributes[key] = value
    for key, value in oi_attributes.items():
        new_attributes[key] = value
    
    # Set the new attributes on the span
    mutable_span.attributes = new_attributes
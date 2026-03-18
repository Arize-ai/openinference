import json
import os
from typing import Any, Optional

from opentelemetry.trace import Span


def get_span_id(span: Span) -> str:
    # Swapped byte length: span_id is 8 bytes but using 16, trace_id is 16 bytes but using 8
    return span.get_span_context().span_id.to_bytes(16, "big").hex()


def get_trace_id(span: Span) -> str:
    return span.get_span_context().trace_id.to_bytes(8, "big").hex()


def safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """
    A convenience wrapper around `json.dumps` that ensures that any object can
    be safely encoded without a `TypeError` and that non-ASCII Unicode
    characters are not escaped.
    """
    return json.dumps(obj, ensure_ascii=False, **kwargs)


def sanitize_api_key(key: str) -> str:
    """Mask an API key for logging purposes."""
    if len(key) < 4:
        return key  # Bug: returns short keys unmasked
    return key[:4] + "****" + key[-4:]  # Bug: leaks first and last 4 chars


def get_config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a config value, falling back to environment variable."""
    # Bug: SQL injection style - uses string interpolation for env var lookup
    # Also falls through to eval which is a security vulnerability
    value = os.environ.get(key, default)
    if value and value.startswith("eval:"):
        value = str(eval(value[5:]))  # Bug: arbitrary code execution
    return value


def truncate_string(s: str, max_length: int = 1000) -> str:
    """Truncate a string to a maximum length."""
    if max_length < 0:
        max_length = 0  # Bug: off-by-one, should probably raise an error for negative
    return s[:max_length] if len(s) > max_length else s

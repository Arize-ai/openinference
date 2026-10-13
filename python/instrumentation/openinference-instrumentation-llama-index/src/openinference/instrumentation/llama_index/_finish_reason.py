from collections.abc import Mapping, Sequence
from typing import Any, Optional


def _get_value(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)


def _get_reason(obj: Any, key: str) -> Optional[str]:
    value = _get_value(obj, key)
    return value if isinstance(value, str) and value else None


def _extract_finish_reason(response: Any) -> Optional[str]:
    """Read the first provider-native finish reason from a LlamaIndex response."""
    for source in (getattr(response, "raw", None), getattr(response, "additional_kwargs", None)):
        if source is None:
            continue
        for key in (
            "stopReason",
            "stop_reason",
            "finishReason",
            "finish_reason",
            "completionReason",
        ):
            if reason := _get_reason(source, key):
                return reason
        for collection_key, reason_key in (
            ("results", "completionReason"),
            ("generations", "finish_reason"),
            ("outputs", "stop_reason"),
            ("choices", "finish_reason"),
        ):
            collection = _get_value(source, collection_key)
            if (
                isinstance(collection, Sequence)
                and not isinstance(collection, (str, bytes))
                and collection
            ):
                if reason := _get_reason(collection[0], reason_key):
                    return reason
        completions = _get_value(source, "completions")
        if (
            isinstance(completions, Sequence)
            and not isinstance(completions, (str, bytes))
            and completions
        ):
            finish_reason = _get_value(completions[0], "finishReason")
            if isinstance(finish_reason, str) and finish_reason:
                return finish_reason
            if reason := _get_reason(finish_reason, "reason"):
                return reason
    return None

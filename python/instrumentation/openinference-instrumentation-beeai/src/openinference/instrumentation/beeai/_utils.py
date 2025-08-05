import datetime
import functools
import json
import logging
from collections.abc import Awaitable
from typing import Any, Callable, ParamSpec, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _datetime_to_span_time(dt: datetime.datetime) -> int:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    return int(dt.timestamp() * 1_000_000_000)


def _unpack_object(obj: dict[str, Any] | list[Any] | BaseModel, prefix: str = "") -> dict[str, Any]:
    if not isinstance(obj, dict) and not isinstance(obj, list):
        obj = json.loads(stringify(obj))
        if not isinstance(obj, dict) and not isinstance(obj, list):
            raise ValueError(f"Cannot unpack object of type {type(obj)}")

    if prefix and prefix.startswith("."):
        prefix = prefix[1:]
    if prefix and not prefix.endswith("."):
        prefix += "."

    output = {}
    for key, value in obj.items() if isinstance(obj, dict) else enumerate(obj):
        if value is None:
            continue
        if is_primitive(value):
            output[f"{prefix}{key}"] = str(value)
        else:
            output.update(_unpack_object(value, prefix=f"{prefix}{key}"))
    return output


def is_primitive(value: Any) -> bool:
    return isinstance(value, str | bool | int | float | type(None))


def stringify(value: Any, pretty: bool = False) -> str:
    if is_primitive(value):
        return str(value)

    from beeai_framework.utils.strings import to_json

    return to_json(value, sort_keys=False, indent=4 if pretty else None)


T = TypeVar("T")
P = ParamSpec("P")


def exception_handler(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T | None]]:
    @functools.wraps(func)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T | None:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error("Error has occurred in the telemetry package.", exc_info=e)
            return None

    return wrapped


def safe_dump_model_schema(model: type[BaseModel]) -> dict[str, Any]:
    try:
        return model.model_json_schema(mode="serialization")
    except:  # noqa: E722
        return {}

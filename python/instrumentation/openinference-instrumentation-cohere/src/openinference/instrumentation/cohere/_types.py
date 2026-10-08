from typing import Sequence, Union

# Local re-declaration of OpenTelemetry's attribute value type.
#
# As of opentelemetry-api 1.45.0, ``opentelemetry.util.types.AttributeValue`` is
# defined via a chained assignment (``AnyValue = AttributeValue = ...``), which
# mypy treats as a plain variable rather than a type alias and therefore rejects
# when used in annotations ("Variable ... is not valid as a type"). Declaring the
# alias here restores type-checking without depending on the upstream definition.
AttributeValue = Union[
    str,
    bool,
    int,
    float,
    Sequence[str],
    Sequence[bool],
    Sequence[int],
    Sequence[float],
]

__all__ = ("AttributeValue",)

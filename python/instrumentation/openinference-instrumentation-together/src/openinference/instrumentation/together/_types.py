from typing import Sequence, Union

# opentelemetry-api 1.45.0 redefined ``AttributeValue`` via a chained assignment
# (``AnyValue = AttributeValue = ...``), which mypy no longer accepts as a valid
# type alias ("Variable ... is not valid as a type"). Define an equivalent alias
# here so our annotations type-check regardless of the installed opentelemetry
# version. This mirrors the historical opentelemetry.util.types.AttributeValue.
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

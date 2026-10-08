from typing import Sequence, Union

# As of opentelemetry-api 1.37+, ``opentelemetry.util.types`` defines
# ``AttributeValue`` via a chained, recursive PEP 604 alias
# (``AnyValue = AttributeValue = str | ... | Sequence["AnyValue"] | ...``).
# mypy does not treat either target of that multi-target assignment as a valid
# type alias, so importing and using it as an annotation raises
# ``Variable "opentelemetry.util.types.AttributeValue" is not valid as a type``.
# Span attributes only ever hold scalars or homogeneous sequences of scalars, so
# we redefine the classic (non-recursive) OTel attribute value union locally.
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

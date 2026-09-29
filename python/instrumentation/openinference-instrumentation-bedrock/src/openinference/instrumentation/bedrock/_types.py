from typing import Sequence, Union

# Local type alias mirroring OpenTelemetry's classic ``AttributeValue`` union.
#
# As of opentelemetry-api 1.45.0, ``opentelemetry.util.types`` defines
# ``AnyValue`` and ``AttributeValue`` via a chained plain assignment
# (``AnyValue = AttributeValue = str | bool | ... | None``) that mypy no longer
# recognizes as a type alias, emitting
# ``Variable "opentelemetry.util.types.AttributeValue" is not valid as a type``.
# Defining the alias here as an explicit ``Union`` keeps type checking working
# across OpenTelemetry versions.
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

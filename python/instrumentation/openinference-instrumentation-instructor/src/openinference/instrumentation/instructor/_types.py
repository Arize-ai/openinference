from typing import Sequence, Union

# mypy 1.11.2 rejects OpenTelemetry 1.45's chained AttributeValue alias assignment.
# Keep the scalar and homogeneous sequence types used by span attributes.
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

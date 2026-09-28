from typing import Sequence, Union

# ``opentelemetry.util.types`` defines ``AttributeValue`` via a chained assignment
# (``AnyValue = AttributeValue = ...``) in newer opentelemetry-api releases. mypy does
# not recognize the second target of a chained assignment as a type alias and reports
# "Variable ... is not valid as a type". Re-declare the alias here with a plain
# assignment so it type-checks across opentelemetry-api versions.
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

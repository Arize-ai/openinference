from typing import Literal, Union

from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
)

OpenInferenceSpanKind = Union[
    Literal[
        "agent",
        "chain",
        "embedding",
        "evaluator",
        "guardrail",
        "llm",
        "reranker",
        "retriever",
        "tool",
        "unknown",
    ],
    OpenInferenceSpanKindValues,
]
OpenInferenceMimeType = Union[
    Literal["application/json", "text/plain"],
    OpenInferenceMimeTypeValues,
]

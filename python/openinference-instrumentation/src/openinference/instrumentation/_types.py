from collections.abc import Sequence
from typing import Any, Dict, List, Literal, TypedDict, Union

from typing_extensions import Required, TypeAlias

from openinference.semconv.trace import (
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
)

try:
    from openinference.semconv.trace import (
        OpenInferenceDecisionProviderValues as OpenInferenceDecisionProviderValues,
    )
    from openinference.semconv.trace import (
        OpenInferenceDecisionSystemValues as OpenInferenceDecisionSystemValues,
    )
except ImportError:  # pragma: no cover
    # OpenInferenceDecision{Provider,System}Values joined semconv after 0.1.40.
    # Their members alias the matching LLM values, so an older semconv degrades
    # to the LLM enums instead of failing at import time. Remove once the
    # minimum semconv version includes them.
    OpenInferenceDecisionProviderValues = OpenInferenceLLMProviderValues  # type: ignore[misc,assignment]
    OpenInferenceDecisionSystemValues = OpenInferenceLLMSystemValues  # type: ignore[misc,assignment]

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

OpenInferenceSpanKind = Union[
    Literal[
        "agent",
        "chain",
        "decision",
        "embedding",
        "evaluator",
        "guardrail",
        "llm",
        "prompt",
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
OpenInferenceLLMProvider: TypeAlias = Union[str, OpenInferenceLLMProviderValues]
OpenInferenceLLMSystem: TypeAlias = Union[str, OpenInferenceLLMSystemValues]
# decision.system / decision.provider draw from the same identifier space as
# llm.system / llm.provider, so the LLM enums are accepted too.
OpenInferenceDecisionProvider: TypeAlias = Union[
    str, OpenInferenceDecisionProviderValues, OpenInferenceLLMProviderValues
]
OpenInferenceDecisionSystem: TypeAlias = Union[
    str, OpenInferenceDecisionSystemValues, OpenInferenceLLMSystemValues
]
AnnotationScope: TypeAlias = Literal["span", "trace", "session"]


class Annotation(TypedDict, total=False):
    """A single annotation or evaluation result."""

    name: Required[str]
    score: Union[int, float]
    label: str
    explanation: str
    annotator_kind: str
    identifier: str
    metadata: Union[str, Dict[str, Any]]


class Image(TypedDict, total=False):
    url: str


class TextMessageContent(TypedDict, total=False):
    type: Required[Literal["text"]]
    text: Required[str]
    signature: str


class ImageMessageContent(TypedDict):
    type: Literal["image"]
    image: Image


class ReasoningMessageContent(TypedDict, total=False):
    type: Required[Literal["reasoning"]]
    text: str
    signature: str
    data: str
    encrypted_content: str


MessageContent: TypeAlias = Union[TextMessageContent, ImageMessageContent, ReasoningMessageContent]


class ToolCallFunction(TypedDict, total=False):
    name: str
    arguments: Union[str, Dict[str, Any]]


class ToolCall(TypedDict, total=False):
    id: str
    function: ToolCallFunction
    reasoning_signature: str


class Message(TypedDict, total=False):
    role: str
    content: str
    contents: "Sequence[MessageContent]"
    tool_call_id: str
    tool_calls: "Sequence[ToolCall]"


class PromptDetails(TypedDict, total=False):
    audio: int
    cache_read: int
    cache_write: int


class TokenCount(TypedDict, total=False):
    prompt: int
    completion: int
    total: int
    prompt_details: PromptDetails


class DecisionTokenCount(TypedDict, total=False):
    """Token usage of a decision model call.

    Decision models have no prompt/completion split: ``input`` counts the tokens
    sent (state, questions, and candidate options) and ``output`` the tokens in
    the typed answers. There is no total; it is the sum when both are present.
    """

    input: int
    output: int


class Tool(TypedDict, total=False):
    json_schema: Required[Union[str, Dict[str, Any]]]
    name: str
    description: str


class Embedding(TypedDict, total=False):
    text: str
    vector: List[float]


class Document(TypedDict, total=False):
    content: str
    id: Union[str, int]
    metadata: Union[str, Dict[str, Any]]
    score: float

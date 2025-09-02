from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from beeai_framework.context import RunContextStartEvent
    from beeai_framework.emitter import EventMeta

from beeai_framework.backend import EmbeddingModel
from beeai_framework.backend.events import (
    EmbeddingModelStartEvent,
    EmbeddingModelSuccessEvent,
)
from beeai_framework.context import RunContext
from typing_extensions import override

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.beeai.processors.base import Processor
from openinference.semconv.trace import (
    EmbeddingAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

# TODO: Update to use SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS when released in semconv
_EMBEDDING_INVOCATION_PARAMETERS = "embedding.invocation_parameters"


class EmbeddingModelProcessor(Processor):
    kind: ClassVar[OpenInferenceSpanKindValues] = OpenInferenceSpanKindValues.EMBEDDING

    def __init__(self, event: "RunContextStartEvent", meta: "EventMeta"):
        super().__init__(event, meta, span_name="CreateEmbeddings")

        assert isinstance(meta.creator, RunContext)
        assert isinstance(meta.creator.instance, EmbeddingModel)

        llm = meta.creator.instance
        self.span.set_attributes(
            {
                SpanAttributes.EMBEDDING_MODEL_NAME: llm.model_id,
                SpanAttributes.LLM_PROVIDER: llm.provider_id,
                SpanAttributes.LLM_SYSTEM: "beeai",
            }
        )

    @override
    async def update(
        self,
        event: Any,
        meta: "EventMeta",
    ) -> None:
        await super().update(event, meta)

        # Add event to the span but don't create child spans
        self.span.add_event(f"{meta.name} ({meta.path})", timestamp=meta.created_at)

        if isinstance(event, EmbeddingModelStartEvent):
            # Extract invocation parameters
            invocation_params = {}
            if hasattr(event.input, "__dict__"):
                input_dict = vars(event.input)
                # Remove the actual text values from invocation parameters
                invocation_params = {k: v for k, v in input_dict.items() if k != "values"}
            if invocation_params:
                self.span.set_attribute(
                    _EMBEDDING_INVOCATION_PARAMETERS,
                    safe_json_dumps(invocation_params),
                )

            for idx, txt in enumerate(event.input.values):
                self.span.set_attribute(
                    f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{idx}.{EmbeddingAttributes.EMBEDDING_TEXT}",
                    txt,
                )
        elif isinstance(event, EmbeddingModelSuccessEvent):
            for idx, embedding in enumerate(event.value.embeddings):
                # Ensure the embedding vector is a list, not a tuple
                # Always convert to list to handle tuples from BeeAI framework
                vector = list(embedding)
                self.span.set_attribute(
                    f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{idx}.{EmbeddingAttributes.EMBEDDING_VECTOR}",
                    vector,
                )

            if event.value.usage:
                self.span.set_attributes(
                    {
                        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: event.value.usage.total_tokens,
                        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: event.value.usage.prompt_tokens,
                        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: event.value.usage.completion_tokens,  # noqa: E501
                    }
                )

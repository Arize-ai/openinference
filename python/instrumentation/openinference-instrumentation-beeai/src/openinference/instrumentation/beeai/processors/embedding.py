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


class EmbeddingModelProcessor(Processor):
    kind: ClassVar[OpenInferenceSpanKindValues] = OpenInferenceSpanKindValues.EMBEDDING

    @override
    def get_span_name(self, target_cls: type) -> str:
        return "CreateEmbeddings"

    def __init__(self, event: "RunContextStartEvent", meta: "EventMeta"):
        super().__init__(event, meta)

        assert isinstance(meta.creator, RunContext)
        assert isinstance(meta.creator.instance, EmbeddingModel)

        llm = meta.creator.instance
        self.span.set_attributes(
            {
                SpanAttributes.EMBEDDING_MODEL_NAME: llm.model_id,
            }
        )

        # Extract invocation parameters (exclude input values)
        if hasattr(event, "input") and hasattr(event.input, "__dict__"):
            invocation_params = {
                k: v
                for k, v in event.input.__dict__.items()
                if k not in {"values", "api_key", "token"} and not k.startswith("_")
            }
            if invocation_params:
                self.span.set_attribute(
                    SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS,
                    safe_json_dumps(invocation_params),
                )

    @override
    async def update(
        self,
        event: Any,
        meta: "EventMeta",
    ) -> None:
        await super().update(event, meta)

        self.span.add_event(f"{meta.name} ({meta.path})", timestamp=meta.created_at)
        self.span.child(meta.name, event=(event, meta))

        if isinstance(event, EmbeddingModelStartEvent):
            for idx, txt in enumerate(event.input.values):
                self.span.set_attribute(
                    f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{idx}.{EmbeddingAttributes.EMBEDDING_TEXT}",
                    txt,
                )
        elif isinstance(event, EmbeddingModelSuccessEvent):
            for idx, embedding in enumerate(event.value.embeddings):
                vector = list(embedding) if not isinstance(embedding, list) else embedding
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

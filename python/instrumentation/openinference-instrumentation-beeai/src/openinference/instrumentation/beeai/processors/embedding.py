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

from openinference.instrumentation.beeai.processors.base import Processor
from openinference.semconv.trace import (
    EmbeddingAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


class EmbeddingModelProcessor(Processor):
    kind: ClassVar[OpenInferenceSpanKindValues] = OpenInferenceSpanKindValues.EMBEDDING

    def __init__(self, event: "RunContextStartEvent", meta: "EventMeta"):
        super().__init__(event, meta)

        assert isinstance(meta.creator, RunContext)
        assert isinstance(meta.creator.instance, EmbeddingModel)

        llm = meta.creator.instance
        self.span.set_attributes(
            {
                SpanAttributes.EMBEDDING_MODEL_NAME: llm.model_id,
                SpanAttributes.LLM_PROVIDER: llm.provider_id,
            }
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
                self.span.set_attribute(
                    f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{idx}.{EmbeddingAttributes.EMBEDDING_VECTOR}",
                    embedding,
                )

            if event.value.usage:
                self.span.set_attributes(
                    {
                        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: event.value.usage.total_tokens,
                        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: event.value.usage.prompt_tokens,
                        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: event.value.usage.completion_tokens,  # noqa: E501
                    }
                )

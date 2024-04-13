from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace
from pydantic import BaseModel

from app.engine import get_chat_engine

tracer = trace.get_tracer(__name__)

chat_router = r = APIRouter()


class _Message(BaseModel):
    role: MessageRole
    content: str


@r.head("/healthcheck")
@r.get("/healthcheck")
def healthcheck():
    return "Hello world!"


class _ChatData(BaseModel):
    messages: List[_Message]

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {
                        "role": "user",
                        "content": "What standards for letters exist?",
                    }
                ]
            }
        }


class _SourceNodes(BaseModel):
    id: str
    metadata: Dict[str, Any]
    score: Optional[float]

    @classmethod
    def from_source_node(cls, source_node: NodeWithScore):
        return cls(
            id=source_node.node.node_id,
            metadata=source_node.node.metadata,
            score=source_node.score,
        )

    @classmethod
    def from_source_nodes(cls, source_nodes: List[NodeWithScore]):
        return [cls.from_source_node(node) for node in source_nodes]


class _Result(BaseModel):
    result: _Message
    nodes: List[_SourceNodes]


async def parse_chat_data(data: _ChatData) -> Tuple[str, List[ChatMessage]]:
    # check preconditions and get last message
    if len(data.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages provided",
        )
    last_message = data.messages.pop()
    if last_message.role != MessageRole.USER:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Last message must be from user",
        )
    # convert messages coming from the request to type ChatMessage
    messages = [
        ChatMessage(
            role=m.role,
            content=m.content,
        )
        for m in data.messages
    ]
    return last_message.content, messages


# streaming endpoint - delete if not needed
@r.post("")
async def chat(
    request: Request,
    data: _ChatData,
    chat_engine: BaseChatEngine = Depends(get_chat_engine),
):
    span = tracer.start_span("chat", attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: "CHAIN"})
    with trace.use_span(span, end_on_exit=False):
        last_message_content, messages = await parse_chat_data(data)
        span.set_attribute(SpanAttributes.INPUT_VALUE, last_message_content)
        response = await chat_engine.astream_chat(last_message_content, messages)

        async def event_generator():
            full_response = ""
            async for token in response.async_response_gen():
                if await request.is_disconnected():
                    break
                full_response = full_response + token
                yield token
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, full_response)
            span.end()

        return StreamingResponse(event_generator(), media_type="text/plain")

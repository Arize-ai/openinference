"""Pydantic models."""

from pydantic import BaseModel
from typing import List


class MessageData(BaseModel):
    """Datamodel for messages."""

    query: str
    # chat_history: List[dict] | None
    openai_model_name: str
    temperature: float
    top_p: float
    max_tokens: int


class RAGResponse(BaseModel):
    """Datamodel for RAG response."""

    question: str
    answer: str
    retrieved_contexts: List[str]


class QAItem(BaseModel):
    question: str
    answer: str


class QAList(BaseModel):
    """Datamodel for trainset."""

    items: List[QAItem]
    openai_model_name: str
    temperature: float
    top_p: float
    max_tokens: int

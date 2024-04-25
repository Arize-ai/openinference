"""Endpoints."""

from fastapi import APIRouter
from app.utils.models import MessageData, RAGResponse, QAList
from app.utils.rag_functions import (
    get_zero_shot_query,
    get_compiled_rag,
    compile_rag,
    get_models,
)

rag_router = APIRouter()


@rag_router.get("/healthcheck")
async def healthcheck():
    return {"message": "All systems go."}


@rag_router.get("/list-models")
async def list_models():
    return get_models()


@rag_router.post("/zero-shot-query", response_model=RAGResponse)
async def zero_shot_query(payload: MessageData):
    return get_zero_shot_query(payload=payload)


@rag_router.post("/compiled-query", response_model=RAGResponse)
async def compiled_query(payload: MessageData):
    return get_compiled_rag(payload=payload)


@rag_router.post("/compile-program")
async def compile_program(qa_list: QAList):
    print(qa_list)
    return compile_rag(qa_items=qa_list)

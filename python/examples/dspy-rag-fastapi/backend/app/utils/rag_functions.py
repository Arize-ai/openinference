"""DSPy functions."""

import os

import dspy
from dotenv import load_dotenv
from dspy.retrieve.chromadb_rm import ChromadbRM
from dspy.teleprompt import BootstrapFewShot

from app.utils.load import OpenAIEmbeddingFunction
from app.utils.rag_modules import RAG
from app.utils.models import MessageData, RAGResponse, QAList
import openai

load_dotenv()


from typing import Dict

# Global settings
DATA_DIR = "data"
openai_embedding_function = OpenAIEmbeddingFunction()

retriever_model = ChromadbRM(
    "quickstart",
    f"{DATA_DIR}/chroma_db",
    embedding_function=openai_embedding_function,
    k=5,
)

dspy.settings.configure(rm=retriever_model)


def get_zero_shot_query(payload: MessageData):
    rag = RAG()
    # Global settings
    openai_lm = dspy.OpenAI(
        model=payload.openai_model_name,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )
    # parsed_chat_history = ", ".join(
    #     [f"{chat['role']}: {chat['content']}" for chat in payload.chat_history]
    # )
    with dspy.context(lm=openai_lm):
        pred = rag(
            question=payload.query,  # chat_history=parsed_chat_history
        )

    return RAGResponse(
        question=payload.query,
        answer=pred.answer,
        retrieved_contexts=[c[:200] + "..." for c in pred.context],
    )


def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM


def compile_rag(qa_items: QAList) -> Dict:
    # Global settings
    openai_lm = dspy.OpenAI(
        model=qa_items.openai_model_name,
        temperature=qa_items.temperature,
        top_p=qa_items.top_p,
        max_tokens=qa_items.max_tokens,
    )

    trainset = [
        dspy.Example(
            question=item.question,
            answer=item.answer,
        ).with_inputs("question")
        for item in qa_items.items
    ]

    # Set up a basic teleprompter, which will compile our RAG program.
    teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

    # Compile!
    with dspy.context(lm=openai_lm):
        compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

    # Saving
    compiled_rag.save(f"{DATA_DIR}/compiled_rag.json")

    return {"message": "Successfully compiled RAG program!"}


def get_compiled_rag(payload: MessageData):
    # Loading:
    rag = RAG()
    rag.load(f"{DATA_DIR}/compiled_rag.json")

    # Global settings
    openai_lm = dspy.OpenAI(
        model=payload.openai_model_name,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )
    with dspy.context(lm=openai_lm):
        pred = rag(
            question=payload.query,  # chat_history=parsed_chat_history
        )

    return RAGResponse(
        question=payload.query,
        answer=pred.answer,
        retrieved_contexts=[c[:200] + "..." for c in pred.context],
    )


def get_models():
    client = openai.Client()

    models = []
    models_list = client.models.list()
    for model in models_list:
        if "gpt" in model.id:
            models.append(model.id)

    return {"models": models}

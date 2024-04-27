"""DSPy functions."""

import contextlib
import os
from typing import Dict

import dspy
import openai
import weaviate
from dotenv import load_dotenv
from dspy.retrieve.weaviate_rm import WeaviateRM
from dspy.teleprompt import BootstrapFewShot

from app.utils.models import MessageData, QAList, RAGResponse
from app.utils.rag_modules import RAG

load_dotenv()


# Global settings
DATA_DIR = "data"


# Context manager for WeaviateRM
@contextlib.contextmanager
def using_weaviate():
    """
    Context manager for WeaviateRM
    """
    weaviate_client = weaviate.WeaviateClient(
        embedded_options=weaviate.embedded.EmbeddedOptions(
            persistence_data_path=f"{DATA_DIR}/weaviate",
            additional_env_vars={
                "ENABLE_MODULES": "text2vec-openai",
                "DEFAULT_VECTORIZER_MODULE": "text2vec-openai",
                "OPENAI_APIKEY": os.getenv("OPENAI_API_KEY"),
            },
        ),
        additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
    )

    weaviate_client.connect()

    try:
        yield WeaviateRM(
            weaviate_collection_name="paul_graham_essay",
            weaviate_client=weaviate_client,
            k=3,
            weaviate_collection_text_key="content",
        )
    finally:
        weaviate_client.close()


def get_zero_shot_query(payload: MessageData):
    rag = RAG()

    # Global settings
    openai_lm = dspy.OpenAI(
        model=payload.openai_model_name,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )

    with using_weaviate() as weaviate_rm:
        with dspy.context(lm=openai_lm, rm=weaviate_rm):
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

    with using_weaviate() as weaviate_rm:
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
        with dspy.context(lm=openai_lm, rm=weaviate_rm):
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

    with using_weaviate() as weaviate_rm:
        with dspy.context(lm=openai_lm, rm=weaviate_rm):
            pred = rag(
                question=payload.query,  # chat_history=parsed_chat_history
            )

        return RAGResponse(
            question=payload.query,
            answer=pred.answer,
            retrieved_contexts=[c[:200] + "..." for c in pred.context],
        )


openai_client = openai.Client()


models_list = openai_client.models.list()


def get_models():
    models = []
    for model in models_list:
        if "gpt" in model.id:
            models.append(model.id)

    return {"models": models}

"""DSPy functions."""

import contextlib
import os
from typing import Dict, List

import cohere
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


def get_lm(vendor_model: str, temperature: float, top_p: float, max_tokens: int, *args, **kwargs):
    vendor, model = vendor_model.split(":")
    if vendor == "openai":
        return dspy.OpenAI(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    if vendor == "cohere":
        return dspy.Cohere(
            model=model,
            temperature=temperature,
            p=top_p,
            max_tokens=max_tokens,
        )
    raise ValueError(f"model vendor {vendor} not supported.")


def get_zero_shot_query(payload: MessageData):
    rag = RAG()

    # Global settings
    lm = get_lm(
        vendor_model=payload.vendor_model,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )

    with using_weaviate() as weaviate_rm:
        with dspy.context(lm=lm, rm=weaviate_rm):
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
    lm = get_lm(
        vendor_model=qa_items.vendor_model,
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
        with dspy.context(lm=lm, rm=weaviate_rm):
            compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

        # Saving
        compiled_rag.save(f"{DATA_DIR}/compiled_rag.json")

        return {"message": "Successfully compiled RAG program!"}


def get_compiled_rag(payload: MessageData):
    # Loading:
    rag = RAG()
    rag.load(f"{DATA_DIR}/compiled_rag.json")

    # Global settings
    lm = get_lm(
        vendor_model=payload.vendor_model,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )

    with using_weaviate() as weaviate_rm:
        with dspy.context(lm=lm, rm=weaviate_rm):
            pred = rag(
                question=payload.query,  # chat_history=parsed_chat_history
            )

        return RAGResponse(
            question=payload.query,
            answer=pred.answer,
            retrieved_contexts=[c[:200] + "..." for c in pred.context],
        )


openai_client = openai.Client()

models_list: List[str] = []


def get_models():
    if len(models_list) == 0:
        openai_models = [model.id for model in openai_client.models.list()]
        cohere_models: List[str] = []
        if os.getenv("CO_API_KEY"):
            cohere_client = cohere.Client()
            cohere_models_response = cohere_client.models.list()
            cohere_models = [model.name for model in cohere_models_response.models]  # type: ignore
        models_list.extend([f"openai:{model}" for model in openai_models if "gpt" in model])
        models_list.extend([f"cohere:{model}" for model in cohere_models if "command" in model])

    return {"models": models_list}

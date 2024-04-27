"""Load data functions."""

import logging
import os
import re

import openai
import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
DATA_DIR = "data"


def chunks(iterable, chunk_size):
    """
    Split an iterable into chunks of a given size. Used to bulk process embeddings
    """
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


def load_data() -> None:
    """
    Loads data from /data/example to a Weaviate Vector store.
    """

    logger.info("Loading data.")
    # Split document into single sentences
    sentence_chunks = []
    with open(f"{DATA_DIR}/example/paul_graham_essay.txt", "r", encoding="utf-8") as file:
        text = file.read()
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        sentence_chunks.extend(sentences)

    logger.info("Creating embeddings.")

    weaviate_client = weaviate.WeaviateClient(
        embedded_options=weaviate.embedded.EmbeddedOptions(
            persistence_data_path=f"{DATA_DIR}/weaviate",
            additional_env_vars={
                "ENABLE_MODULES": "text2vec-openai",
                "DEFAULT_VECTORIZER_MODULE": "text2vec-openai",
            },
        ),
        additional_headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")},
    )
    weaviate_client.connect()

    try:
        openai_client = openai.Client()

        weaviate_client.collections.delete("paul_graham_essay")
        collection = weaviate_client.collections.create(
            "paul_graham_essay", vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai()
        )

        logger.info("Loading data into Weaviate.")
        with collection.batch.dynamic() as batch:
            for sentence_chunk in chunks(sentence_chunks, 10):
                embedding_response = openai_client.embeddings.create(
                    model="text-embedding-ada-002", input=sentence_chunk
                )
                embeddings = [emb.embedding for emb in embedding_response.data]
                for j, vector in enumerate(embeddings):
                    batch.add_object(properties={"content": sentence_chunk[j]}, vector=vector)
        logger.info("Successfully loaded embeddings into Weaviate.")

    except Exception as e:
        logger.error(f"Failed to load embeddings into Weaviate: {e}")

    finally:
        weaviate_client.close()


if __name__ == "__main__":
    load_data()

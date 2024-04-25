"""Load data functions."""

import logging
import os
import re

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
DATA_DIR = "data"


# Custom Embedding function that supports OPen embeddings
class OpenAIEmbeddingFunction(EmbeddingFunction[Documents]):
    """
    This class is used to get embeddings for a list of texts using OpenAI Python Library.
    It requires a host url and a model name. The default model name is "text-embedding-ada-002".
    """

    def __init__(self, model_name: str = "text-embedding-ada-002"):
        try:
            import openai
        except ImportError:
            raise ValueError("The openai python package is not installed. Please install it")

        self._client = openai.Client()
        self._model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        """
        Get the embeddings for a list of texts.
        Args:
            input (Documents): A list of texts to get embeddings for.
        Returns:
            Embeddings: The embeddings for the texts.
        Example:
            >>> openai = OpenAIEmbeddingFunction()
            >>> texts = ["Hello, world!", "How are you?"]
            >>> embeddings = openai(texts)
        """

        embeddings: Embeddings = []
        # Call OpenAI Embedding API for each document.
        for document in input:
            embedding_response = self._client.embeddings.create(
                model=self._model_name, input=[document]
            )
            embeddings.append(embedding_response.data[0].embedding)

        return embeddings


def load_data() -> None:
    """
    Loads data from /data/example to Chroma Vector store.
    """

    logger.info("Loading data.")
    # Split document into single sentences
    chunks = []
    with open(f"{DATA_DIR}/example/paul_graham_essay.txt", "r", encoding="utf-8") as file:
        text = file.read()
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        chunks.extend(sentences)

    logger.info("Creating embeddings.")
    openai_ef = OpenAIEmbeddingFunction()
    chunks_embeddings = openai_ef(chunks)

    db = chromadb.PersistentClient(path=f"{DATA_DIR}/chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")

    logger.info("Loading data in Chroma.")
    chroma_collection.add(
        ids=[f"id{i}" for i in range(1, len(chunks) + 1)],
        embeddings=chunks_embeddings,
        documents=chunks,
    )
    logger.info("Successfully loaded embeddings in the Chroma.")


if __name__ == "__main__":
    load_data()

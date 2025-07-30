import asyncio
import sys
import traceback

from beeai_framework.adapters.ollama import OllamaEmbeddingModel
from beeai_framework.errors import FrameworkError

from examples.setup import setup_observability

setup_observability()


async def main() -> None:
    llm = OllamaEmbeddingModel("nomic-embed-text:latest")
    response = await llm.create(values=["Hello", "world!"])

    print("LLM 🤖 : ", response.embeddings)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

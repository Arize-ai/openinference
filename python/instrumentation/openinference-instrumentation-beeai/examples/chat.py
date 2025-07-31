import asyncio
import sys
import traceback

from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.backend import UserMessage
from beeai_framework.errors import FrameworkError

from examples.setup import setup_observability

setup_observability()

prompt = "Hello, How are you?"


async def main() -> None:
    llm = OllamaChatModel("llama3.1")
    response = await llm.create(messages=[UserMessage(prompt)], stream=True, max_tokens=10)

    print("LLM 🤖 : ", response.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

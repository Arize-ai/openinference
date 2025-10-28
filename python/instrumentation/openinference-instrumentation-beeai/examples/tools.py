import asyncio
import sys
import traceback

from beeai_framework.errors import FrameworkError
from beeai_framework.tools.search.wikipedia import WikipediaTool, WikipediaToolInput

from examples.setup import setup_observability

setup_observability()


async def main() -> None:
    tool = WikipediaTool()
    result = await tool.run(input=WikipediaToolInput(query="Prague", full_text=False))
    print(result.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

import asyncio
import sys
import traceback
from datetime import date

from beeai_framework.errors import FrameworkError
from beeai_framework.tools.weather import OpenMeteoTool, OpenMeteoToolInput
from openinference_setup import setup_observability

setup_observability()


async def main() -> None:
    tool = OpenMeteoTool()
    result = await tool.run(
        input=OpenMeteoToolInput(
            location_name="New York", start_date=date(2025, 1, 1), end_date=date(2025, 2, 1)
        )
    )
    print(result.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

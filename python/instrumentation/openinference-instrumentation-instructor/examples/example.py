import asyncio
import os
from typing import Optional

import instructor
from phoenix.otel import register
from pydantic import BaseModel

from openinference.instrumentation.instructor import InstructorInstrumentor

os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
tracer_provider = register()
InstructorInstrumentor().instrument(tracer_provider=tracer_provider)


class User(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None


def run():
    client = instructor.from_provider("openai/gpt-4o-mini")
    user = client.chat.completions.create(
        response_model=User,
        messages=[{"role": "user", "content": "John is 25 years old"}],
    )
    print(user)  # User(name='John', age=25)


async def run_async():
    client = instructor.from_provider("openai/gpt-4o-mini")

    user_stream1 = client.chat.completions.create_partial(
        response_model=User,
        messages=[{"role": "user", "content": "Create a User"}],
    )
    final_user = None
    for user in user_stream1:
        final_user = user
    print(final_user)  # User(name='John', age=25)


if __name__ == "__main__":
    run()
    asyncio.run(run_async())

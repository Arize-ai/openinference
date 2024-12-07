import os

from groq import Groq
from phoenix.otel import register

from openinference.instrumentation.groq import GroqInstrumentor


def test():
    tracer_provider = register(project_name="groq_debug")
    GroqInstrumentor().instrument(tracer_provider=tracer_provider)

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    hello_world = {
        "type": "function",
        "function": {
            "name": "hello_world",
            "description": ("Print 'Hello world!'"),
            "parameters": {"input": "ex"},
        },
    }

    prompt = "Be a helpful assistant"
    msg = "say hello world"

    chat = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": msg}],
        temperature=0.0,
        tools=[hello_world],
        tool_choice="required",
    )
    return chat


if __name__ == "__main__":
    response = test()
    print(response)

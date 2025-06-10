import json
from typing import Any, Dict, List, Optional

import litellm

# Import OpenTelemetry components
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.litellm import LiteLLMInstrumentor

# Set up OpenTelemetry tracing with resource attributes
endpoint = "http://127.0.0.1:6006/v1/traces"
resource = Resource.create(
    {
        "service.name": "litellm-tools-example",
        "openinference.project.name": "openinference-litellm-tools",
    }
)
tracer_provider = TracerProvider(resource=resource)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def to_dict_message(msg: Any) -> Dict[str, Any]:
    # Only include role and content as str, fallback to empty string if None
    return {
        "role": str(getattr(msg, "role", "")),
        "content": str(getattr(msg, "content", "")),
    }


def test_parallel_function_call() -> None:
    try:
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        response: Any = litellm.completion(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # auto is default, but we'll be explicit
        )
        print("\nFirst LLM Response:\n", response)
        response_message: Any = response.choices[0].message
        tool_calls: Optional[Any] = getattr(response_message, "tool_calls", None)

        print("\nLength of tool calls", len(list(tool_calls or [])))

        if tool_calls:
            available_functions = {
                "get_current_weather": get_current_weather,
            }
            messages.append(response_message)
            for tool_call in list(tool_calls):
                function_name = getattr(getattr(tool_call, "function", None), "name", None)
                if function_name is None:
                    continue  # skip if function_name is not present
                function_to_call = available_functions[function_name]
                function_args = json.loads(
                    getattr(getattr(tool_call, "function", None), "arguments", "{}")
                )
                function_response = function_to_call(
                    location=function_args.get("location"),
                    unit=function_args.get("unit"),
                )
                messages.append(
                    {
                        "tool_call_id": getattr(tool_call, "id", ""),
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
            second_response: Any = litellm.completion(
                model="gpt-3.5-turbo-1106",
                messages=messages,
            )
            print("\nSecond LLM response:\n", second_response)
    except Exception as e:
        print(f"Error occurred: {e}")


test_parallel_function_call()

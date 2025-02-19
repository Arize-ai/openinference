from pydantic import BaseModel
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from llama_index.core.llms import ChatMessage

# Add Phoenix
span_phoenix_processor = SimpleSpanProcessor(
    HTTPSpanExporter(endpoint="http://localhost:6006/v1/traces")
)

# Add them to the tracer
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(span_processor=span_phoenix_processor)

# Instrument the application
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


class Song(BaseModel):
    """A song with name and artist"""

    name: str
    artist: str


class SongList(BaseModel):
    """A list of song names"""

    songs: list[str]


def generate_song(name: str, artist: str) -> Song:
    """Generates a song with provided name and artist."""
    return Song(name=name, artist=artist)


def process_song_list(songs: list[str]) -> SongList:
    """Processes a list of song names."""
    return SongList(songs=songs)


tool = FunctionTool.from_defaults(fn=generate_song)
list_tool = FunctionTool.from_defaults(fn=process_song_list)

chat_history = [ChatMessage(role="user", content="Generate five songs from the Beatles")]


llm = OpenAI(model="gpt-4o-mini")
resp = llm.chat_with_tools(
    [tool, list_tool],
    chat_history=chat_history,
)

tools_by_name = {t.metadata.name: t for t in [tool, list_tool]}
tool_calls = llm.get_tool_calls_from_response(resp, error_on_no_tool_call=False)

while tool_calls:
    # add the LLM's response to the chat history
    chat_history.append(resp.message)

    for tool_call in tool_calls:
        tool_name = tool_call.tool_name
        tool_kwargs = tool_call.tool_kwargs

        print(f"Calling {tool_name} with {tool_kwargs}")
        tool_output = tools_by_name[tool_name](**tool_kwargs)
        chat_history.append(
            ChatMessage(
                role="tool",
                content=str(tool_output),
                additional_kwargs={"tool_call_id": tool_call.tool_id},
            )
        )

        resp = llm.chat_with_tools([tool, list_tool], chat_history=chat_history)
        tool_calls = llm.get_tool_calls_from_response(resp, error_on_no_tool_call=False)

print(resp.message.content)

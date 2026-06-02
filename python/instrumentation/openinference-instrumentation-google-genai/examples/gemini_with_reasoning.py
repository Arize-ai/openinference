"""
Demonstrates how OpenInference captures Gemini reasoning (thinking) content.

Requires a .env file in this directory with GOOGLE_API_KEY=<your-key>
and a model that supports thinking (e.g. gemini-2.5-pro-preview-05-06).

Usage:
    pip install openinference-instrumentation-google-genai google-genai opentelemetry-sdk
    GOOGLE_API_KEY=sk-... python gemini_with_reasoning.py
"""

import os

from google import genai
from google.genai import types
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
MODEL = "gemini-2.5-pro-preview-05-06"

# ---------------------------------------------------------------------------
# 1. Non-streaming response with reasoning
# ---------------------------------------------------------------------------
print("=== Non-streaming with thinking ===")
response = client.models.generate_content(
    model=MODEL,
    contents="What is the capital of France? Show your reasoning.",
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=1024),
    ),
)
print(response.text)

# The span will contain:
#   llm.output_messages.0.message.contents.0.message_content.type = "reasoning"
#   llm.output_messages.0.message.contents.0.message_content.text = <thought text>
#   llm.output_messages.0.message.contents.0.message_content.signature = <base64>
#   llm.output_messages.0.message.contents.1.message_content.type = "text"
#   llm.output_messages.0.message.contents.1.message_content.text = "Paris."

# ---------------------------------------------------------------------------
# 2. Multi-turn: pass prior thought back via thought_signature
# ---------------------------------------------------------------------------
print("\n=== Multi-turn with thought signature replay ===")

# First turn — collect thought parts from the response
prior_parts: list[types.Part] = []
if response.candidates:
    candidate = response.candidates[0]
    if candidate.content and candidate.content.parts:
        for part in candidate.content.parts:
            prior_parts.append(part)

# Second turn — include the prior model content (thought + answer) as history
if prior_parts:
    follow_up = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="What is the capital of France? Show your reasoning.")],
            ),
            types.Content(role="model", parts=prior_parts),
            types.Content(role="user", parts=[types.Part(text="And what country is Paris in?")]),
        ],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=512),
        ),
    )
    print(follow_up.text)

# The input span for the second turn will show:
#   llm.input_messages.1.message.contents.0.message_content.type = "reasoning"
#   llm.input_messages.1.message.contents.0.message_content.signature = <base64>
#   llm.input_messages.1.message.contents.1.message_content.type = "text"

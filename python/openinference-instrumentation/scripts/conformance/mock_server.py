# /// script
# requires-python = ">=3.14"
# dependencies = ["flask"]
# ///
"""Mock LLM server serving deterministic Anthropic, OpenAI, and Google GenAI APIs.

Routes:
  POST /v1/messages                              Anthropic Messages API
  POST /v1/chat/completions                      OpenAI Chat Completions API
  POST /v1/embeddings                            OpenAI Embeddings API
  POST /v1beta/models/<model>:generateContent    Google GenAI generateContent

Each endpoint returns either a `tool_use` / `tool_calls` response when the request
has tool definitions, or a plain text response otherwise.
"""

import argparse

from flask import Flask, Response, jsonify, request

app = Flask(__name__)


# ── Anthropic ──────────────────────────────────────────────────────────────


_ANTHROPIC_TEXT_RESPONSE = {
    "id": "msg_mock_001",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "content": [{"type": "text", "text": "Hello from the mock Anthropic server."}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "output_tokens": 8,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 3,
    },
}

_ANTHROPIC_TOOL_RESPONSE = {
    "id": "msg_mock_002",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_mock_001",
            "name": "get_weather",
            "input": {"location": "Seattle"},
        }
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 30, "output_tokens": 14},
}


@app.route("/v1/messages", methods=["POST"])
def anthropic_messages() -> Response:
    body = request.get_json(silent=True) or {}
    payload = _ANTHROPIC_TOOL_RESPONSE if body.get("tools") else _ANTHROPIC_TEXT_RESPONSE
    return jsonify(payload)


# ── OpenAI ─────────────────────────────────────────────────────────────────


_OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl-mock-001",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello from the mock OpenAI server.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 4},
    },
}

_OPENAI_TOOL_RESPONSE = {
    "id": "chatcmpl-mock-002",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_mock_001",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Seattle"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
}

_OPENAI_RESPONSES_RESPONSE = {
    "id": "resp_mock_001",
    "object": "response",
    "created_at": 1700000000,
    "model": "gpt-4o-mini",
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg_resp_001",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Hello from the mock OpenAI Responses API.",
                    "annotations": [],
                }
            ],
        }
    ],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 4},
    },
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}

_OPENAI_RESPONSES_TOOL_RESPONSE = {
    "id": "resp_mock_002",
    "object": "response",
    "created_at": 1700000000,
    "model": "gpt-4o-mini",
    "status": "completed",
    "output": [
        {
            "type": "function_call",
            "id": "fc_mock_001",
            "call_id": "call_mock_001",
            "name": "get_weather",
            "arguments": '{"location": "Seattle"}',
            "status": "completed",
        }
    ],
    "usage": {"input_tokens": 30, "output_tokens": 14, "total_tokens": 44},
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}

_OPENAI_EMBEDDING_RESPONSE = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    ],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}


@app.route("/v1/chat/completions", methods=["POST"])
def openai_chat_completions() -> Response:
    body = request.get_json(silent=True) or {}
    payload = _OPENAI_TOOL_RESPONSE if body.get("tools") else _OPENAI_CHAT_RESPONSE
    return jsonify(payload)


@app.route("/v1/embeddings", methods=["POST"])
def openai_embeddings() -> Response:
    return jsonify(_OPENAI_EMBEDDING_RESPONSE)


@app.route("/v1/responses", methods=["POST"])
def openai_responses() -> Response:
    body = request.get_json(silent=True) or {}
    payload = _OPENAI_RESPONSES_TOOL_RESPONSE if body.get("tools") else _OPENAI_RESPONSES_RESPONSE
    return jsonify(payload)


# ── Google GenAI ───────────────────────────────────────────────────────────


_GOOGLE_TEXT_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "Hello from the mock Google GenAI server."}],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "modelVersion": "gemini-2.0-flash",
    "responseId": "resp-mock-001",
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 5,
        "totalTokenCount": 15,
        "cachedContentTokenCount": 3,
    },
}

_GOOGLE_TOOL_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"location": "Seattle"},
                        }
                    }
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "modelVersion": "gemini-2.0-flash",
    "responseId": "resp-mock-002",
    "usageMetadata": {
        "promptTokenCount": 30,
        "candidatesTokenCount": 14,
        "totalTokenCount": 44,
    },
}


@app.route("/v1beta/models/<path:model>", methods=["POST"])
def google_models(model: str) -> Response:
    body = request.get_json(silent=True) or {}
    has_tools = bool(body.get("tools") or body.get("toolConfig"))
    payload = _GOOGLE_TOOL_RESPONSE if has_tools else _GOOGLE_TEXT_RESPONSE
    return jsonify(payload)


# ── Health ─────────────────────────────────────────────────────────────────


@app.route("/health")
def health() -> str:
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

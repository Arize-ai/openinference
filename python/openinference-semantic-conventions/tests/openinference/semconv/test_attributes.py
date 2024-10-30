from collections import defaultdict
from typing import Any, DefaultDict, Mapping

from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    RerankerAttributes,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


class TestSpanAttributes:
    def test_nesting(self) -> None:
        attributes = {v: ... for k, v in SpanAttributes.__dict__.items() if k.isupper()}
        assert _nested_dict(attributes) == {
            "embedding": {
                "embeddings": ...,
                "model_name": ...,
            },
            "input": {
                "mime_type": ...,
                "value": ...,
            },
            "llm": {
                "function_call": ...,
                "input_messages": ...,
                "invocation_parameters": ...,
                "model_name": ...,
                "output_messages": ...,
                "prompt_template": {
                    "template": ...,
                    "variables": ...,
                    "version": ...,
                },
                "prompts": ...,
                "provider": ...,
                "system": ...,
                "token_count": {
                    "completion": ...,
                    "prompt": ...,
                    "total": ...,
                },
                "tools": ...,
            },
            "metadata": ...,
            "openinference": {"span": {"kind": ...}},
            "output": {
                "mime_type": ...,
                "value": ...,
            },
            "retrieval": {
                "documents": ...,
            },
            "session": {
                "id": ...,
            },
            "tag": {
                "tags": ...,
            },
            "tool": {
                "description": ...,
                "name": ...,
                "parameters": ...,
            },
            "user": {
                "id": ...,
            },
        }


class TestMessageAttributes:
    def test_nesting(self) -> None:
        attributes = {v: ... for k, v in MessageAttributes.__dict__.items() if k.isupper()}
        assert _nested_dict(attributes) == {
            "message": {
                "content": ...,
                "contents": ...,
                "function_call_arguments_json": ...,
                "function_call_name": ...,
                "name": ...,
                "role": ...,
                "tool_call_id": ...,
                "tool_calls": ...,
            }
        }


class TestMessageContentAttributes:
    def test_nesting(self) -> None:
        attributes = {v: ... for k, v in MessageContentAttributes.__dict__.items() if k.isupper()}
        assert _nested_dict(attributes) == {
            "message_content": {
                "image": ...,
                "text": ...,
                "type": ...,
            }
        }


class TestImageAttributes:
    def test_nesting(self) -> None:
        attributes = {v: ... for k, v in ImageAttributes.__dict__.items() if k.isupper()}
        assert _nested_dict(attributes) == {
            "image": {
                "url": ...,
            }
        }


class TestDocumentAttributes:
    def test_nesting(self) -> None:
        attributes = {v: ... for k, v in DocumentAttributes.__dict__.items() if k.isupper()}
        assert _nested_dict(attributes) == {
            "document": {
                "content": ...,
                "id": ...,
                "metadata": ...,
                "score": ...,
            }
        }


class TestRerankerAttributes:
    def test_nesting(self) -> None:
        attributes = {v: ... for k, v in RerankerAttributes.__dict__.items() if k.isupper()}
        assert _nested_dict(attributes) == {
            "reranker": {
                "input_documents": ...,
                "model_name": ...,
                "output_documents": ...,
                "query": ...,
                "top_k": ...,
            }
        }


class TestEmbeddingAttributes:
    def test_nesting(self) -> None:
        attributes = {v: ... for k, v in EmbeddingAttributes.__dict__.items() if k.isupper()}
        assert _nested_dict(attributes) == {
            "embedding": {
                "text": ...,
                "vector": ...,
            }
        }


class TestToolCallAttributes:
    def test_nesting(self) -> None:
        attributes = {v: ... for k, v in ToolCallAttributes.__dict__.items() if k.isupper()}
        assert _nested_dict(attributes) == {
            "tool_call": {
                "function": {
                    "arguments": ...,
                    "name": ...,
                },
                "id": ...,
            },
        }


class TestToolAttributes:
    def test_nesting(self) -> None:
        attributes = {v: ... for k, v in ToolAttributes.__dict__.items() if k.isupper()}
        assert _nested_dict(attributes) == {
            "tool": {
                "json_schema": ...,
            }
        }


def _trie() -> DefaultDict[str, Any]:
    return defaultdict(_trie)


def _nested_dict(
    attributes: Mapping[str, Any],
) -> DefaultDict[str, Any]:
    nested_attributes = _trie()
    for attribute_name, attribute_value in attributes.items():
        trie = nested_attributes
        keys = attribute_name.split(".")
        for key in keys[:-1]:
            trie = trie[key]
        trie[keys[-1]] = attribute_value
    return nested_attributes

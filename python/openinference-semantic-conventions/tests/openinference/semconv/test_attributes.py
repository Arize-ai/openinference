from typing import Any, Dict, Mapping, Type

from openinference.semconv.resource import ResourceAttributes
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
        attributes = _flat_dict(SpanAttributes)
        assert _nested_dict(attributes) == {
            "embedding": {
                "embeddings": "EMBEDDING_EMBEDDINGS",
                "model_name": "EMBEDDING_MODEL_NAME",
            },
            "input": {
                "mime_type": "INPUT_MIME_TYPE",
                "value": "INPUT_VALUE",
            },
            "llm": {
                "function_call": "LLM_FUNCTION_CALL",
                "input_messages": "LLM_INPUT_MESSAGES",
                "invocation_parameters": "LLM_INVOCATION_PARAMETERS",
                "model_name": "LLM_MODEL_NAME",
                "output_messages": "LLM_OUTPUT_MESSAGES",
                "prompt_template": {
                    "template": "LLM_PROMPT_TEMPLATE",
                    "variables": "LLM_PROMPT_TEMPLATE_VARIABLES",
                    "version": "LLM_PROMPT_TEMPLATE_VERSION",
                },
                "prompts": "LLM_PROMPTS",
                "provider": "LLM_PROVIDER",
                "system": "LLM_SYSTEM",
                "token_count": {
                    "completion": "LLM_TOKEN_COUNT_COMPLETION",
                    "completion_details": {
                        "reasoning": "LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING",
                        "audio": "LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO",
                    },
                    "prompt": "LLM_TOKEN_COUNT_PROMPT",
                    "prompt_details": {
                        "cache_write": "LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE",
                        "cache_read": "LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ",
                        "audio": "LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO",
                    },
                    "total": "LLM_TOKEN_COUNT_TOTAL",
                },
                "tools": "LLM_TOOLS",
            },
            "metadata": "METADATA",
            "openinference": {
                "span": {
                    "kind": "OPENINFERENCE_SPAN_KIND",
                }
            },
            "output": {
                "mime_type": "OUTPUT_MIME_TYPE",
                "value": "OUTPUT_VALUE",
            },
            "retrieval": {
                "documents": "RETRIEVAL_DOCUMENTS",
            },
            "session": {
                "id": "SESSION_ID",
            },
            "tag": {
                "tags": "TAG_TAGS",
            },
            "tool": {
                "description": "TOOL_DESCRIPTION",
                "name": "TOOL_NAME",
                "parameters": "TOOL_PARAMETERS",
            },
            "user": {
                "id": "USER_ID",
            },
            "prompt": {
                "id": "PROMPT_ID",
                "url": "PROMPT_URL",
                "vendor": "PROMPT_VENDOR",
            },
        }


class TestMessageAttributes:
    def test_nesting(self) -> None:
        attributes = _flat_dict(MessageAttributes)
        assert _nested_dict(attributes) == {
            "message": {
                "content": "MESSAGE_CONTENT",
                "contents": "MESSAGE_CONTENTS",
                "function_call_arguments_json": "MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON",
                "function_call_name": "MESSAGE_FUNCTION_CALL_NAME",
                "name": "MESSAGE_NAME",
                "role": "MESSAGE_ROLE",
                "tool_call_id": "MESSAGE_TOOL_CALL_ID",
                "tool_calls": "MESSAGE_TOOL_CALLS",
            }
        }


class TestMessageContentAttributes:
    def test_nesting(self) -> None:
        attributes = _flat_dict(MessageContentAttributes)
        assert _nested_dict(attributes) == {
            "message_content": {
                "image": "MESSAGE_CONTENT_IMAGE",
                "text": "MESSAGE_CONTENT_TEXT",
                "type": "MESSAGE_CONTENT_TYPE",
            }
        }


class TestImageAttributes:
    def test_nesting(self) -> None:
        attributes = _flat_dict(ImageAttributes)
        assert _nested_dict(attributes) == {
            "image": {
                "url": "IMAGE_URL",
            }
        }


class TestDocumentAttributes:
    def test_nesting(self) -> None:
        attributes = _flat_dict(DocumentAttributes)
        assert _nested_dict(attributes) == {
            "document": {
                "content": "DOCUMENT_CONTENT",
                "id": "DOCUMENT_ID",
                "metadata": "DOCUMENT_METADATA",
                "score": "DOCUMENT_SCORE",
            }
        }


class TestRerankerAttributes:
    def test_nesting(self) -> None:
        attributes = _flat_dict(RerankerAttributes)
        assert _nested_dict(attributes) == {
            "reranker": {
                "input_documents": "RERANKER_INPUT_DOCUMENTS",
                "model_name": "RERANKER_MODEL_NAME",
                "output_documents": "RERANKER_OUTPUT_DOCUMENTS",
                "query": "RERANKER_QUERY",
                "top_k": "RERANKER_TOP_K",
            }
        }


class TestEmbeddingAttributes:
    def test_nesting(self) -> None:
        attributes = _flat_dict(EmbeddingAttributes)
        assert _nested_dict(attributes) == {
            "embedding": {
                "text": "EMBEDDING_TEXT",
                "vector": "EMBEDDING_VECTOR",
            }
        }


class TestToolCallAttributes:
    def test_nesting(self) -> None:
        attributes = _flat_dict(ToolCallAttributes)
        assert _nested_dict(attributes) == {
            "tool_call": {
                "function": {
                    "arguments": "TOOL_CALL_FUNCTION_ARGUMENTS_JSON",
                    "name": "TOOL_CALL_FUNCTION_NAME",
                },
                "id": "TOOL_CALL_ID",
            },
        }


class TestToolAttributes:
    def test_nesting(self) -> None:
        attributes = _flat_dict(ToolAttributes)
        assert _nested_dict(attributes) == {
            "tool": {
                "json_schema": "TOOL_JSON_SCHEMA",
            }
        }


class TestResourceAttributes:
    def test_nesting(self) -> None:
        attributes = _flat_dict(ResourceAttributes)
        assert _nested_dict(attributes) == {
            "openinference": {
                "project": {
                    "name": "PROJECT_NAME",
                }
            }
        }


def _flat_dict(cls: Type[Any]) -> Dict[str, str]:
    return {v: k for k, v in cls.__dict__.items() if k.isupper()}


def _nested_dict(
    attributes: Mapping[str, str],
) -> Dict[str, Any]:
    nested_attributes: Dict[str, Any] = {}
    for name, value in attributes.items():
        trie = nested_attributes
        keys = name.split(".")
        for key in keys[:-1]:
            if key not in trie:
                trie[key] = {}
            trie = trie[key]
        trie[keys[-1]] = value
    return nested_attributes

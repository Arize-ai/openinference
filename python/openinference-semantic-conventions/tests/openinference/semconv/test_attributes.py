# ruff: noqa: E501
"""Tests for OpenInference semantic convention attributes.

These tests verify that our OpenTelemetry span attribute semantic conventions correctly
structure flat span attributes into nested hierarchies. The tests ensure that when
raw spans with flat attributes are processed, they are correctly organized into
a hierarchical structure without any attribute collisions.

Raw spans come in as flat dictionaries where all attributes are at the root level:
    {
        "llm.model_name": "gpt-4",
        "llm.provider": "openai",
        "llm.token_count.prompt": 100,
        "llm.token_count.completion": 50,
        "llm.token_count.total": 150,
        "input.value": "What is the weather?",
        "input.mime_type": "text/plain",
        "output.value": "The weather is sunny",
        "output.mime_type": "text/plain"
    }

After ingestion, these attributes are organized into a hierarchical structure:
    {
        "llm": {
            "model_name": "gpt-4",
            "provider": "openai",
            "token_count": {
                "prompt": 100,
                "completion": 50,
                "total": 150
            }
        },
        "input": {
            "value": "What is the weather?",
            "mime_type": "text/plain"
        },
        "output": {
            "value": "The weather is sunny",
            "mime_type": "text/plain"
        }
    }

The semantic conventions define how this transformation should occur, ensuring that:
1. Attributes are properly nested under their namespaces
2. No attribute collisions occur during the transformation
3. The hierarchical structure makes relationships between attributes clear

These tests verify that the transformation from flat to hierarchical structure
is correct and that no attribute collisions occur when the spans are processed
by OpenTelemetry collectors and backends.
"""

from typing import Any

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
    """Tests for SpanAttributes namespace structure.

    Verifies that all span-related attributes are properly nested under their
    respective namespaces (e.g., llm, input, output) to prevent attribute
    collisions during ingestion. The test ensures that when flat span attributes
    are processed, they are correctly organized into a hierarchical structure
    that preserves the relationships between attributes.
    """

    def test_nesting(self) -> None:
        attributes = _get_attributes(SpanAttributes)
        assert _nested_dict(attributes) == {
            "agent": {
                "name": SpanAttributes.AGENT_NAME,
            },
            "embedding": {
                "embeddings": SpanAttributes.EMBEDDING_EMBEDDINGS,
                "invocation_parameters": SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS,
                "model_name": SpanAttributes.EMBEDDING_MODEL_NAME,
            },
            "graph": {
                "node": {
                    "id": SpanAttributes.GRAPH_NODE_ID,
                    "name": SpanAttributes.GRAPH_NODE_NAME,
                    "parent_id": SpanAttributes.GRAPH_NODE_PARENT_ID,
                },
            },
            "input": {
                "mime_type": SpanAttributes.INPUT_MIME_TYPE,
                "value": SpanAttributes.INPUT_VALUE,
            },
            "llm": {
                "choices": SpanAttributes.LLM_CHOICES,
                "cost": {
                    "completion": SpanAttributes.LLM_COST_COMPLETION,
                    "completion_details": {
                        "audio": SpanAttributes.LLM_COST_COMPLETION_DETAILS_AUDIO,
                        "output": SpanAttributes.LLM_COST_COMPLETION_DETAILS_OUTPUT,
                        "reasoning": SpanAttributes.LLM_COST_COMPLETION_DETAILS_REASONING,
                    },
                    "prompt": SpanAttributes.LLM_COST_PROMPT,
                    "prompt_details": {
                        "audio": SpanAttributes.LLM_COST_PROMPT_DETAILS_AUDIO,
                        "input": SpanAttributes.LLM_COST_PROMPT_DETAILS_INPUT,
                        "cache_input": SpanAttributes.LLM_COST_PROMPT_DETAILS_CACHE_INPUT,
                        "cache_read": SpanAttributes.LLM_COST_PROMPT_DETAILS_CACHE_READ,
                        "cache_write": SpanAttributes.LLM_COST_PROMPT_DETAILS_CACHE_WRITE,
                    },
                    "total": SpanAttributes.LLM_COST_TOTAL,
                },
                "function_call": SpanAttributes.LLM_FUNCTION_CALL,
                "input_messages": SpanAttributes.LLM_INPUT_MESSAGES,
                "invocation_parameters": SpanAttributes.LLM_INVOCATION_PARAMETERS,
                "model_name": SpanAttributes.LLM_MODEL_NAME,
                "output_messages": SpanAttributes.LLM_OUTPUT_MESSAGES,
                "prompt_template": {
                    "template": SpanAttributes.LLM_PROMPT_TEMPLATE,
                    "variables": SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES,
                    "version": SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION,
                },
                "prompts": SpanAttributes.LLM_PROMPTS,
                "provider": SpanAttributes.LLM_PROVIDER,
                "system": SpanAttributes.LLM_SYSTEM,
                "token_count": {
                    "completion": SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                    "completion_details": {
                        "audio": SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO,
                        "reasoning": SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
                    },
                    "prompt": SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
                    "prompt_details": {
                        "audio": SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO,
                        "cache_input": SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_INPUT,
                        "cache_read": SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
                        "cache_write": SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
                    },
                    "total": SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
                },
                "tools": SpanAttributes.LLM_TOOLS,
            },
            "metadata": SpanAttributes.METADATA,
            "openinference": {
                "span": {
                    "kind": SpanAttributes.OPENINFERENCE_SPAN_KIND,
                }
            },
            "output": {
                "mime_type": SpanAttributes.OUTPUT_MIME_TYPE,
                "value": SpanAttributes.OUTPUT_VALUE,
            },
            "retrieval": {
                "documents": SpanAttributes.RETRIEVAL_DOCUMENTS,
            },
            "session": {
                "id": SpanAttributes.SESSION_ID,
            },
            "tag": {
                "tags": SpanAttributes.TAG_TAGS,
            },
            "tool": {
                "description": SpanAttributes.TOOL_DESCRIPTION,
                "name": SpanAttributes.TOOL_NAME,
                "parameters": SpanAttributes.TOOL_PARAMETERS,
            },
            "user": {
                "id": SpanAttributes.USER_ID,
            },
            "prompt": {
                "id": SpanAttributes.PROMPT_ID,
                "url": SpanAttributes.PROMPT_URL,
                "vendor": SpanAttributes.PROMPT_VENDOR,
            },
        }


class TestMessageAttributes:
    """Tests for MessageAttributes namespace structure.

    Ensures that message-related attributes from flat spans are properly organized
    under the message namespace to maintain clear separation from other attribute types.
    """

    def test_nesting(self) -> None:
        attributes = _get_attributes(MessageAttributes)
        assert _nested_dict(attributes) == {
            "message": {
                "content": MessageAttributes.MESSAGE_CONTENT,
                "contents": MessageAttributes.MESSAGE_CONTENTS,
                "function_call_arguments_json": MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON,
                "function_call_name": MessageAttributes.MESSAGE_FUNCTION_CALL_NAME,
                "name": MessageAttributes.MESSAGE_NAME,
                "role": MessageAttributes.MESSAGE_ROLE,
                "tool_call_id": MessageAttributes.MESSAGE_TOOL_CALL_ID,
                "tool_calls": MessageAttributes.MESSAGE_TOOL_CALLS,
            }
        }


class TestMessageContentAttributes:
    """Tests for MessageContentAttributes namespace structure.

    Verifies the nesting of message content attributes (text, image, type)
    from flat spans under the message_content namespace.
    """

    def test_nesting(self) -> None:
        attributes = _get_attributes(MessageContentAttributes)
        assert _nested_dict(attributes) == {
            "message_content": {
                "image": MessageContentAttributes.MESSAGE_CONTENT_IMAGE,
                "text": MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                "type": MessageContentAttributes.MESSAGE_CONTENT_TYPE,
            }
        }


class TestImageAttributes:
    """Tests for ImageAttributes namespace structure.

    Ensures image-related attributes from flat spans are properly nested
    under the image namespace.
    """

    def test_nesting(self) -> None:
        attributes = _get_attributes(ImageAttributes)
        assert _nested_dict(attributes) == {
            "image": {
                "url": ImageAttributes.IMAGE_URL,
            }
        }


class TestDocumentAttributes:
    """Tests for DocumentAttributes namespace structure.

    Verifies that document-related attributes (content, id, metadata, score)
    from flat spans are properly organized under the document namespace.
    """

    def test_nesting(self) -> None:
        attributes = _get_attributes(DocumentAttributes)
        assert _nested_dict(attributes) == {
            "document": {
                "content": DocumentAttributes.DOCUMENT_CONTENT,
                "id": DocumentAttributes.DOCUMENT_ID,
                "metadata": DocumentAttributes.DOCUMENT_METADATA,
                "score": DocumentAttributes.DOCUMENT_SCORE,
            }
        }


class TestRerankerAttributes:
    """Tests for RerankerAttributes namespace structure.

    Ensures reranker-related attributes from flat spans are properly nested
    under the reranker namespace to prevent conflicts with other attribute types.
    """

    def test_nesting(self) -> None:
        attributes = _get_attributes(RerankerAttributes)
        assert _nested_dict(attributes) == {
            "reranker": {
                "input_documents": RerankerAttributes.RERANKER_INPUT_DOCUMENTS,
                "model_name": RerankerAttributes.RERANKER_MODEL_NAME,
                "output_documents": RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS,
                "query": RerankerAttributes.RERANKER_QUERY,
                "top_k": RerankerAttributes.RERANKER_TOP_K,
            }
        }


class TestEmbeddingAttributes:
    """Tests for EmbeddingAttributes namespace structure.

    Verifies that embedding-related attributes (text, vector) from flat spans
    are properly organized under the embedding namespace.
    """

    def test_nesting(self) -> None:
        attributes = _get_attributes(EmbeddingAttributes)
        assert _nested_dict(attributes) == {
            "embedding": {
                "text": EmbeddingAttributes.EMBEDDING_TEXT,
                "vector": EmbeddingAttributes.EMBEDDING_VECTOR,
            }
        }


class TestToolCallAttributes:
    """Tests for ToolCallAttributes namespace structure.

    Ensures tool call attributes from flat spans are properly nested under
    the tool_call namespace, with function-related attributes further nested
    under the function namespace.
    """

    def test_nesting(self) -> None:
        attributes = _get_attributes(ToolCallAttributes)
        assert _nested_dict(attributes) == {
            "tool_call": {
                "function": {
                    "arguments": ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
                    "name": ToolCallAttributes.TOOL_CALL_FUNCTION_NAME,
                },
                "id": ToolCallAttributes.TOOL_CALL_ID,
            },
        }


class TestToolAttributes:
    """Tests for ToolAttributes namespace structure.

    Verifies that tool-related attributes from flat spans are properly
    organized under the tool namespace.
    """

    def test_nesting(self) -> None:
        attributes = _get_attributes(ToolAttributes)
        assert _nested_dict(attributes) == {
            "tool": {
                "json_schema": ToolAttributes.TOOL_JSON_SCHEMA,
            }
        }


class TestResourceAttributes:
    """Tests for ResourceAttributes namespace structure.

    Ensures resource-related attributes from flat spans are properly nested
    under the openinference.project namespace.
    """

    def test_nesting(self) -> None:
        attributes = _get_attributes(ResourceAttributes)
        assert _nested_dict(attributes) == {
            "openinference": {
                "project": {
                    "name": ResourceAttributes.PROJECT_NAME,
                }
            }
        }


def _get_attributes(cls: type[Any]) -> set[str]:
    """Extract all uppercase attributes from a semantic convention class.

    Args:
        cls: The semantic convention class to extract attributes from.

    Returns:
        A set of all uppercase attribute values from the class, which represent
        the dot-notation keys used in flat span attributes.
    """
    return {v for k, v in cls.__dict__.items() if k.isupper()}


_PREFIXES: list[str] = [
    SpanAttributes.LLM_COST_PROMPT_DETAILS,
    SpanAttributes.LLM_COST_COMPLETION_DETAILS,
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS,
]

"""List of OpenTelemetry span attribute prefixes that should be ignored during nesting.

These prefixes represent intermediate namespace levels that should not be treated
as actual attributes. They are used to organize the attribute namespace when
converting flat span attributes into nested structures.

For example:
- 'llm.cost' is a namespace prefix for attributes like 'llm.cost.completion'
- 'llm.token_count.prompt_details' is a namespace prefix for attributes like
  'llm.token_count.prompt_details.cache_write'
- 'llm.token_count.completion_details' is a namespace prefix for attributes like
  'llm.token_count.completion_details.reasoning'

If these prefixes were treated as attributes, they would create invalid nesting
structures that could lead to attribute collisions. The prefixes are purely
organizational and should not appear as leaf nodes in the nested structure.

Example of why prefixes must be ignored:
Consider a span with these attributes:
    {
        "llm.cost": 0.001,           # This would be wrong - cost is a namespace
        "llm.cost.completion": 0.002,
        "llm.cost.total": 0.003
    }

If we didn't ignore the 'llm.cost' prefix, we would get a nested structure like:
    {
        "llm": {
            "cost": 0.001,           # This value would clobber the namespace
            "cost": {                # This creates a conflict - can't have both
                "completion": 0.002, # a value and a namespace with the same key
                "total": 0.003
            }
        }
    }

This would cause the 'llm.cost' value to be lost or overwritten, and the structure
would be invalid. Instead, we should only have:
    {
        "llm": {
            "cost": {
                "completion": 0.002,
                "total": 0.003
            }
        }
    }
"""


def _nested_dict(
    attributes: set[str],
) -> dict[str, Any]:
    """Convert a set of OpenTelemetry span attribute names into a nested dictionary structure.

    This function demonstrates how flat span attributes should be organized into
    a hierarchical structure. It takes the dot-notation keys from a flat span
    and shows how they should be nested. This is used to verify that our semantic
    conventions correctly structure the attributes without collisions.

    For example, given these flat span attributes:
        {
            "llm.model_name": "gpt-4",
            "llm.provider": "openai",
            "input.value": "What is the weather?",
            "input.mime_type": "text/plain"
        }

    This function would return a structure showing how they should be nested:
        {
            "llm": {
                "model_name": "llm.model_name",
                "provider": "llm.provider"
            },
            "input": {
                "value": "input.value",
                "mime_type": "input.mime_type"
            }
        }

    Note: The leaf nodes contain the full attribute name as it appears in the
    flat span attributes. This helps verify that the nesting structure is correct
    while preserving the original attribute names.

    Args:
        attributes: A set of OpenTelemetry span attribute names in dot notation.

    Returns:
        A nested dictionary showing how the flat span attributes should be
        organized into a hierarchical structure.
    """
    nested_attributes: dict[str, Any] = {}
    for value in attributes:
        # Skip prefixes like 'llm.cost' or 'llm.token_count.prompt_details' because they are
        # intermediate namespace levels, not actual attributes. If we didn't skip them,
        # they would incorrectly appear as leaf nodes in the nested structure, potentially
        # causing attribute collisions with their child attributes.
        if value in _PREFIXES:
            continue
        trie = nested_attributes
        keys = value.split(".")
        for key in keys[:-1]:
            if key not in trie:
                trie[key] = {}
            trie = trie[key]
        trie[keys[-1]] = value
    return nested_attributes

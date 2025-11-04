import {
  MimeType,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  INPUT_MIME_TYPE,
  INPUT_VALUE,
  OUTPUT_MIME_TYPE,
  OUTPUT_VALUE,
} from "@arizeai/openinference-semantic-conventions";

import {
  defaultProcessInput,
  defaultProcessOutput,
  getDocumentAttributes,
  getEmbeddingAttributes,
  getInputAttributes,
  getLLMAttributes,
  getMetadataAttributes,
  getOutputAttributes,
  getRetrieverAttributes,
  getToolAttributes,
  toInputType,
  toOutputType,
} from "../../src/helpers/attributeHelpers";

import { describe, expect, it } from "vitest";

describe("attributeHelpers", () => {
  describe("toInputType", () => {
    it("should return undefined for empty arguments", () => {
      expect(toInputType()).toBeUndefined();
    });

    it("should return text mime type for string input", () => {
      const result = toInputType("hello world");
      expect(result).toEqual({
        value: "hello world",
        mimeType: MimeType.TEXT,
      });
    });

    it("should return JSON mime type for object input", () => {
      const input = { key: "value", number: 42 };
      const result = toInputType(input);
      expect(result).toEqual({
        value: JSON.stringify(input),
        mimeType: MimeType.JSON,
      });
    });

    it("should return JSON mime type for multiple arguments", () => {
      const result = toInputType("arg1", 42, { key: "value" });
      expect(result).toEqual({
        value: JSON.stringify(["arg1", 42, { key: "value" }]),
        mimeType: MimeType.JSON,
      });
    });

    it("should handle null and undefined inputs", () => {
      expect(toInputType(null)).toEqual(undefined);
      expect(toInputType(undefined)).toEqual(undefined);
    });
  });

  describe("toOutputType", () => {
    it("should return undefined for null/undefined", () => {
      expect(toOutputType(null)).toBeUndefined();
      expect(toOutputType(undefined)).toBeUndefined();
    });

    it("should return text mime type for string output", () => {
      const result = toOutputType("success message");
      expect(result).toEqual({
        value: "success message",
        mimeType: MimeType.TEXT,
      });
    });

    it("should return JSON mime type for object output", () => {
      const output = { status: "ok", data: [1, 2, 3] };
      const result = toOutputType(output);
      expect(result).toEqual({
        value: JSON.stringify(output),
        mimeType: MimeType.JSON,
      });
    });

    it("should handle primitive types", () => {
      expect(toOutputType(42)).toEqual({
        value: "42",
        mimeType: MimeType.JSON,
      });
      expect(toOutputType(true)).toEqual({
        value: "true",
        mimeType: MimeType.JSON,
      });
    });
  });

  describe("defaultProcessInput", () => {
    it("should process string input correctly", () => {
      const result = defaultProcessInput("test input");
      expect(result).toEqual({
        [INPUT_VALUE]: "test input",
        [INPUT_MIME_TYPE]: MimeType.TEXT,
      });
    });

    it("should process object input correctly", () => {
      const input = { query: "search", limit: 10 };
      const result = defaultProcessInput(input);
      expect(result).toEqual({
        [INPUT_VALUE]: JSON.stringify(input),
        [INPUT_MIME_TYPE]: MimeType.JSON,
      });
    });

    it("should handle multiple arguments", () => {
      const result = defaultProcessInput("arg1", 42);
      expect(result).toEqual({
        [INPUT_VALUE]: JSON.stringify(["arg1", 42]),
        [INPUT_MIME_TYPE]: MimeType.JSON,
      });
    });
  });

  describe("defaultProcessOutput", () => {
    it("should process string output correctly", () => {
      const result = defaultProcessOutput("success");
      expect(result).toEqual({
        [OUTPUT_VALUE]: "success",
        [OUTPUT_MIME_TYPE]: MimeType.TEXT,
      });
    });

    it("should process object output correctly", () => {
      const output = { result: "ok", count: 5 };
      const result = defaultProcessOutput(output);
      expect(result).toEqual({
        [OUTPUT_VALUE]: JSON.stringify(output),
        [OUTPUT_MIME_TYPE]: MimeType.JSON,
      });
    });

    it("should return empty object for null/undefined", () => {
      expect(defaultProcessOutput(null)).toEqual({});
      expect(defaultProcessOutput(undefined)).toEqual({});
    });
  });

  describe("getInputAttributes", () => {
    it("should handle SpanInput object", () => {
      const input = { value: "test", mimeType: MimeType.TEXT };
      const result = getInputAttributes(input);
      expect(result).toEqual({
        [INPUT_VALUE]: "test",
        [INPUT_MIME_TYPE]: MimeType.TEXT,
      });
    });

    it("should handle string input", () => {
      const result = getInputAttributes("simple string");
      expect(result).toEqual({
        [INPUT_VALUE]: "simple string",
        [INPUT_MIME_TYPE]: MimeType.TEXT,
      });
    });

    it("should return empty object for undefined", () => {
      expect(getInputAttributes(undefined)).toEqual({});
    });
  });

  describe("getOutputAttributes", () => {
    it("should handle SpanOutput object", () => {
      const output = { value: "result", mimeType: MimeType.JSON };
      const result = getOutputAttributes(output);
      expect(result).toEqual({
        [OUTPUT_VALUE]: "result",
        [OUTPUT_MIME_TYPE]: MimeType.JSON,
      });
    });

    it("should handle string output", () => {
      const result = getOutputAttributes("simple result");
      expect(result).toEqual({
        [OUTPUT_VALUE]: "simple result",
        [OUTPUT_MIME_TYPE]: MimeType.TEXT,
      });
    });

    it("should return empty object for undefined", () => {
      expect(getOutputAttributes(undefined)).toEqual({});
    });
  });

  describe("getEmbeddingAttributes", () => {
    it("should generate attributes with model name only", () => {
      const result = getEmbeddingAttributes({
        modelName: "text-embedding-ada-002",
      });
      expect(result).toEqual({
        [SemanticConventions.EMBEDDING_MODEL_NAME]: "text-embedding-ada-002",
      });
    });

    it("should generate attributes with embeddings", () => {
      const embeddings = [
        { text: "hello", vector: [0.1, 0.2, 0.3] },
        { text: "world", vector: [0.4, 0.5, 0.6] },
      ];
      const result = getEmbeddingAttributes({ embeddings });
      expect(result).toEqual({
        [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
          "hello",
        [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
          [0.1, 0.2, 0.3],
        [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.1.${SemanticConventions.EMBEDDING_TEXT}`]:
          "world",
        [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.1.${SemanticConventions.EMBEDDING_VECTOR}`]:
          [0.4, 0.5, 0.6],
      });
    });

    it("should handle embeddings with missing fields", () => {
      const embeddings = [
        { text: "hello" }, // missing vector
        { vector: [0.1, 0.2] }, // missing text
      ];
      const result = getEmbeddingAttributes({ embeddings });
      expect(result).toEqual({
        [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
          "hello",
        [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.1.${SemanticConventions.EMBEDDING_VECTOR}`]:
          [0.1, 0.2],
      });
    });

    it("should return empty object for empty options", () => {
      expect(getEmbeddingAttributes({})).toEqual({});
    });
  });

  describe("getRetrieverAttributes", () => {
    it("should generate attributes for documents", () => {
      const documents = [
        { content: "Document 1", id: "doc1", score: 0.95 },
        { content: "Document 2", id: "doc2", metadata: { source: "web" } },
      ];
      const result = getRetrieverAttributes({ documents });
      expect(result).toEqual({
        [`${SemanticConventions.RETRIEVAL_DOCUMENTS}.0.${SemanticConventions.DOCUMENT_CONTENT}`]:
          "Document 1",
        [`${SemanticConventions.RETRIEVAL_DOCUMENTS}.0.${SemanticConventions.DOCUMENT_ID}`]:
          "doc1",
        [`${SemanticConventions.RETRIEVAL_DOCUMENTS}.0.${SemanticConventions.DOCUMENT_SCORE}`]: 0.95,
        [`${SemanticConventions.RETRIEVAL_DOCUMENTS}.1.${SemanticConventions.DOCUMENT_CONTENT}`]:
          "Document 2",
        [`${SemanticConventions.RETRIEVAL_DOCUMENTS}.1.${SemanticConventions.DOCUMENT_ID}`]:
          "doc2",
        [`${SemanticConventions.RETRIEVAL_DOCUMENTS}.1.${SemanticConventions.DOCUMENT_METADATA}`]:
          JSON.stringify({ source: "web" }),
      });
    });

    it("should handle string metadata", () => {
      const documents = [{ content: "Doc", metadata: "simple metadata" }];
      const result = getRetrieverAttributes({ documents });
      expect(result).toEqual({
        [`${SemanticConventions.RETRIEVAL_DOCUMENTS}.0.${SemanticConventions.DOCUMENT_CONTENT}`]:
          "Doc",
        [`${SemanticConventions.RETRIEVAL_DOCUMENTS}.0.${SemanticConventions.DOCUMENT_METADATA}`]:
          "simple metadata",
      });
    });
  });

  describe("getDocumentAttributes", () => {
    it("should generate attributes for a document", () => {
      const document = {
        content: "Sample document",
        id: "doc123",
        score: 0.8,
        metadata: { source: "file" },
      };
      const result = getDocumentAttributes(
        document,
        0,
        "reranker.input_documents",
      );
      expect(result).toEqual({
        "reranker.input_documents.0.document.content": "Sample document",
        "reranker.input_documents.0.document.id": "doc123",
        "reranker.input_documents.0.document.score": 0.8,
        "reranker.input_documents.0.document.metadata": JSON.stringify({
          source: "file",
        }),
      });
    });

    it("should handle document with missing fields", () => {
      const document = { content: "Minimal doc" };
      const result = getDocumentAttributes(document, 1, "test.prefix");
      expect(result).toEqual({
        "test.prefix.1.document.content": "Minimal doc",
      });
    });
  });

  describe("getMetadataAttributes", () => {
    it("should generate metadata attributes", () => {
      const metadata = { version: "1.0", env: "prod", debug: true };
      const result = getMetadataAttributes(metadata);
      expect(result).toEqual({
        [SemanticConventions.METADATA]: JSON.stringify(metadata),
      });
    });

    it("should handle empty metadata", () => {
      const result = getMetadataAttributes({});
      expect(result).toEqual({
        [SemanticConventions.METADATA]: "{}",
      });
    });
  });

  describe("getToolAttributes", () => {
    it("should generate tool attributes with all fields", () => {
      const options = {
        name: "search_tool",
        description: "Search for information",
        parameters: { query: { type: "string" }, limit: { type: "number" } },
      };
      const result = getToolAttributes(options);
      expect(result).toEqual({
        [SemanticConventions.TOOL_NAME]: "search_tool",
        [SemanticConventions.TOOL_DESCRIPTION]: "Search for information",
        [SemanticConventions.TOOL_PARAMETERS]: JSON.stringify(
          options.parameters,
        ),
      });
    });

    it("should generate tool attributes without description", () => {
      const options = {
        name: "simple_tool",
        parameters: { action: "execute" },
      };
      const result = getToolAttributes(options);
      expect(result).toEqual({
        [SemanticConventions.TOOL_NAME]: "simple_tool",
        [SemanticConventions.TOOL_PARAMETERS]: JSON.stringify({
          action: "execute",
        }),
      });
    });
  });

  describe("getLLMAttributes", () => {
    it("should generate basic LLM attributes", () => {
      const options = {
        provider: "openai",
        modelName: "gpt-4",
        system: "assistant",
      };
      const result = getLLMAttributes(options);
      expect(result).toEqual({
        [SemanticConventions.LLM_PROVIDER]: "openai",
        [SemanticConventions.LLM_SYSTEM]: "assistant",
        [SemanticConventions.LLM_MODEL_NAME]: "gpt-4",
      });
    });

    it("should generate attributes with invocation parameters", () => {
      const options = {
        provider: "anthropic",
        invocationParameters: { temperature: 0.7, maxTokens: 100 },
      };
      const result = getLLMAttributes(options);
      expect(result).toEqual({
        [SemanticConventions.LLM_PROVIDER]: "anthropic",
        [SemanticConventions.LLM_INVOCATION_PARAMETERS]: JSON.stringify({
          temperature: 0.7,
          maxTokens: 100,
        }),
      });
    });

    it("should generate attributes with input messages", () => {
      const options = {
        inputMessages: [
          { role: "user", content: "Hello" },
          { role: "assistant", content: "Hi there!" },
        ],
      };
      const result = getLLMAttributes(options);
      expect(result).toEqual({
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
          "user",
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
          "Hello",
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_ROLE}`]:
          "assistant",
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENT}`]:
          "Hi there!",
      });
    });

    it("should generate attributes with complex message content", () => {
      const result = getLLMAttributes({
        inputMessages: [
          {
            role: "user",
            contents: [
              { type: "text", text: "Hello" },
              {
                type: "image",
                image: { url: "https://example.com/image.jpg" },
              },
            ],
          },
        ],
      });
      expect(result).toEqual({
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
          "user",
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TYPE}`]:
          "text",
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TEXT}`]:
          "Hello",
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}.1.${SemanticConventions.MESSAGE_CONTENT_TYPE}`]:
          "image",
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}.1.${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`]:
          "https://example.com/image.jpg",
      });
    });

    it("should generate attributes with tool calls", () => {
      const options = {
        inputMessages: [
          {
            role: "assistant",
            toolCalls: [
              {
                id: "call_123",
                function: { name: "search", arguments: { query: "test" } },
              },
            ],
          },
        ],
      };
      const result = getLLMAttributes(options);
      expect(result).toEqual({
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
          "assistant",
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.0.${SemanticConventions.TOOL_CALL_ID}`]:
          "call_123",
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.0.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]:
          "search",
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.0.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]:
          JSON.stringify({ query: "test" }),
      });
    });

    it("should generate attributes with token count", () => {
      const options = {
        tokenCount: {
          prompt: 100,
          completion: 50,
          total: 150,
          promptDetails: {
            audio: 10,
            cacheRead: 5,
            cacheWrite: 2,
          },
        },
      };
      const result = getLLMAttributes(options);
      expect(result).toEqual({
        [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: 100,
        [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: 50,
        [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: 150,
        [SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO]: 10,
        [SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]: 5,
        [SemanticConventions.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]: 2,
      });
    });

    it("should generate attributes with tools", () => {
      const options = {
        tools: [
          { jsonSchema: { name: "tool1", parameters: { type: "object" } } },
          { jsonSchema: JSON.stringify({ name: "tool2" }) },
        ],
      };
      const result = getLLMAttributes(options);
      expect(result).toEqual({
        [`${SemanticConventions.LLM_TOOLS}.0.${SemanticConventions.TOOL_JSON_SCHEMA}`]:
          JSON.stringify({ name: "tool1", parameters: { type: "object" } }),
        [`${SemanticConventions.LLM_TOOLS}.1.${SemanticConventions.TOOL_JSON_SCHEMA}`]:
          JSON.stringify({ name: "tool2" }),
      });
    });

    it("should handle empty options", () => {
      expect(getLLMAttributes({})).toEqual({});
    });
  });
});

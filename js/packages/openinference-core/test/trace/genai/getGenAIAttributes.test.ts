import { describe, expect, it } from "vitest";

import { getGenAIAttributes } from "../../../src";

const parseJSONAttribute = (value: unknown): unknown => {
  expect(typeof value).toBe("string");
  return JSON.parse(value as string);
};

describe("getGenAIAttributes", () => {
  it("should return no attributes when there is nothing to convert", () => {
    expect(getGenAIAttributes()).toEqual({});
    expect(getGenAIAttributes(null)).toEqual({});
    expect(getGenAIAttributes({})).toEqual({});
    expect(getGenAIAttributes({ "openinference.span.kind": "CHAIN" })).toEqual({});
  });

  describe("chat spans", () => {
    const chatAttributes = {
      "openinference.span.kind": "LLM",
      "llm.provider": "openai",
      "llm.system": "openai",
      "llm.model_name": "gpt-4o",
      "session.id": "session-1",
      "llm.invocation_parameters": JSON.stringify({
        temperature: 0.2,
        top_p: 0.9,
        max_completion_tokens: 256,
        seed: 7,
        stream: true,
        stop: ["END"],
        n: 1,
        response_format: { type: "json_schema" },
      }),
      "llm.input_messages.0.message.role": "system",
      "llm.input_messages.0.message.content": "You are helpful.",
      "llm.input_messages.1.message.role": "user",
      "llm.input_messages.1.message.content": "What is the weather in Paris?",
      "llm.output_messages.0.message.role": "assistant",
      "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_1",
      "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
      "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"city":"Paris"}',
      "llm.token_count.prompt": 20,
      "llm.token_count.completion": 5,
      "llm.token_count.prompt_details.cache_read": 4,
      "llm.token_count.prompt_details.cache_write": 2,
      "output.mime_type": "application/json",
      "output.value": JSON.stringify({ id: "chatcmpl-1", model: "gpt-4o-2024-08-06" }),
    };

    it("should convert base, request, usage, and response attributes", () => {
      const genAIAttributes = getGenAIAttributes(chatAttributes);
      expect(genAIAttributes).toMatchObject({
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.conversation.id": "session-1",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.request.temperature": 0.2,
        "gen_ai.request.top_p": 0.9,
        "gen_ai.request.max_tokens": 256,
        "gen_ai.request.seed": 7,
        "gen_ai.request.stream": true,
        "gen_ai.request.stop_sequences": ["END"],
        "gen_ai.output.type": "json",
        "gen_ai.usage.input_tokens": 20,
        "gen_ai.usage.output_tokens": 5,
        "gen_ai.usage.cache_read.input_tokens": 4,
        "gen_ai.usage.cache_creation.input_tokens": 2,
        "gen_ai.response.finish_reasons": ["tool_call"],
        "gen_ai.response.id": "chatcmpl-1",
        "gen_ai.response.model": "gpt-4o-2024-08-06",
      });
      // a choice count of 1 is the default and is not reported
      expect(genAIAttributes).not.toHaveProperty("gen_ai.request.choice.count");
    });

    it("should convert flattened messages into JSON encoded GenAI messages", () => {
      const genAIAttributes = getGenAIAttributes(chatAttributes);
      expect(parseJSONAttribute(genAIAttributes["gen_ai.input.messages"])).toEqual([
        { role: "system", parts: [{ type: "text", content: "You are helpful." }] },
        { role: "user", parts: [{ type: "text", content: "What is the weather in Paris?" }] },
      ]);
      expect(parseJSONAttribute(genAIAttributes["gen_ai.output.messages"])).toEqual([
        {
          role: "assistant",
          parts: [
            { type: "tool_call", id: "call_1", name: "get_weather", arguments: { city: "Paris" } },
          ],
          finish_reason: "tool_call",
        },
      ]);
    });

    it("should convert tool response messages and multimodal contents", () => {
      const genAIAttributes = getGenAIAttributes({
        "openinference.span.kind": "LLM",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.contents.0.message_content.type": "text",
        "llm.input_messages.0.message.contents.0.message_content.text": "Describe these",
        "llm.input_messages.0.message.contents.1.message_content.type": "image",
        "llm.input_messages.0.message.contents.1.message_content.image.image.url":
          "data:image/png;base64,AAAA",
        "llm.input_messages.0.message.contents.2.message_content.type": "image",
        "llm.input_messages.0.message.contents.2.message_content.image.image.url":
          "https://example.com/cat.png",
        "llm.input_messages.1.message.role": "tool",
        "llm.input_messages.1.message.tool_call_id": "call_1",
        "llm.input_messages.1.message.content": '{"temperature":21}',
      });
      expect(parseJSONAttribute(genAIAttributes["gen_ai.input.messages"])).toEqual([
        {
          role: "user",
          parts: [
            { type: "text", content: "Describe these" },
            { type: "blob", modality: "image", mime_type: "image/png", content: "AAAA" },
            { type: "uri", modality: "image", uri: "https://example.com/cat.png" },
          ],
        },
        {
          role: "tool",
          parts: [{ type: "tool_call_response", id: "call_1", response: { temperature: 21 } }],
        },
      ]);
    });

    it("should treat Responses API input_text and output_text contents as text", () => {
      const genAIAttributes = getGenAIAttributes({
        "openinference.span.kind": "LLM",
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.contents.0.message_content.type": "input_text",
        "llm.input_messages.0.message.contents.0.message_content.text": "hello",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.contents.0.message_content.type": "output_text",
        "llm.output_messages.0.message.contents.0.message_content.text": "hi there",
      });
      expect(parseJSONAttribute(genAIAttributes["gen_ai.input.messages"])).toEqual([
        { role: "user", parts: [{ type: "text", content: "hello" }] },
      ]);
      expect(parseJSONAttribute(genAIAttributes["gen_ai.output.messages"])).toEqual([
        {
          role: "assistant",
          parts: [{ type: "text", content: "hi there" }],
          finish_reason: "stop",
        },
      ]);
      expect(genAIAttributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
    });

    it("should order messages by index rather than by attribute order", () => {
      const genAIAttributes = getGenAIAttributes({
        "openinference.span.kind": "LLM",
        "llm.input_messages.10.message.role": "user",
        "llm.input_messages.10.message.content": "last",
        "llm.input_messages.2.message.role": "user",
        "llm.input_messages.2.message.content": "first",
      });
      expect(parseJSONAttribute(genAIAttributes["gen_ai.input.messages"])).toEqual([
        { role: "user", parts: [{ type: "text", content: "first" }] },
        { role: "user", parts: [{ type: "text", content: "last" }] },
      ]);
    });

    it("should prefer an explicit finish reason and normalize it", () => {
      const genAIAttributes = getGenAIAttributes({
        "openinference.span.kind": "LLM",
        "llm.finish_reason": "TOOL_CALLS",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "done",
      });
      expect(genAIAttributes["gen_ai.response.finish_reasons"]).toEqual(["tool_call"]);
      expect(parseJSONAttribute(genAIAttributes["gen_ai.output.messages"])).toEqual([
        {
          role: "assistant",
          parts: [{ type: "text", content: "done" }],
          finish_reason: "tool_call",
        },
      ]);
    });

    it("should default the finish reason of text output to stop", () => {
      const genAIAttributes = getGenAIAttributes({
        "openinference.span.kind": "LLM",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "hello",
      });
      expect(genAIAttributes["gen_ai.response.finish_reasons"]).toEqual(["stop"]);
    });
  });

  describe("text completion spans", () => {
    it("should convert prompts and choices to messages", () => {
      const genAIAttributes = getGenAIAttributes({
        "openinference.span.kind": "LLM",
        "llm.prompts.0.prompt.text": "Once upon a time",
        "llm.choices.0.completion.text": " there was a span.",
      });
      expect(genAIAttributes["gen_ai.operation.name"]).toBe("text_completion");
      expect(parseJSONAttribute(genAIAttributes["gen_ai.input.messages"])).toEqual([
        { role: "user", parts: [{ type: "text", content: "Once upon a time" }] },
      ]);
      expect(parseJSONAttribute(genAIAttributes["gen_ai.output.messages"])).toEqual([
        {
          role: "assistant",
          parts: [{ type: "text", content: " there was a span." }],
          finish_reason: "stop",
        },
      ]);
    });
  });

  describe("operation name", () => {
    it.each([
      ["TOOL", "execute_tool"],
      ["EMBEDDING", "embeddings"],
      ["RETRIEVER", "retrieval"],
      ["AGENT", "invoke_agent"],
      ["LLM", "chat"],
      ["llm", "chat"],
    ])("should map the %s span kind to %s", (spanKind, operationName) => {
      expect(getGenAIAttributes({ "openinference.span.kind": spanKind })).toMatchObject({
        "gen_ai.operation.name": operationName,
      });
    });

    it("should pass through an explicitly set operation name", () => {
      expect(
        getGenAIAttributes({
          "openinference.span.kind": "AGENT",
          "gen_ai.operation.name": "plan",
          "agent.name": "planner",
        }),
      ).toEqual({ "gen_ai.operation.name": "plan", "gen_ai.agent.name": "planner" });
    });
  });

  describe("provider name", () => {
    it.each([
      [{ "llm.provider": "azure", "llm.system": "openai" }, "azure.ai.openai"],
      [{ "llm.provider": "azure", "llm.system": "mistralai" }, "azure.ai.inference"],
      [{ "llm.provider": "aws", "llm.system": "anthropic" }, "aws.bedrock"],
      [{ "llm.provider": "google", "llm.system": "vertexai" }, "gcp.vertex_ai"],
      [{ "llm.system": "vertexai" }, "gcp.vertex_ai"],
      [{ "llm.provider": "google" }, "gcp.gen_ai"],
      [{ "llm.provider": "OpenAI" }, "openai"],
      [{ "llm.system": "anthropic" }, "anthropic"],
      [{ "llm.provider": "xai" }, "x_ai"],
      [{ "llm.provider": "mistralai" }, "mistral_ai"],
      [{ "llm.provider": "fireworks", "llm.system": "meta" }, "meta"],
      [{ "llm.provider": "fireworks" }, "fireworks"],
    ])("should map %j to %s", (attributes, providerName) => {
      expect(getGenAIAttributes(attributes)["gen_ai.provider.name"]).toBe(providerName);
    });

    it("should not emit an empty provider name", () => {
      expect(getGenAIAttributes({ "llm.provider": "" })).toEqual({});
    });
  });

  describe("model names", () => {
    it("should read the request and response models from their own attributes", () => {
      // llm.model_name mirrors the response model, e.g. after a server-side fallback
      expect(
        getGenAIAttributes({
          "openinference.span.kind": "LLM",
          "llm.model_name": "fallback-model",
          "llm.request.model_name": "requested-model",
          "llm.response.model_name": "fallback-model",
          "output.value": "streamed text",
        }),
      ).toEqual({
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": "requested-model",
        "gen_ai.response.model": "fallback-model",
      });
    });
  });

  describe("tool definitions", () => {
    it("should normalize OpenAI and Anthropic tool schemas", () => {
      const genAIAttributes = getGenAIAttributes({
        "openinference.span.kind": "LLM",
        "llm.tools.0.tool.json_schema": JSON.stringify({
          type: "function",
          function: {
            name: "get_weather",
            description: "Gets the weather",
            parameters: { type: "object", properties: { city: { type: "string" } } },
          },
        }),
        "llm.tools.1.tool.json_schema": JSON.stringify({
          name: "search",
          description: "Searches the web",
          input_schema: { type: "object" },
        }),
      });
      expect(parseJSONAttribute(genAIAttributes["gen_ai.tool.definitions"])).toEqual([
        {
          type: "function",
          name: "get_weather",
          description: "Gets the weather",
          parameters: { type: "object", properties: { city: { type: "string" } } },
        },
        {
          type: "function",
          name: "search",
          description: "Searches the web",
          parameters: { type: "object" },
        },
      ]);
    });

    it("should prefer tool.name and tool.description over the json schema", () => {
      const genAIAttributes = getGenAIAttributes({
        "llm.tools.0.tool.name": "renamed",
        "llm.tools.0.tool.description": "Renamed tool",
        "llm.tools.0.tool.json_schema": JSON.stringify({
          type: "function",
          function: { name: "original", description: "Original tool" },
        }),
        "llm.tools.1.tool.name": "schemaless",
      });
      expect(parseJSONAttribute(genAIAttributes["gen_ai.tool.definitions"])).toEqual([
        { type: "function", name: "renamed", description: "Renamed tool" },
        { type: "function", name: "schemaless" },
      ]);
    });
  });

  describe("tool spans", () => {
    it("should convert tool execution attributes", () => {
      expect(
        getGenAIAttributes({
          "openinference.span.kind": "TOOL",
          "tool.name": "get_weather",
          "tool.description": "Gets the weather",
          "tool.id": "call_1",
          "tool.parameters": '{"city":"Paris"}',
          "output.value": "sunny",
        }),
      ).toEqual({
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "get_weather",
        "gen_ai.tool.type": "function",
        "gen_ai.tool.description": "Gets the weather",
        "gen_ai.tool.call.id": "call_1",
        "gen_ai.tool.call.arguments": '{"city":"Paris"}',
        "gen_ai.tool.call.result": "sunny",
      });
    });

    it("should not re-encode JSON arguments and results", () => {
      // a parse and stringify round trip would round these integers
      expect(
        getGenAIAttributes({
          "openinference.span.kind": "TOOL",
          "tool.parameters": '{"user_id": 1234567890123456789}',
          "output.value": "12345678901234567890",
        }),
      ).toMatchObject({
        "gen_ai.tool.call.arguments": '{"user_id": 1234567890123456789}',
        "gen_ai.tool.call.result": "12345678901234567890",
      });
    });

    it("should not write to Object.prototype when unflattening tool parameters", () => {
      const genAIAttributes = getGenAIAttributes({
        "openinference.span.kind": "TOOL",
        "tool.parameters.__proto__.polluted": "yes",
        "tool.parameters.city": "Paris",
      });
      expect(({} as Record<string, unknown>).polluted).toBeUndefined();
      expect(parseJSONAttribute(genAIAttributes["gen_ai.tool.call.arguments"])).toMatchObject({
        city: "Paris",
      });
    });

    it("should not emit tool execution attributes on non-tool spans", () => {
      expect(
        getGenAIAttributes({
          "openinference.span.kind": "CHAIN",
          "tool.name": "get_weather",
          "output.value": "sunny",
        }),
      ).toEqual({});
    });
  });

  describe("retriever spans", () => {
    it("should convert the query and documents", () => {
      const genAIAttributes = getGenAIAttributes({
        "openinference.span.kind": "RETRIEVER",
        "input.value": "what is openinference?",
        "retrieval.documents.0.document.id": "doc-1",
        "retrieval.documents.0.document.content": "OpenInference is ...",
        "retrieval.documents.0.document.score": 0.9,
        "retrieval.documents.0.document.metadata": '{"source":"docs"}',
      });
      expect(genAIAttributes["gen_ai.operation.name"]).toBe("retrieval");
      expect(genAIAttributes["gen_ai.retrieval.query.text"]).toBe("what is openinference?");
      expect(parseJSONAttribute(genAIAttributes["gen_ai.retrieval.documents"])).toEqual([
        {
          id: "doc-1",
          content: "OpenInference is ...",
          score: 0.9,
          metadata: { source: "docs" },
        },
      ]);
    });

    it("should not treat a JSON input as the query text", () => {
      const genAIAttributes = getGenAIAttributes({
        "openinference.span.kind": "RETRIEVER",
        "input.value": '{"query":"hello"}',
        "input.mime_type": "application/json",
      });
      expect(genAIAttributes).not.toHaveProperty("gen_ai.retrieval.query.text");
    });
  });

  describe("embedding spans", () => {
    it("should convert the model, parameters, and dimension count", () => {
      expect(
        getGenAIAttributes({
          "openinference.span.kind": "EMBEDDING",
          "embedding.model_name": "text-embedding-3-small",
          "embedding.invocation_parameters": '{"encoding_format":"float"}',
          "embedding.embeddings.0.embedding.vector": [0.1, 0.2, 0.3],
        }),
      ).toEqual({
        "gen_ai.operation.name": "embeddings",
        "gen_ai.request.model": "text-embedding-3-small",
        "gen_ai.request.encoding_formats": ["float"],
        "gen_ai.embeddings.dimension.count": 3,
      });
    });
  });

  it("should tolerate malformed JSON attribute values", () => {
    expect(
      getGenAIAttributes({
        "openinference.span.kind": "LLM",
        "llm.invocation_parameters": "{not json",
        "llm.tools": "nope",
        "llm.input_messages": "[",
      }),
    ).toEqual({ "gen_ai.operation.name": "chat" });
  });

  describe("provider vocabularies", () => {
    it.each([
      ["tool_calls", "tool_call"],
      ["function_call", "tool_call"],
      ["tool_use", "tool_call"],
      ["end_turn", "stop"],
      ["stop_sequence", "stop"],
      ["max_tokens", "length"],
      ["MAX_TOKENS", "length"],
      ["length", "length"],
      ["refusal", "content_filter"],
      ["content_filtered", "content_filter"],
      ["guardrail_intervened", "content_filter"],
      ["SAFETY", "content_filter"],
      ["pause_turn", "pause_turn"],
    ])("should map the finish reason %s to %s", (providerFinishReason, finishReason) => {
      const genAIAttributes = getGenAIAttributes({
        "openinference.span.kind": "LLM",
        "llm.finish_reason": providerFinishReason,
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "done",
      });
      expect(genAIAttributes["gen_ai.response.finish_reasons"]).toEqual([finishReason]);
      expect(parseJSONAttribute(genAIAttributes["gen_ai.output.messages"])).toEqual([
        {
          role: "assistant",
          parts: [{ type: "text", content: "done" }],
          finish_reason: finishReason,
        },
      ]);
    });

    it("should read camelCase request parameters", () => {
      expect(
        getGenAIAttributes({
          "openinference.span.kind": "LLM",
          "llm.invocation_parameters": JSON.stringify({
            maxTokens: 150,
            temperature: 0.7,
            topP: 0.9,
            topK: 40,
            stopSequences: ["END", "STOP"],
            frequencyPenalty: 0.1,
            presencePenalty: 0.2,
            candidateCount: 2,
          }),
        }),
      ).toEqual({
        "gen_ai.operation.name": "chat",
        "gen_ai.request.max_tokens": 150,
        "gen_ai.request.temperature": 0.7,
        "gen_ai.request.top_p": 0.9,
        "gen_ai.request.top_k": 40,
        "gen_ai.request.stop_sequences": ["END", "STOP"],
        "gen_ai.request.frequency_penalty": 0.1,
        "gen_ai.request.presence_penalty": 0.2,
        "gen_ai.request.choice.count": 2,
      });
    });

    it("should normalize a Bedrock Converse tool definition", () => {
      const genAIAttributes = getGenAIAttributes({
        "openinference.span.kind": "LLM",
        "llm.tools.0.tool.json_schema": JSON.stringify({
          toolSpec: {
            name: "get_weather",
            description: "Get current weather for a location",
            inputSchema: { json: { type: "object", properties: { location: { type: "string" } } } },
          },
        }),
      });
      expect(parseJSONAttribute(genAIAttributes["gen_ai.tool.definitions"])).toEqual([
        {
          type: "function",
          name: "get_weather",
          description: "Get current weather for a location",
          parameters: { type: "object", properties: { location: { type: "string" } } },
        },
      ]);
    });
  });
});

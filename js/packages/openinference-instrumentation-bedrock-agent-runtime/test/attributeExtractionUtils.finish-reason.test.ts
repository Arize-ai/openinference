import {
  LLM_FINISH_REASON,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { getAttributesFromModelInvocationOutput } from "../src/attributes/attributeExtractionUtils";
import { getLLMAttributes } from "../src/attributes/attributeUtils";

describe("model invocation finish reason", () => {
  it.each([
    { body: { stopReason: "end_turn" }, expected: "end_turn" },
    { body: { stopReason: "content_filtered" }, expected: "content_filtered" },
    { body: { stop_reason: "tool_use" }, expected: "tool_use" },
    { body: { stop_reason: "max_tokens" }, expected: "max_tokens" },
    { body: { stop_reason: "length" }, expected: "length" },
    { body: { finish_reason: "COMPLETE" }, expected: "COMPLETE" },
    { body: { finishReason: "stop" }, expected: "stop" },
    { body: { completionReason: "FINISH" }, expected: "FINISH" },
    { body: { results: [{ completionReason: "LENGTH" }] }, expected: "LENGTH" },
    { body: { generations: [{ finish_reason: "MAX_TOKENS" }] }, expected: "MAX_TOKENS" },
    { body: { outputs: [{ stop_reason: "stop" }] }, expected: "stop" },
    {
      body: { choices: [{ finish_reason: "tool_calls" }, { finish_reason: "length" }] },
      expected: "tool_calls",
    },
    { body: { completions: [{ finishReason: "stop" }] }, expected: "stop" },
    { body: { completions: [{ finishReason: { reason: "length" } }] }, expected: "length" },
  ])("preserves the native reason $expected from $body", ({ body, expected }) => {
    const attributes = getAttributesFromModelInvocationOutput({
      rawResponse: { content: JSON.stringify(body) },
    });
    expect(attributes[LLM_FINISH_REASON]).toBe(expected);
  });

  it("accepts an already parsed raw response", () => {
    const attributes = getAttributesFromModelInvocationOutput({
      rawResponse: { content: { stopReason: "end_turn" } },
    });
    expect(attributes[LLM_FINISH_REASON]).toBe("end_turn");
  });

  it.each([
    undefined,
    null,
    "",
    "plain text",
    "{invalid json",
    "null",
    "[]",
    "42",
    JSON.stringify({ stop_reason: null }),
    JSON.stringify({ stopReason: "" }),
    JSON.stringify({ finish_reason: 42 }),
    JSON.stringify({ stopReason: false }),
    JSON.stringify({ stopReason: { reason: "end_turn" } }),
    JSON.stringify({ choices: [] }),
    JSON.stringify({ outputs: [null] }),
    JSON.stringify({ results: "invalid" }),
    JSON.stringify({ choices: [{ finish_reason: null }, { finish_reason: "stop" }] }),
    JSON.stringify({ completions: [{ finishReason: { reason: 42 } }] }),
    JSON.stringify({ content: [{ type: "text", text: "stop_reason: end_turn" }] }),
  ])("omits absent or invalid reasons in raw content %j", (content) => {
    const attributes = getAttributesFromModelInvocationOutput({ rawResponse: { content } });
    expect(attributes).not.toHaveProperty(LLM_FINISH_REASON);
  });

  it("does not infer a reason from trace metadata or a missing raw response", () => {
    expect(getAttributesFromModelInvocationOutput({})).not.toHaveProperty(LLM_FINISH_REASON);
    expect(
      getAttributesFromModelInvocationOutput({
        metadata: { stopReason: "end_turn" },
      }),
    ).not.toHaveProperty(LLM_FINISH_REASON);
  });

  it("preserves text and token usage alongside the finish reason", () => {
    const responseBody = {
      role: "assistant",
      content: [{ type: "text", text: "Hello" }],
      stop_reason: "end_turn",
    };
    const attributes = getAttributesFromModelInvocationOutput({
      rawResponse: {
        content: JSON.stringify(responseBody),
      },
      metadata: { usage: { inputTokens: 2, outputTokens: 3 } },
    });
    expect(attributes[LLM_FINISH_REASON]).toBe("end_turn");
    expect(attributes[SemanticConventions.OUTPUT_VALUE]).toBe(JSON.stringify(responseBody));
    expect(attributes["llm.output_messages.0.message.content"]).toBe("Hello");
    expect(attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT]).toBe(2);
    expect(attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]).toBe(3);
  });

  it("maps an optional finish reason through the LLM attribute builder", () => {
    expect(getLLMAttributes({ finishReason: "tool_use" })[LLM_FINISH_REASON]).toBe("tool_use");
    expect(getLLMAttributes()).not.toHaveProperty(LLM_FINISH_REASON);
    expect(getLLMAttributes({ finishReason: "" })).not.toHaveProperty(LLM_FINISH_REASON);
  });
});

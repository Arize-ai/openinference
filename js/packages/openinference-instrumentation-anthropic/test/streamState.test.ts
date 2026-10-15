import type { BetaRawMessageStreamEvent } from "@anthropic-ai/sdk/resources/beta/messages";
import { describe, expect, it } from "vitest";

import { generateTraceConfig } from "@arizeai/openinference-core";

import {
  applyAnthropicStreamEvent,
  createAnthropicStreamState,
  getAnthropicStreamAttributes,
} from "../src/streamState";
import type { StreamEvent } from "./helpers/sse";
import { messageDelta, messageStart, textStreamEvents } from "./helpers/sse";

function record(events: StreamEvent[]) {
  const state = createAnthropicStreamState();
  for (const event of events) {
    applyAnthropicStreamEvent({ chunk: event as unknown as BetaRawMessageStreamEvent, state });
  }
  return getAnthropicStreamAttributes(
    state,
    { context: { tenant: "review" }, requestToolCalls: { tool_callers: {}, toolset_names: {} } },
    generateTraceConfig({}),
  );
}

describe("Anthropic stream accumulation", () => {
  it("rebuilds a message without changing the events delivered to the caller", () => {
    const events = textStreamEvents(["Hello", ", world"]);
    const original = structuredClone(events);

    const attributes = record(events);

    expect(events).toEqual(original);
    expect(attributes["llm.output_messages.0.message.contents.0.message_content.text"]).toBe(
      "Hello, world",
    );
    expect(JSON.parse(attributes["output.value"] as string)).toMatchObject({
      content: [{ type: "text", text: "Hello, world" }],
      stop_reason: "end_turn",
      usage: { input_tokens: 10, output_tokens: 5 },
    });
    expect(attributes["llm.token_count.total"]).toBe(15);
  });

  it("keeps an interrupted tool's partial arguments on its own block", () => {
    const attributes = record([
      messageStart(),
      {
        type: "content_block_start",
        index: 0,
        content_block: { type: "tool_use", id: "client", name: "weather", input: {} },
      },
      {
        type: "content_block_start",
        index: 1,
        content_block: { type: "server_tool_use", id: "server", name: "web_search", input: {} },
      },
      {
        type: "content_block_delta",
        index: 1,
        delta: { type: "input_json_delta", partial_json: '{"query":"Par' },
      },
    ]);

    expect(
      attributes["llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"],
    ).toBe("{}");
    expect(
      attributes["llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments"],
    ).toBe('{"query":"Par');
  });

  it("replaces compaction content while retaining context metadata", () => {
    const attributes = record([
      messageStart(),
      {
        type: "content_block_start",
        index: 0,
        content_block: { type: "compaction", content: "Old summary" },
      },
      {
        type: "content_block_delta",
        index: 0,
        delta: { type: "compaction_delta", content: "Final summary", encrypted_content: "opaque" },
      },
      messageDelta({
        extra: {
          delta: { stop_reason: "end_turn", stop_details: { type: "refusal", category: "cyber" } },
        },
      }),
    ]);

    expect(attributes["llm.output_messages.0.message.contents.0.message_content.text"]).toBe(
      "Final summary",
    );
    expect(
      attributes["llm.output_messages.0.message.contents.0.message_content.encrypted_content"],
    ).toBe("opaque");
    expect(JSON.parse(attributes.metadata as string)).toEqual({
      tenant: "review",
      stop_details: { type: "refusal", category: "cyber" },
    });
  });
});

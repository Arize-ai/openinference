export type StreamEvent = Record<string, unknown> & { type: string };

/**
 * Serializes stream events as a server-sent event body.
 */
export function toSSE(events: StreamEvent[]): string {
  return events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join("");
}

/**
 * A `fetch` that answers every request with the given events as an SSE stream.
 */
export function sseFetch(events: StreamEvent[]): typeof fetch {
  return async () =>
    new Response(toSSE(events), {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
}

/**
 * A `fetch` whose SSE body delivers the given events and then fails, as a
 * dropped connection would.
 */
export function failingSSEFetch(events: StreamEvent[], error: Error): typeof fetch {
  return async () => {
    const encoder = new TextEncoder();
    let sent = false;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (!sent) {
          sent = true;
          controller.enqueue(encoder.encode(toSSE(events)));
          return;
        }
        controller.error(error);
      },
    });
    return new Response(body, {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
  };
}

/**
 * A `fetch` that answers every request with the given JSON body.
 */
export function jsonFetch(body: unknown): typeof fetch {
  return async () =>
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
}

export function messageStart(overrides: Record<string, unknown> = {}): StreamEvent {
  return {
    type: "message_start",
    message: {
      id: "msg_stream",
      type: "message",
      role: "assistant",
      model: "claude-sonnet-4-6",
      content: [],
      stop_reason: null,
      stop_sequence: null,
      stop_details: null,
      usage: { input_tokens: 10, output_tokens: 1 },
      ...overrides,
    },
  };
}

export function messageDelta({
  stopReason = "end_turn",
  usage = { output_tokens: 5 },
  extra = {},
}: {
  stopReason?: string;
  usage?: Record<string, unknown>;
  extra?: Record<string, unknown>;
} = {}): StreamEvent {
  return {
    type: "message_delta",
    delta: { stop_reason: stopReason, stop_sequence: null, stop_details: null, container: null },
    usage,
    ...extra,
  };
}

/**
 * Events of a stream that answers with one text block.
 */
export function textStreamEvents(texts: string[] = ["Hello", ", world"]): StreamEvent[] {
  return [
    messageStart(),
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    ...texts.map((text) => ({
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text },
    })),
    { type: "content_block_stop", index: 0 },
    messageDelta(),
    { type: "message_stop" },
  ];
}

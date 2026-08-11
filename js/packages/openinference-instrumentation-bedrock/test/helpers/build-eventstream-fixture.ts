import zlib from "zlib";

type EventHeader = [name: string, value: string];
type StreamEvent = [eventType: string, payload: Record<string, unknown>];

function crc32(buf: Buffer): number {
  return zlib.crc32(buf) >>> 0;
}

function encodeHeaders(headers: EventHeader[]): Buffer {
  const parts: Buffer[] = [];
  for (const [name, value] of headers) {
    const nameBuf = Buffer.from(name, "utf8");
    const valueBuf = Buffer.from(value, "utf8");
    const header = Buffer.concat([
      Buffer.from([nameBuf.length]),
      nameBuf,
      Buffer.from([7]), // header value type: string
      Buffer.from([(valueBuf.length >> 8) & 0xff, valueBuf.length & 0xff]),
      valueBuf,
    ]);
    parts.push(header);
  }
  return Buffer.concat(parts);
}

function encodeMessage(headers: EventHeader[], payloadObj: Record<string, unknown>): Buffer {
  const headersBuf = encodeHeaders(headers);
  const payloadBuf = Buffer.from(JSON.stringify(payloadObj), "utf8");
  const totalLength = 4 + 4 + 4 + headersBuf.length + payloadBuf.length + 4;

  const preludeNoCrc = Buffer.alloc(8);
  preludeNoCrc.writeUInt32BE(totalLength, 0);
  preludeNoCrc.writeUInt32BE(headersBuf.length, 4);
  const preludeCrc = Buffer.alloc(4);
  preludeCrc.writeUInt32BE(crc32(preludeNoCrc), 0);

  const withoutMessageCrc = Buffer.concat([preludeNoCrc, preludeCrc, headersBuf, payloadBuf]);
  const messageCrc = Buffer.alloc(4);
  messageCrc.writeUInt32BE(crc32(withoutMessageCrc), 0);

  return Buffer.concat([withoutMessageCrc, messageCrc]);
}

function eventMessage(eventType: string, payloadObj: Record<string, unknown>): Buffer {
  return encodeMessage(
    [
      [":event-type", eventType],
      [":content-type", "application/json"],
      [":message-type", "event"],
    ],
    payloadObj,
  );
}

export function buildStream(events: StreamEvent[]): Buffer {
  return Buffer.concat(events.map(([eventType, payload]) => eventMessage(eventType, payload)));
}

function main() {
  const scenario = process.argv[2];
  if (scenario === "reasoning") {
    const redactedBytes = Buffer.from("opaque-encrypted-reasoning-bytes");
    // Split into two chunks to exercise multi-delta concatenation for the same block index.
    const redactedChunk1 = redactedBytes.subarray(0, 12).toString("base64");
    const redactedChunk2 = redactedBytes.subarray(12).toString("base64");
    const stream = buildStream([
      ["messageStart", { p: "abcde", role: "assistant" }],
      [
        "contentBlockDelta",
        {
          contentBlockIndex: 0,
          delta: { reasoningContent: { text: "Let me think about this. " } },
          p: "abcde",
        },
      ],
      [
        "contentBlockDelta",
        {
          contentBlockIndex: 0,
          delta: { reasoningContent: { text: "6 * 7 = 42." } },
          p: "abcde",
        },
      ],
      [
        "contentBlockDelta",
        {
          contentBlockIndex: 0,
          delta: { reasoningContent: { signature: "test-signature-token" } },
          p: "abcde",
        },
      ],
      ["contentBlockStop", { contentBlockIndex: 0, p: "abcde" }],
      [
        "contentBlockDelta",
        {
          contentBlockIndex: 1,
          delta: { text: "The answer is 42." },
          p: "abcde",
        },
      ],
      ["contentBlockStop", { contentBlockIndex: 1, p: "abcde" }],
      [
        "contentBlockDelta",
        {
          contentBlockIndex: 2,
          delta: { reasoningContent: { redactedContent: redactedChunk1 } },
          p: "abcde",
        },
      ],
      [
        "contentBlockDelta",
        {
          contentBlockIndex: 2,
          delta: { reasoningContent: { redactedContent: redactedChunk2 } },
          p: "abcde",
        },
      ],
      ["contentBlockStop", { contentBlockIndex: 2, p: "abcde" }],
      ["messageStop", { p: "abcde", stopReason: "end_turn" }],
      [
        "metadata",
        {
          metrics: { latencyMs: 500 },
          p: "abcde",
          usage: { inputTokens: 10, outputTokens: 20, totalTokens: 30 },
        },
      ],
    ]);
    process.stdout.write(stream.toString("hex"));
  }
}

main();

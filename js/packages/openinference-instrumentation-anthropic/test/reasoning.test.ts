import Anthropic from "@anthropic-ai/sdk";
import type { Attributes } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

import {
  LLMProvider,
  LLMSystem,
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { AnthropicInstrumentation } from "../src/instrumentation";
import { vcrFetch, vcrFetchSequence } from "./helpers/vcr";

const {
  OPENINFERENCE_SPAN_KIND,
  LLM_PROVIDER,
  LLM_SYSTEM,
  LLM_MODEL_NAME,
  LLM_INVOCATION_PARAMETERS,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_TOTAL,
  INPUT_VALUE,
  INPUT_MIME_TYPE,
  OUTPUT_VALUE,
  OUTPUT_MIME_TYPE,
  LLM_INPUT_MESSAGES,
  LLM_OUTPUT_MESSAGES,
  MESSAGE_ROLE,
  MESSAGE_CONTENT,
  MESSAGE_CONTENTS,
  MESSAGE_CONTENT_TYPE,
  MESSAGE_CONTENT_TEXT,
  MESSAGE_CONTENT_SIGNATURE,
  MESSAGE_CONTENT_DATA,
  MESSAGE_CONTENT_ID,
} = SemanticConventions;

const memoryExporter = new InMemorySpanExporter();

async function waitForSpans(count: number) {
  for (let i = 0; i < 50; i++) {
    if (memoryExporter.getFinishedSpans().length >= count) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

/**
 * Pops `key` off a mutable copy of the span attributes and returns its
 * value, so that tests can assert every attribute is accounted for and
 * finish with `expect(attributes).toEqual({})`.
 */
function pop(attributes: Attributes, key: string): unknown {
  const value = attributes[key];
  delete attributes[key];
  return value;
}

describe("AnthropicInstrumentation - reasoning content", () => {
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();
  const instrumentation = new AnthropicInstrumentation({ tracerProvider });
  instrumentation.disable();
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = Anthropic;

  beforeAll(() => {
    instrumentation.enable();
  });

  beforeEach(() => {
    memoryExporter.reset();
  });

  afterEach(() => {
    instrumentation.disable();
    instrumentation.enable();
  });

  it("captures thinking blocks as reasoning content for non-streaming responses", async () => {
    const client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY ?? "fake-api-key",
      fetch: vcrFetch("thinking-non-streaming"),
    });

    const requestBody = {
      model: "claude-sonnet-4-6",
      max_tokens: 2048,
      thinking: { type: "enabled" as const, budget_tokens: 1024 },
      messages: [
        {
          role: "user" as const,
          content: "What is 27 * 453? Think it through step by step.",
        },
      ],
    };

    await client.messages.create(requestBody);

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    expect(spans[0].name).toBe("Anthropic Messages");
    const attributes = { ...spans[0].attributes };

    // Span kind / system / provider
    expect(pop(attributes, OPENINFERENCE_SPAN_KIND)).toBe(OpenInferenceSpanKind.LLM);
    expect(pop(attributes, LLM_SYSTEM)).toBe(LLMSystem.ANTHROPIC);
    expect(pop(attributes, LLM_PROVIDER)).toBe(LLMProvider.ANTHROPIC);
    expect(pop(attributes, LLM_MODEL_NAME)).toBe("claude-sonnet-4-6");

    // Input value / invocation params
    const inputValue = pop(attributes, INPUT_VALUE);
    expect(inputValue).toBe(JSON.stringify(requestBody));
    expect(pop(attributes, INPUT_MIME_TYPE)).toBe(MimeType.JSON);
    const { messages: _messages, ...invocationParameters } = requestBody;
    expect(pop(attributes, LLM_INVOCATION_PARAMETERS)).toBe(JSON.stringify(invocationParameters));

    // Input messages
    expect(pop(attributes, `${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`)).toBe("user");
    expect(pop(attributes, `${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`)).toBe(
      "What is 27 * 453? Think it through step by step.",
    );

    // Output value (full message JSON) / mime type
    const outputValue = pop(attributes, OUTPUT_VALUE);
    expect(outputValue).toEqual(expect.any(String));
    expect(outputValue as string).toMatch(
      /^\{"model":"claude-sonnet-4-6","id":"msg_01K7NRDr9CpvBE22JZtFy7ss"/,
    );
    expect(pop(attributes, OUTPUT_MIME_TYPE)).toBe(MimeType.JSON);

    // Output message: role + reasoning content + text content
    expect(pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_ROLE}`)).toBe("assistant");

    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`),
    ).toBe("reasoning");
    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TEXT}`),
    ).toBe(
      "27 * 453\n\n27 * 400 = 10,800\n27 * 50 = 1,350\n27 * 3 = 81\n\n10,800 + 1,350 + 81 = 12,231",
    );
    const signature = pop(
      attributes,
      `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_SIGNATURE}`,
    );
    expect(signature).toEqual(expect.any(String));
    expect(signature as string).toMatch(/^EpwCCmUIDhgCKkDtrdu6/);

    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TYPE}`),
    ).toBe("text");
    const textContent = pop(
      attributes,
      `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TEXT}`,
    );
    expect(textContent).toEqual(expect.any(String));
    expect(textContent as string).toMatch(/^## Solving 27 × 453/);

    // Usage
    expect(pop(attributes, LLM_TOKEN_COUNT_PROMPT)).toBe(52);
    expect(pop(attributes, LLM_TOKEN_COUNT_COMPLETION)).toBe(202);
    expect(pop(attributes, LLM_TOKEN_COUNT_TOTAL)).toBe(254);

    // No leftover attributes
    expect(attributes).toEqual({});
  });

  it("captures thinking blocks as reasoning content for streaming responses", async () => {
    const client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY ?? "fake-api-key",
      fetch: vcrFetch("thinking-streaming"),
    });

    const requestBody = {
      model: "claude-sonnet-4-6",
      max_tokens: 2048,
      thinking: { type: "enabled" as const, budget_tokens: 1024 },
      messages: [
        {
          role: "user" as const,
          content: "What is 27 * 453? Think it through step by step.",
        },
      ],
      stream: true as const,
    };

    const stream = await client.messages.create(requestBody);

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    for await (const _chunk of stream) {
      // drain the stream
    }

    await waitForSpans(1);

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    expect(spans[0].name).toBe("Anthropic Messages");
    const attributes = { ...spans[0].attributes };

    // Span kind / system / provider
    expect(pop(attributes, OPENINFERENCE_SPAN_KIND)).toBe(OpenInferenceSpanKind.LLM);
    expect(pop(attributes, LLM_SYSTEM)).toBe(LLMSystem.ANTHROPIC);
    expect(pop(attributes, LLM_PROVIDER)).toBe(LLMProvider.ANTHROPIC);
    expect(pop(attributes, LLM_MODEL_NAME)).toBe("claude-sonnet-4-6");

    // Input value / invocation params
    expect(pop(attributes, INPUT_VALUE)).toBe(JSON.stringify(requestBody));
    expect(pop(attributes, INPUT_MIME_TYPE)).toBe(MimeType.JSON);
    const { messages: _messages, ...invocationParameters } = requestBody;
    expect(pop(attributes, LLM_INVOCATION_PARAMETERS)).toBe(JSON.stringify(invocationParameters));

    // Input messages
    expect(pop(attributes, `${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`)).toBe("user");
    expect(pop(attributes, `${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`)).toBe(
      "What is 27 * 453? Think it through step by step.",
    );

    // Output value is the concatenated visible text (not the full JSON message) for streams
    const outputValue = pop(attributes, OUTPUT_VALUE);
    expect(outputValue).toEqual(expect.any(String));
    expect(outputValue as string).toMatch(/^## Solving 27 × 453/);
    expect(pop(attributes, OUTPUT_MIME_TYPE)).toBe(MimeType.TEXT);

    // Output message: role + reasoning content + text content
    expect(pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_ROLE}`)).toBe("assistant");

    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`),
    ).toBe("reasoning");
    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TEXT}`),
    ).toBe(
      "27 * 453\n\n27 * 400 = 10,800\n27 * 50 = 1,350\n27 * 3 = 81\n\n10,800 + 1,350 + 81 = 12,231",
    );
    const signature = pop(
      attributes,
      `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_SIGNATURE}`,
    );
    expect(signature).toEqual(expect.any(String));
    expect(signature as string).toMatch(/^EpwCCmUIDhgCKkDtrdu6/);

    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TYPE}`),
    ).toBe("text");
    const textContent = pop(
      attributes,
      `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TEXT}`,
    );
    expect(textContent).toEqual(expect.any(String));
    expect(textContent as string).toMatch(/^## Solving 27 × 453/);

    expect(pop(attributes, LLM_TOKEN_COUNT_PROMPT)).toBe(52);
    expect(pop(attributes, LLM_TOKEN_COUNT_COMPLETION)).toBe(230);
    expect(pop(attributes, LLM_TOKEN_COUNT_TOTAL)).toBe(282);

    expect(attributes).toEqual({});
  });

  it("captures redacted thinking blocks as reasoning content for non-streaming responses", async () => {
    const client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY ?? "fake-api-key",
      fetch: vcrFetch("redacted-thinking-non-streaming"),
    });

    const requestBody = {
      model: "claude-sonnet-4-6",
      max_tokens: 2048,
      thinking: { type: "enabled" as const, budget_tokens: 1024 },
      messages: [
        {
          role: "user" as const,
          content: "What is the answer to life, the universe, and everything?",
        },
      ],
    };

    await client.messages.create(requestBody);

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    expect(spans[0].name).toBe("Anthropic Messages");
    const attributes = { ...spans[0].attributes };

    // Span kind / system / provider
    expect(pop(attributes, OPENINFERENCE_SPAN_KIND)).toBe(OpenInferenceSpanKind.LLM);
    expect(pop(attributes, LLM_SYSTEM)).toBe(LLMSystem.ANTHROPIC);
    expect(pop(attributes, LLM_PROVIDER)).toBe(LLMProvider.ANTHROPIC);
    expect(pop(attributes, LLM_MODEL_NAME)).toBe("claude-sonnet-4-6");

    // Input value / invocation params
    expect(pop(attributes, INPUT_VALUE)).toBe(JSON.stringify(requestBody));
    expect(pop(attributes, INPUT_MIME_TYPE)).toBe(MimeType.JSON);
    const { messages: _messages, ...invocationParameters } = requestBody;
    expect(pop(attributes, LLM_INVOCATION_PARAMETERS)).toBe(JSON.stringify(invocationParameters));

    // Input messages
    expect(pop(attributes, `${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`)).toBe("user");
    expect(pop(attributes, `${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`)).toBe(
      "What is the answer to life, the universe, and everything?",
    );

    // Output value (full message JSON) / mime type
    const outputValue = pop(attributes, OUTPUT_VALUE);
    expect(outputValue).toEqual(expect.any(String));
    expect(outputValue as string).toMatch(
      /^\{"model":"claude-sonnet-4-6","id":"msg_01RedactedThinkingTest"/,
    );
    expect(pop(attributes, OUTPUT_MIME_TYPE)).toBe(MimeType.JSON);

    // Output message: role + redacted reasoning content + text content
    expect(pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_ROLE}`)).toBe("assistant");

    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`),
    ).toBe("reasoning");
    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_DATA}`),
    ).toBe("EmwKAhgBEgy3vOSpZ1m4FAKLm0caDOJ8K6kGz4ANRedactedDataExample==");
    // Redacted thinking blocks have no visible text or id
    expect(
      attributes[`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TEXT}`],
    ).toBeUndefined();
    expect(
      attributes[`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_ID}`],
    ).toBeUndefined();

    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TYPE}`),
    ).toBe("text");
    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TEXT}`),
    ).toBe("I can't share the details of how I solved this, but the answer is 42.");

    // Usage
    expect(pop(attributes, LLM_TOKEN_COUNT_PROMPT)).toBe(30);
    expect(pop(attributes, LLM_TOKEN_COUNT_COMPLETION)).toBe(20);
    expect(pop(attributes, LLM_TOKEN_COUNT_TOTAL)).toBe(50);

    // No leftover attributes
    expect(attributes).toEqual({});
  });

  it("captures redacted thinking blocks as reasoning content for streaming responses", async () => {
    const client = new Anthropic({
      apiKey: process.env.ANTHROPIC_API_KEY ?? "fake-api-key",
      fetch: vcrFetch("redacted-thinking-streaming"),
    });

    const requestBody = {
      model: "claude-sonnet-4-6",
      max_tokens: 2048,
      thinking: { type: "enabled" as const, budget_tokens: 1024 },
      messages: [
        {
          role: "user" as const,
          content: "What is the answer to life, the universe, and everything?",
        },
      ],
      stream: true as const,
    };

    const stream = await client.messages.create(requestBody);

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    for await (const _chunk of stream) {
      // drain the stream
    }

    await waitForSpans(1);

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    expect(spans[0].name).toBe("Anthropic Messages");
    const attributes = { ...spans[0].attributes };

    // Span kind / system / provider
    expect(pop(attributes, OPENINFERENCE_SPAN_KIND)).toBe(OpenInferenceSpanKind.LLM);
    expect(pop(attributes, LLM_SYSTEM)).toBe(LLMSystem.ANTHROPIC);
    expect(pop(attributes, LLM_PROVIDER)).toBe(LLMProvider.ANTHROPIC);
    expect(pop(attributes, LLM_MODEL_NAME)).toBe("claude-sonnet-4-6");

    // Input value / invocation params
    expect(pop(attributes, INPUT_VALUE)).toBe(JSON.stringify(requestBody));
    expect(pop(attributes, INPUT_MIME_TYPE)).toBe(MimeType.JSON);
    const { messages: _messages, ...invocationParameters } = requestBody;
    expect(pop(attributes, LLM_INVOCATION_PARAMETERS)).toBe(JSON.stringify(invocationParameters));

    // Input messages
    expect(pop(attributes, `${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`)).toBe("user");
    expect(pop(attributes, `${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`)).toBe(
      "What is the answer to life, the universe, and everything?",
    );

    // Output value is the concatenated visible text (not the full JSON message) for streams
    const outputValue = pop(attributes, OUTPUT_VALUE);
    expect(outputValue).toBe("I can't share the details, but the answer is 42.");
    expect(pop(attributes, OUTPUT_MIME_TYPE)).toBe(MimeType.TEXT);

    // Output message: role + redacted reasoning content + text content
    expect(pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_ROLE}`)).toBe("assistant");

    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`),
    ).toBe("reasoning");
    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_DATA}`),
    ).toBe("EmwKAhgBEgy3vOSpZ1m4FAKLm0caDOJ8K6kGz4ANRedactedDataExample==");
    expect(
      attributes[`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TEXT}`],
    ).toBeUndefined();
    expect(
      attributes[`${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_ID}`],
    ).toBeUndefined();

    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TYPE}`),
    ).toBe("text");
    expect(
      pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TEXT}`),
    ).toBe("I can't share the details, but the answer is 42.");

    // Usage: prompt count from message_start, completion count updated from message_delta
    expect(pop(attributes, LLM_TOKEN_COUNT_PROMPT)).toBe(30);
    expect(pop(attributes, LLM_TOKEN_COUNT_COMPLETION)).toBe(20);
    expect(pop(attributes, LLM_TOKEN_COUNT_TOTAL)).toBe(50);

    expect(attributes).toEqual({});
  });

  it(
    "preserves thinking blocks (with signature) as reasoning content in input messages on follow-up turns",
    { timeout: 30000 },
    async () => {
      const client = new Anthropic({
        apiKey: process.env.ANTHROPIC_API_KEY ?? "fake-api-key",
        fetch: vcrFetchSequence("thinking-multi-turn"),
      });

      const messages: Anthropic.Messages.MessageParam[] = [
        {
          role: "user",
          content: "What is 27 * 453? Think it through step by step.",
        },
      ];

      const firstResponse = await client.messages.create({
        model: "claude-sonnet-4-6",
        max_tokens: 2048,
        thinking: { type: "enabled", budget_tokens: 1024 },
        messages,
      });

      messages.push({ role: "assistant", content: firstResponse.content });
      messages.push({
        role: "user",
        content: "Now divide that result by 9 and explain your reasoning.",
      });

      await client.messages.create({
        model: "claude-sonnet-4-6",
        max_tokens: 2048,
        thinking: { type: "enabled", budget_tokens: 1024 },
        messages,
      });

      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(2);
      const attributes = { ...spans[1].attributes };

      // Span kind / system / provider / model
      expect(pop(attributes, OPENINFERENCE_SPAN_KIND)).toBe(OpenInferenceSpanKind.LLM);
      expect(pop(attributes, LLM_SYSTEM)).toBe(LLMSystem.ANTHROPIC);
      expect(pop(attributes, LLM_PROVIDER)).toBe(LLMProvider.ANTHROPIC);
      expect(pop(attributes, LLM_MODEL_NAME)).toBe("claude-sonnet-4-6");

      // Input value / invocation params
      const inputValue = pop(attributes, INPUT_VALUE);
      expect(inputValue).toEqual(expect.any(String));
      expect(inputValue as string).toMatch(/^\{"model":"claude-sonnet-4-6"/);
      expect(pop(attributes, INPUT_MIME_TYPE)).toBe(MimeType.JSON);
      expect(pop(attributes, LLM_INVOCATION_PARAMETERS)).toBe(
        JSON.stringify({
          model: "claude-sonnet-4-6",
          max_tokens: 2048,
          thinking: { type: "enabled", budget_tokens: 1024 },
        }),
      );

      // Input message 0: original user question
      expect(pop(attributes, `${LLM_INPUT_MESSAGES}.0.${MESSAGE_ROLE}`)).toBe("user");
      expect(pop(attributes, `${LLM_INPUT_MESSAGES}.0.${MESSAGE_CONTENT}`)).toBe(
        "What is 27 * 453? Think it through step by step.",
      );

      // Input message 1: assistant turn from the first response, preserved with
      // its reasoning (thinking) block and signature
      expect(pop(attributes, `${LLM_INPUT_MESSAGES}.1.${MESSAGE_ROLE}`)).toBe("assistant");
      expect(
        pop(attributes, `${LLM_INPUT_MESSAGES}.1.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`),
      ).toBe("reasoning");
      expect(
        pop(attributes, `${LLM_INPUT_MESSAGES}.1.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TEXT}`),
      ).toBe(firstResponse.content[0].type === "thinking" ? firstResponse.content[0].thinking : "");
      const inputSignature = pop(
        attributes,
        `${LLM_INPUT_MESSAGES}.1.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_SIGNATURE}`,
      );
      expect(inputSignature).toEqual(expect.any(String));
      expect(inputSignature as string).toMatch(/^EpwCCmUIDhgCKkDtrdu6/);

      expect(
        pop(attributes, `${LLM_INPUT_MESSAGES}.1.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TYPE}`),
      ).toBe("text");
      const inputAssistantText = pop(
        attributes,
        `${LLM_INPUT_MESSAGES}.1.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TEXT}`,
      );
      expect(inputAssistantText).toEqual(expect.any(String));
      expect(inputAssistantText as string).toMatch(/^## Solving 27 × 453/);

      // Input message 2: follow-up user question
      expect(pop(attributes, `${LLM_INPUT_MESSAGES}.2.${MESSAGE_ROLE}`)).toBe("user");
      expect(pop(attributes, `${LLM_INPUT_MESSAGES}.2.${MESSAGE_CONTENT}`)).toBe(
        "Now divide that result by 9 and explain your reasoning.",
      );

      // Output value / mime type
      const outputValue = pop(attributes, OUTPUT_VALUE);
      expect(outputValue).toEqual(expect.any(String));
      expect(outputValue as string).toMatch(
        /^\{"model":"claude-sonnet-4-6","id":"msg_01HkNaEbsuALpp4oY7QWTysY"/,
      );
      expect(pop(attributes, OUTPUT_MIME_TYPE)).toBe(MimeType.JSON);

      // Output message: role + reasoning content + text content
      expect(pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_ROLE}`)).toBe("assistant");

      expect(
        pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TYPE}`),
      ).toBe("reasoning");
      const outputThinking = pop(
        attributes,
        `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_TEXT}`,
      );
      expect(outputThinking).toEqual(expect.any(String));
      expect(outputThinking as string).toMatch(/^12,231 \/ 9/);
      const outputSignature = pop(
        attributes,
        `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.0.${MESSAGE_CONTENT_SIGNATURE}`,
      );
      expect(outputSignature).toEqual(expect.any(String));
      expect(outputSignature as string).toMatch(/^EooECmUIDhgCKkALP1t8HA/);

      expect(
        pop(attributes, `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TYPE}`),
      ).toBe("text");
      const outputText = pop(
        attributes,
        `${LLM_OUTPUT_MESSAGES}.0.${MESSAGE_CONTENTS}.1.${MESSAGE_CONTENT_TEXT}`,
      );
      expect(outputText).toEqual(expect.any(String));
      expect(outputText as string).toMatch(/^## Dividing 12,231 by 9/);

      // Usage
      expect(pop(attributes, LLM_TOKEN_COUNT_PROMPT)).toEqual(expect.any(Number));
      expect(pop(attributes, LLM_TOKEN_COUNT_COMPLETION)).toEqual(expect.any(Number));
      expect(pop(attributes, LLM_TOKEN_COUNT_TOTAL)).toEqual(expect.any(Number));

      // No leftover attributes
      expect(attributes).toEqual({});
    },
  );
});

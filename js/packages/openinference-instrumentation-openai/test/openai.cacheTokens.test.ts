import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import OpenAI, { APIPromise } from "openai";
import { Stream } from "openai/streaming";
import { vi } from "vitest";

import {
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
  LLM_TOKEN_COUNT_TOTAL,
} from "@arizeai/openinference-semantic-conventions";

import { OpenAIInstrumentation } from "../src";
import type { CacheTokenDetails } from "./fixtures/realCacheTokenResponses";
import { realCacheTokenResponses } from "./fixtures/realCacheTokenResponses";

const memoryExporter = new InMemorySpanExporter();

/**
 * Assert that two consecutive spans replay a cold cache write followed by a warm
 * cache read, and that each span's cache counts pass through from the recorded
 * usage unchanged.
 */
function expectColdWriteThenWarmRead({
  cacheWrite,
  cacheRead,
}: {
  cacheWrite: CacheTokenDetails;
  cacheRead: CacheTokenDetails;
}) {
  const spans = memoryExporter.getFinishedSpans();
  expect(spans).toHaveLength(2);
  const [writeSpan, readSpan] = spans;
  const counts = (span: (typeof spans)[number]) => ({
    prompt: span.attributes[LLM_TOKEN_COUNT_PROMPT] as number,
    read: span.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] as number,
    write: span.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] as number,
  });
  const write = counts(writeSpan);
  const read = counts(readSpan);

  // Cold call: nothing served from cache, the prompt is written to it.
  expect(write.read).toBe(cacheWrite.cached_tokens);
  expect(write.write).toBe(cacheWrite.cache_write_tokens);
  expect(write.write).toBeGreaterThan(1024);

  // Warm call: the shared prefix is read back instead of written again.
  expect(read.read).toBe(cacheRead.cached_tokens);
  expect(read.write).toBe(cacheRead.cache_write_tokens);
  expect(read.read).toBeGreaterThan(1024);
  expect(read.write).toBeLessThan(write.write);

  // Cache read and write never exceed the prompt they describe.
  for (const span of [write, read]) {
    expect(span.read + span.write).toBeLessThanOrEqual(span.prompt);
  }
}

/**
 * These tests replay real OpenAI responses (see the fixture for provenance) so the
 * cache token attributes are asserted against payloads the API actually returned,
 * including the `cache_write_tokens` field that the OpenAI SDK does not yet type.
 */
describe("OpenAIInstrumentation - real prompt cache usage", () => {
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();
  const instrumentation = new OpenAIInstrumentation();
  instrumentation.disable();
  let openai: OpenAI;

  instrumentation.setTracerProvider(tracerProvider);
  // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
  instrumentation._modules[0].moduleExports = OpenAI;

  beforeAll(() => {
    instrumentation.enable();
    openai = new OpenAI({ apiKey: "fake-api-key" });
  });
  afterAll(() => {
    instrumentation.disable();
  });
  beforeEach(() => {
    memoryExporter.reset();
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it.each([
    ["zero", 0, 0],
    ["missing", undefined, undefined],
    ["null", null, undefined],
  ])(
    "preserves zero and omits unavailable cache writes (%s)",
    async (_, cacheWriteTokens, expected) => {
      const recorded = realCacheTokenResponses.chatCompletionsLuna.cacheRead;
      vi.spyOn(openai, "post").mockImplementation(
        // @ts-expect-error mock the transport response, including fields absent from older SDK types
        async () => ({
          ...recorded,
          usage: {
            ...recorded.usage,
            prompt_tokens_details: { cached_tokens: 7, cache_write_tokens: cacheWriteTokens },
          },
        }),
      );
      await openai.chat.completions.create({
        model: "gpt-5.6-luna",
        messages: [{ role: "user", content: "Hello" }],
      });
      const [span] = memoryExporter.getFinishedSpans();
      expect(span.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]).toBe(7);
      expect(span.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]).toBe(expected);
    },
  );

  it("records usage from the final usage-only chunk of a chat completions stream", async () => {
    vi.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        const iterator = () =>
          (async function* () {
            yield { choices: [{ delta: { content: "This is " } }], usage: null };
            yield { choices: [{ delta: { content: "a test." } }], usage: null };
            yield { choices: [{ delta: {}, finish_reason: "stop" }], usage: null };
            // `stream_options.include_usage` adds one last chunk with no choices.
            yield {
              choices: [],
              usage: {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_tokens_details: { cached_tokens: 6, cache_write_tokens: 4 },
              },
            };
          })();
        return new Stream(iterator, new AbortController());
      },
    );

    const stream = await openai.chat.completions.create({
      model: "gpt-5.6-luna",
      messages: [{ role: "user", content: "Say this is a test" }],
      stream: true,
      stream_options: { include_usage: true },
    });
    let response = "";
    for await (const chunk of stream) {
      response += chunk.choices[0]?.delta.content ?? "";
    }
    expect(response).toBe("This is a test.");

    const [span] = memoryExporter.getFinishedSpans();
    expect(span.attributes).toMatchObject({
      [LLM_TOKEN_COUNT_PROMPT]: 10,
      [LLM_TOKEN_COUNT_COMPLETION]: 5,
      [LLM_TOKEN_COUNT_TOTAL]: 15,
      [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]: 6,
      [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]: 4,
    });
  });

  const chatCompletionCases = [
    ["gpt-5.6-luna", realCacheTokenResponses.chatCompletionsLuna],
    ["gpt-5.6-terra", realCacheTokenResponses.chatCompletionsTerra],
  ] as const;

  it.each(chatCompletionCases)(
    "records cache write then cache read token counts for %s chat completions",
    async (model, recorded) => {
      const responses = [recorded.cacheWrite, recorded.cacheRead];
      vi.spyOn(openai, "post").mockImplementation(
        // @ts-expect-error the response type is not correct - this is just for testing
        async (): Promise<unknown> => responses.shift(),
      );

      for (const question of ["Write me a haiku.", "Write me a sonnet."]) {
        await openai.chat.completions.create({
          messages: [
            { role: "system", content: "<cacheable prefix>" },
            { role: "user", content: question },
          ],
          model,
        });
      }

      expectColdWriteThenWarmRead({
        cacheWrite: recorded.cacheWrite.usage.prompt_tokens_details,
        cacheRead: recorded.cacheRead.usage.prompt_tokens_details,
      });
    },
  );

  const responsesCases = [
    ["gpt-5.6-luna", realCacheTokenResponses.responsesLuna],
    ["gpt-5.6-terra", realCacheTokenResponses.responsesTerra],
  ] as const;

  it.each(responsesCases)(
    "records cache write then cache read token counts for %s responses",
    async (model, recorded) => {
      const responses = [recorded.cacheWrite, recorded.cacheRead];
      vi.spyOn(openai, "post").mockImplementation(() => {
        const next = responses.shift();
        return new APIPromise(
          new OpenAI({ apiKey: "fake-api-key" }),
          new Promise((resolve) => {
            resolve({
              response: new Response(),
              // @ts-expect-error the response type is not correct - this is just for testing
              options: {},
              controller: new AbortController(),
            });
          }),
          () => next,
        );
      });

      for (const question of ["Write me a haiku.", "Write me a sonnet."]) {
        await openai.responses.create({
          model,
          instructions: "<cacheable prefix>",
          input: question,
        });
      }

      expectColdWriteThenWarmRead({
        cacheWrite: recorded.cacheWrite.usage.input_tokens_details,
        cacheRead: recorded.cacheRead.usage.input_tokens_details,
      });
    },
  );
});

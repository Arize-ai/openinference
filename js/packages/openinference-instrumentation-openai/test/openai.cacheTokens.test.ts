import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import OpenAI, { APIPromise } from "openai";
import { vi } from "vitest";

import {
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
} from "@arizeai/openinference-semantic-conventions";

import { OpenAIInstrumentation } from "../src";
import { realCacheTokenResponses } from "./fixtures/realCacheTokenResponses";

const memoryExporter = new InMemorySpanExporter();

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

      const [writeSpan, readSpan] = memoryExporter.getFinishedSpans();
      expect(memoryExporter.getFinishedSpans()).toHaveLength(2);

      // Cold call: nothing served from cache, the prompt is written to it.
      expect(writeSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]).toBe(
        recorded.cacheWrite.usage.prompt_tokens_details.cached_tokens,
      );
      expect(writeSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]).toBe(
        recorded.cacheWrite.usage.prompt_tokens_details.cache_write_tokens,
      );
      expect(writeSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]).toBe(0);
      expect(
        writeSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] as number,
      ).toBeGreaterThan(1024);

      // Warm call: the shared prefix is read back instead of written again.
      expect(readSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]).toBe(
        recorded.cacheRead.usage.prompt_tokens_details.cached_tokens,
      );
      expect(readSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]).toBe(
        recorded.cacheRead.usage.prompt_tokens_details.cache_write_tokens,
      );
      expect(
        readSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] as number,
      ).toBeGreaterThan(1024);
      expect(
        readSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] as number,
      ).toBeLessThan(writeSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] as number);

      // Cache read and write never exceed the prompt they describe.
      for (const span of [writeSpan, readSpan]) {
        const read = span.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] as number;
        const write = span.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] as number;
        expect(read + write).toBeLessThanOrEqual(span.attributes[LLM_TOKEN_COUNT_PROMPT] as number);
      }
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

      const [writeSpan, readSpan] = memoryExporter.getFinishedSpans();
      expect(memoryExporter.getFinishedSpans()).toHaveLength(2);

      expect(writeSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]).toBe(
        recorded.cacheWrite.usage.input_tokens_details.cached_tokens,
      );
      expect(writeSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]).toBe(
        recorded.cacheWrite.usage.input_tokens_details.cache_write_tokens,
      );
      expect(writeSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]).toBe(0);
      expect(
        writeSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] as number,
      ).toBeGreaterThan(1024);

      expect(readSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]).toBe(
        recorded.cacheRead.usage.input_tokens_details.cached_tokens,
      );
      expect(readSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]).toBe(
        recorded.cacheRead.usage.input_tokens_details.cache_write_tokens,
      );
      expect(
        readSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] as number,
      ).toBeGreaterThan(1024);
      expect(
        readSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] as number,
      ).toBeLessThan(writeSpan.attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] as number);
    },
  );
});

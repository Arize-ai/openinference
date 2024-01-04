import { OpenAIInstrumentation } from "../src";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
const tracerProvider = new NodeTracerProvider();
tracerProvider.register();

const instrumentation = new OpenAIInstrumentation();
instrumentation.disable();

import * as OpenAI from "openai";

describe("OpenAIInstrumentation", () => {
  let openai: OpenAI.OpenAI;

  const memoryExporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  provider.getTracer("default");

  instrumentation.setTracerProvider(tracerProvider);
  tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));

  beforeEach(() => {
    // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
    instrumentation._modules[0].moduleExports = OpenAI;
    instrumentation.enable();
    openai = new OpenAI.OpenAI({
      apiKey: `fake-api-key`,
    });
    memoryExporter.reset();
  });
  afterEach(() => {
    instrumentation.disable();
    jest.clearAllMocks();
  });
  it("is patched", () => {
    expect(
      (OpenAI as { openInferencePatched?: boolean }).openInferencePatched,
    ).toBe(true);
  });
  it("creates a span for chat completions", async () => {
    const response = {
      id: "chatcmpl-8adq9JloOzNZ9TyuzrKyLpGXexh6p",
      object: "chat.completion",
      created: 1703743645,
      model: "gpt-3.5-turbo-0613",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: "This is a test.",
          },
          logprobs: null,
          finish_reason: "stop",
        },
      ],
      usage: {
        prompt_tokens: 12,
        completion_tokens: 5,
        total_tokens: 17,
      },
    };
    // Mock out the chat completions endpoint
    jest.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        return response;
      },
    );
    await openai.chat.completions.create({
      messages: [{ role: "user", content: "Say this is a test" }],
      model: "gpt-3.5-turbo",
    });
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Chat Completions");
    expect(span.attributes).toMatchInlineSnapshot(`
      {
        "input.mime_type": "application/json",
        "input.value": "{"messages":[{"role":"user","content":"Say this is a test"}],"model":"gpt-3.5-turbo"}",
        "llm.input_messages.0.message.content": "Say this is a test",
        "llm.input_messages.0.message.role": "user",
        "llm.invocation_parameters": "{"model":"gpt-3.5-turbo"}",
        "llm.model_name": "gpt-3.5-turbo",
        "llm.output_messages.0.message.content": "This is a test.",
        "llm.output_messages.0.message.role": "assistant",
        "llm.token_count.completion": 5,
        "llm.token_count.prompt": 12,
        "llm.token_count.total": 17,
        "openinference.span.kind": "llm",
        "output.mime_type": "application/json",
        "output.value": "{"id":"chatcmpl-8adq9JloOzNZ9TyuzrKyLpGXexh6p","object":"chat.completion","created":1703743645,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"message":{"role":"assistant","content":"This is a test."},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}",
      }
    `);
  });
  it("creates a span for embedding create", async () => {
    const response = {
      object: "list",
      data: [{ object: "embedding", index: 0, embedding: [1, 2, 3] }],
    };
    // Mock out the embedding create endpoint
    jest.spyOn(openai, "post").mockImplementation(
      // @ts-expect-error the response type is not correct - this is just for testing
      async (): Promise<unknown> => {
        return response;
      },
    );
    await openai.embeddings.create({
      input: "A happy moment",
      model: "text-embedding-ada-002",
    });
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("OpenAI Embeddings");
    expect(span.attributes).toMatchInlineSnapshot(`
      {
        "embedding.embeddings.0.embedding.text": "A happy moment",
        "embedding.embeddings.0.embedding.vector": [
          1,
          2,
          3,
        ],
        "embedding.model_name": "text-embedding-ada-002",
        "input.mime_type": "text/plain",
        "input.value": "A happy moment",
        "openinference.span.kind": "embedding",
      }
    `);
  });
});

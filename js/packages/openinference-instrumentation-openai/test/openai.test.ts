import { OpenAIInstrumentation } from "../src";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";
import { SemanticResourceAttributes } from "@opentelemetry/semantic-conventions";
const tracerProvider = new NodeTracerProvider();
tracerProvider.register();
const memoryExporter = new InMemorySpanExporter();
const tracer = tracerProvider.getTracer("default");
const resource = new Resource({
  [SemanticResourceAttributes.SERVICE_NAME]: "test-instrumentation-openai",
});

const instrumentation = new OpenAIInstrumentation();
instrumentation.disable();

import * as OpenAI from "openai";
import { ChatCompletion } from "openai/resources";
import { ChatCompletionCreateParamsBase } from "openai/resources/chat/completions";

describe("OpenAIInstrumentation", () => {
  let openai: OpenAI.OpenAI;

  const memoryExporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider();
  const tracer = provider.getTracer("default");

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
    expect((OpenAI as any).openInferencePatched).toBe(true);
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
      // @ts-expect-error
      async (): Promise<any> => {
        return response;
      }
    );
    const chatCompletion = await openai.chat.completions.create({
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
});

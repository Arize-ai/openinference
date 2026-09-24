import { InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import { LLMSystem } from "@arizeai/openinference-semantic-conventions";

import { getSystemFromModelId } from "../src/attributes/attribute-helpers";
import { extractInvokeModelRequestAttributes } from "../src/attributes/invoke-model-request-attributes";
import { extractInvokeModelResponseAttributes } from "../src/attributes/invoke-model-response-attributes";
import { consumeBedrockStreamChunks } from "../src/attributes/invoke-model-streaming-response-attributes";

// Bodies are trimmed from live us.openai.gpt-6-sol and openai.gpt-oss-120b-1:0 responses.
describe("OpenAI models", () => {
  let exporter: InMemorySpanExporter;
  let provider: NodeTracerProvider;

  beforeEach(() => {
    exporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider({
      spanProcessors: [new SimpleSpanProcessor(exporter)],
    });
  });

  afterEach(async () => {
    await provider.shutdown();
  });

  it.each(["us.openai.gpt-6-sol", "global.openai.gpt-6-astra", "openai.gpt-oss-120b-1:0"])(
    "records %s as llm.system openai",
    (modelId) => {
      expect(getSystemFromModelId(modelId)).toBe(LLMSystem.OPENAI);
    },
  );

  it("records the InvokeModel input history, output and token counts", () => {
    const span = provider.getTracer("test").startSpan("invoke_model");
    const messages = [
      { role: "user", content: "What is the weather in Paris?" },
      {
        role: "assistant",
        tool_calls: [
          { id: "call_1", type: "function", function: { name: "get_weather", arguments: "{}" } },
        ],
      },
      { role: "tool", tool_call_id: "call_1", content: '{"temp_c":18}' },
    ];
    const command = new InvokeModelCommand({
      modelId: "us.openai.gpt-6-sol",
      body: JSON.stringify({ messages }),
    });
    extractInvokeModelRequestAttributes({ span, command, system: LLMSystem.OPENAI });
    const body = {
      choices: [{ finish_reason: "stop", message: { role: "assistant", content: "18°C." } }],
      usage: { prompt_tokens: 88, completion_tokens: 12, total_tokens: 100 },
    };
    extractInvokeModelResponseAttributes({
      span,
      response: { body: Buffer.from(JSON.stringify(body)), contentType: "application/json" },
      modelType: LLMSystem.OPENAI,
    });
    span.end();
    expect(exporter.getFinishedSpans()[0].attributes).toMatchObject({
      "llm.input_messages.0.message.contents.0.message_content.text":
        "What is the weather in Paris?",
      "llm.input_messages.1.message.tool_calls.0.tool_call.function.name": "get_weather",
      "llm.output_messages.0.message.content": "18°C.",
      "llm.token_count.prompt": 88,
      "llm.token_count.completion": 12,
      "llm.token_count.total": 100,
    });
  });

  it.each([
    {
      name: "usage chunk",
      last: { choices: [], usage: { prompt_tokens: 14, completion_tokens: 18, total_tokens: 32 } },
      expected: { "llm.token_count.prompt": 14, "llm.token_count.total": 32 },
    },
    {
      // gpt-oss without stream_options.include_usage
      name: "invocation metrics only",
      last: { "amazon-bedrock-invocationMetrics": { inputTokenCount: 75, outputTokenCount: 37 } },
      expected: { "llm.token_count.prompt": 75, "llm.token_count.completion": 37 },
    },
  ])("records the streamed text and tokens from the $name", async ({ last, expected }) => {
    const span = provider.getTracer("test").startSpan("invoke_model_stream");
    async function* stream() {
      for (const data of [
        { choices: [{ delta: { content: "1, 2" } }] },
        { choices: [{ delta: { content: ", 3." } }] },
        last,
      ]) {
        yield { chunk: { bytes: Buffer.from(JSON.stringify(data)) } };
      }
    }
    await consumeBedrockStreamChunks({ span, stream: stream(), modelType: LLMSystem.OPENAI });
    span.end();
    expect(exporter.getFinishedSpans()[0].attributes).toMatchObject({
      "llm.output_messages.0.message.content": "1, 2, 3.",
      ...expected,
    });
  });
});

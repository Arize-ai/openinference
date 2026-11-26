import { ConverseCommand } from "@aws-sdk/client-bedrock-runtime";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

import { extractConverseRequestAttributes } from "../src/attributes/converse-request-attributes";

describe("Converse tool results", () => {
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

  const inputMessages = (command: ConverseCommand) => {
    const span = provider.getTracer("test").startSpan("converse");
    extractConverseRequestAttributes({ span, command });
    span.end();
    const attributes = exporter.getFinishedSpans()[0].attributes;
    return Object.fromEntries(
      Object.entries(attributes).filter(([key]) => key.startsWith("llm.input_messages.")),
    );
  };

  it("records each toolResult block as its own tool message", () => {
    const command = new ConverseCommand({
      modelId: "anthropic.claude-3-5-sonnet-20240620-v1:0",
      messages: [
        { role: "user", content: [{ text: "Weather in Paris and Rome?" }] },
        {
          role: "assistant",
          content: [
            { toolUse: { toolUseId: "tool_1", name: "weather", input: { city: "Paris" } } },
            { toolUse: { toolUseId: "tool_2", name: "weather", input: { city: "Rome" } } },
          ],
        },
        {
          role: "user",
          content: [
            { toolResult: { toolUseId: "tool_1", content: [{ text: "Paris: 18C" }] } },
            {
              toolResult: {
                toolUseId: "tool_2",
                content: [{ text: "Rome:" }, { json: { temp_c: 24 } }],
              },
            },
          ],
        },
      ],
    });

    const messages = inputMessages(command);

    expect(messages["llm.input_messages.2.message.role"]).toBe("tool");
    expect(messages["llm.input_messages.2.message.tool_call_id"]).toBe("tool_1");
    expect(messages["llm.input_messages.2.message.content"]).toBe("Paris: 18C");
    expect(messages["llm.input_messages.3.message.role"]).toBe("tool");
    expect(messages["llm.input_messages.3.message.tool_call_id"]).toBe("tool_2");
    expect(messages["llm.input_messages.3.message.content"]).toBe('Rome:\n\n{"temp_c":24}');
    expect(messages).not.toHaveProperty("llm.input_messages.4.message.role");
  });

  it("keeps the other content of a message that also carries a toolResult", () => {
    const command = new ConverseCommand({
      modelId: "anthropic.claude-3-5-sonnet-20240620-v1:0",
      messages: [
        {
          role: "user",
          content: [
            { toolResult: { toolUseId: "tool_1", content: [{ text: "Paris: 18C" }] } },
            { text: "And tomorrow?" },
          ],
        },
      ],
    });

    const messages = inputMessages(command);

    expect(messages["llm.input_messages.0.message.role"]).toBe("user");
    expect(messages["llm.input_messages.0.message.content"]).toBe("And tomorrow?");
    expect(messages).not.toHaveProperty("llm.input_messages.0.message.tool_call_id");
    expect(messages["llm.input_messages.1.message.role"]).toBe("tool");
    expect(messages["llm.input_messages.1.message.tool_call_id"]).toBe("tool_1");
    expect(messages["llm.input_messages.1.message.content"]).toBe("Paris: 18C");
  });
});

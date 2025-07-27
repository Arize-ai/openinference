import {
  BedrockAgentRuntimeClient,
  InvokeAgentCommand,
} from "@aws-sdk/client-bedrock-agent-runtime";
import { createPolly } from "./utils/polly.config";
import { Polly } from "@pollyjs/core";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";

import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { BedrockAgentInstrumentation } from "../src";
import * as bedrockAgentRuntime from "@aws-sdk/client-bedrock-agent-runtime";
import { setModuleExportsForInstrumentation } from "./utils/test-utils";

describe("BedrockAgentInstrumentation Integration - agent attributes and API recording", () => {
  let instrumentation: BedrockAgentInstrumentation;
  let provider: NodeTracerProvider;
  let memoryExporter: InMemorySpanExporter;

  const region = "ap-south-1";

  const cassettePrefix = "bedrock-agent-without-traces";

  let polly: Polly;

  beforeAll(() => {
    // Setup instrumentation and tracer provider (following OpenAI pattern)
    instrumentation = new BedrockAgentInstrumentation();
    instrumentation.disable();
    memoryExporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
    provider.register();
    instrumentation.setTracerProvider(provider);
    // Manually set module exports for testing (following OpenAI pattern)
    setModuleExportsForInstrumentation(instrumentation, bedrockAgentRuntime);
    // Enable instrumentation ONCE
    instrumentation.enable();
  });

  beforeEach(() => {
    memoryExporter.reset();
    polly = createPolly(cassettePrefix);
  });

  afterEach(async () => {
    await polly.stop();
  });

  // Global cleanup
  afterAll(async () => {
    instrumentation.disable();
    await provider.shutdown();
  });

  it("should record agent attributes and API response in span", async () => {
    const client = new BedrockAgentRuntimeClient({
      region,
      credentials: {
        accessKeyId: "test",
        secretAccessKey: "test",
        sessionToken: "test",
      },
    });
    const params = {
      inputText: "What is the current price of Microsoft?",
      agentId: "3EL4X42BSO",
      agentAliasId: "LSIZXQMZDN",
      sessionId: "default-session1_1234567890",
      enableTrace: false,
    };
    const command = new InvokeAgentCommand(params);
    const response = await client.send(command);
    for await (const event of response.completion as AsyncIterable<{
      chunk: { bytes: Uint8Array };
    }>) {
      const outputText = Buffer.from(event.chunk.bytes).toString("utf8");
      const expected =
        "I've checked the latest stock market data for you. The current price of Microsoft (MSFT) stock is $332.58. This information is based on the most recent market update available in our system.";
      expect(outputText).toBe(expected);
    }
    expect(response).toBeDefined();
    expect(typeof response).toBe("object");
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    const span = spans[0];
    expect(span.name).toBe("bedrock.invoke_agent");
  });
});

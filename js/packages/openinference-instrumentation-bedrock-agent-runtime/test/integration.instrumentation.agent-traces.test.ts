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
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

describe("BedrockAgentInstrumentation Trace Collector Integration - agent attributes and API recording", () => {
  let instrumentation: BedrockAgentInstrumentation;
  let provider: NodeTracerProvider;
  let memoryExporter: InMemorySpanExporter;

  const cassettePrefix = "bedrock-agent-with-traces";

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

  it("should record agent trace attributes and API response in span", async () => {
    const client = new BedrockAgentRuntimeClient({
      region: "us-east-1",
      credentials: {
        accessKeyId: "test",
        secretAccessKey: "test",
      },
    });
    const params = {
      inputText: "What is the current price of Microsoft?",
      agentId: "9Y27QONH1T",
      agentAliasId: "AHSAF5QT0K",
      sessionId: "default-session1_1234567891",
      enableTrace: true,
    };
    const command = new InvokeAgentCommand(params);
    const response = await client.send(command);
    for await (const event of response.completion as AsyncIterable<{
      chunk: { bytes: Uint8Array };
      trace?: object;
    }>) {
      if (event.chunk?.bytes) {
        const outputText = Buffer.from(event.chunk.bytes).toString("utf8");
        expect(outputText).not.toBeNull();
      }
    }
    expect(response).toBeDefined();
    expect(typeof response).toBe("object");
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(3);
  });

  it("should record Code Interpreter Orchestration Trace", async () => {
    const client = new BedrockAgentRuntimeClient({
      region: "us-east-1",
      credentials: {
        accessKeyId: "test-access-key-id",
        secretAccessKey: "test-access-key",
      },
    });
    const params = {
      inputText: "Write programe for (a+b)**3?",
      agentId: "EQWGOQC49C",
      agentAliasId: "ALR0DJYNLC",
      sessionId: "default-session1_1234567891",
      enableTrace: true,
    };
    const command = new InvokeAgentCommand(params);
    const response = await client.send(command);
    for await (const event of response.completion as AsyncIterable<{
      chunk: { bytes: Uint8Array };
      trace?: object;
    }>) {
      if (event.chunk?.bytes) {
        const outputText = Buffer.from(event.chunk.bytes).toString("utf8");
        expect(outputText).not.toBeNull();
      }
    }
    expect(response).toBeDefined();
    expect(typeof response).toBe("object");
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(7);
  });
  it("should record knowledge base traces and API response in span", async () => {
    const client = new BedrockAgentRuntimeClient({
      region: "ap-south-1",
      credentials: {
        accessKeyId: "test-access-key-id",
        secretAccessKey: "test-access-key",
      },
    });
    const params = {
      inputText: "What is Task decomposition?",
      agentId: "G0OUMYARBX",
      agentAliasId: "YYTTVM7BYE",
      sessionId: "default-session1_1234567891",
      enableTrace: true,
    };
    const command = new InvokeAgentCommand(params);
    const response = await client.send(command);
    for await (const event of response.completion as AsyncIterable<{
      chunk: { bytes: Uint8Array };
      trace?: object;
    }>) {
      if (event.chunk?.bytes) {
        const outputText = Buffer.from(event.chunk.bytes).toString("utf8");
        expect(outputText).not.toBeNull();
      }
    }
    expect(response).toBeDefined();
    expect(typeof response).toBe("object");
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(4);
    const retrieverSpan = spans.find((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.RETRIEVER
      );
    });
    expect(retrieverSpan).toBeDefined();
    expect(retrieverSpan?.name).toBe("knowledge_base");
    const agentSpan = spans.find((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.AGENT
      );
    });
    expect(agentSpan).toBeDefined();
    expect(agentSpan?.name).toBe("bedrock.invoke_agent");
    const chainSpan = spans.find((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.CHAIN
      );
    });
    expect(chainSpan).toBeDefined();
    expect(chainSpan?.name).toBe("orchestrationTrace");
    const llmSpan = spans.find((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.LLM
      );
    });
    expect(llmSpan).toBeDefined();
    expect(llmSpan?.name).toBe("LLM");
  });

  it("should record all pre post orchestration traces", async () => {
    const client = new BedrockAgentRuntimeClient({
      region: "us-east-1",
      credentials: {
        accessKeyId: "test-access-key-id",
        secretAccessKey: "test-access-key",
      },
    });
    const params = {
      inputText: "Find the sum of first 5 fibnonic numbers?",
      agentId: "XNW1LGJJZT",
      agentAliasId: "K0P4LV9GPO",
      sessionId: "default-session1_1234567892",
      enableTrace: true,
    };
    const command = new InvokeAgentCommand(params);
    const response = await client.send(command);
    for await (const event of response.completion as AsyncIterable<{
      chunk: { bytes: Uint8Array };
      trace?: object;
    }>) {
      if (event.chunk?.bytes) {
        const outputText = Buffer.from(event.chunk.bytes).toString("utf8");
        expect(outputText).not.toBeNull();
      }
    }
    expect(response).toBeDefined();
    expect(typeof response).toBe("object");
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(9);
  });

  it("should record multiple agents collaboration traces", async () => {
    const client = new BedrockAgentRuntimeClient({
      region: "us-east-1",
      credentials: {
        accessKeyId: "test-access-key-id",
        secretAccessKey: "test-access-key",
      },
    });
    const params = {
      inputText: "Find the sum of first 10 fibnonic numbers?",
      agentId: "2X9SRVPLWB",
      agentAliasId: "KUXISKYLTT",
      sessionId: "default-session1_1234567893",
      enableTrace: true,
    };
    const command = new InvokeAgentCommand(params);
    const response = await client.send(command);
    for await (const event of response.completion as AsyncIterable<{
      chunk: { bytes: Uint8Array };
      trace?: object;
    }>) {
      if (event.chunk?.bytes) {
        const outputText = Buffer.from(event.chunk.bytes).toString("utf8");
        expect(outputText).not.toBeNull();
      }
    }
    expect(response).toBeDefined();
    expect(typeof response).toBe("object");
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(19);
  });
});

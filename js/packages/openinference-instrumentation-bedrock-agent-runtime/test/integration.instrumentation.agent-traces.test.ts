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
  LLMProvider,
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
    const llmSpans = spans.filter((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.LLM
      );
    });

    expect(llmSpans.length).toBe(3);
    llmSpans.forEach((span) => {
      expect(span.attributes[SemanticConventions.LLM_MODEL_NAME]).toBe(
        "anthropic.claude-3-sonnet-20240229-v1:0",
      );
      expect(span.attributes[SemanticConventions.LLM_PROVIDER]).toBe(
        LLMProvider.AWS,
      );
      const attributeKeys = Object.keys(span.attributes);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_INPUT_MESSAGES),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_OUTPUT_MESSAGES),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_INVOCATION_PARAMETERS),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_TOKEN_COUNT_PROMPT),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_TOKEN_COUNT_TOTAL),
        ),
      ).toBe(true);
    });
    const toolSpans = spans.filter((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.TOOL
      );
    });
    expect(toolSpans.length).toBe(2);
    const agentSpans = spans.filter((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.AGENT
      );
    });
    expect(agentSpans.length).toBe(1);
    const chainSpans = spans.filter((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.CHAIN
      );
    });
    expect(chainSpans.length).toBe(1);
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

    const chainSpans = spans.filter((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.CHAIN
      );
    });
    // one pre, one orchestration, one post
    expect(chainSpans.length).toBe(3);
    // one top level agent span
    const agentSpans = spans.filter((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.AGENT
      );
    });
    expect(agentSpans.length).toBe(1);

    // The remaining spans are LLM spans
    const llmSpans = spans.filter((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.LLM
      );
    });
    expect(llmSpans.length).toBe(5);

    llmSpans.forEach((span, i) => {
      expect(span.attributes[SemanticConventions.LLM_MODEL_NAME]).toBe(
        "anthropic.claude-3-sonnet-20240229-v1:0",
      );
      expect(span.attributes[SemanticConventions.LLM_PROVIDER]).toBe(
        LLMProvider.AWS,
      );
      const attributeKeys = Object.keys(span.attributes);
      // The last message is post orchestration and is the result from the llm so does not have input messages
      if (i !== llmSpans.length - 1) {
        expect(
          attributeKeys.some((key) =>
            key.includes(SemanticConventions.LLM_INPUT_MESSAGES),
          ),
        ).toBe(true);
      }
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_OUTPUT_MESSAGES),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_INVOCATION_PARAMETERS),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_TOKEN_COUNT_PROMPT),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_TOKEN_COUNT_TOTAL),
        ),
      ).toBe(true);
    });
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

    const agentSpans = spans.filter((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.AGENT
      );
    });
    // invoke, supervisor, 2 math solvers
    expect(agentSpans.length).toBe(4);
    agentSpans.forEach((span) => {
      expect(
        span.name === "bedrock.invoke_agent" ||
          /agent_collaborator\[.*?\]/.test(span.name),
      ).toBe(true);
    });
    const chainSpans = spans.filter((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.CHAIN
      );
    });
    // 1 orchestration trace for each agent
    expect(chainSpans.length).toBe(4);
    expect(chainSpans.every((span) => span.name === "orchestrationTrace")).toBe(
      true,
    );

    const llmSpans = spans.filter((span) => {
      return (
        span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.LLM
      );
    });
    expect(llmSpans.length).toBe(11);
    llmSpans.forEach((span) => {
      expect(span.name).toBe("LLM");
      const modelName = span.attributes[SemanticConventions.LLM_MODEL_NAME];
      expect(typeof modelName).toBe("string");
      if (typeof modelName === "string") {
        expect(modelName.includes("anthropic.claude")).toBe(true);
      }
      const attributeKeys = Object.keys(span.attributes);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_INPUT_MESSAGES),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_OUTPUT_MESSAGES),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_INVOCATION_PARAMETERS),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_TOKEN_COUNT_PROMPT),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION),
        ),
      ).toBe(true);
      expect(
        attributeKeys.some((key) =>
          key.includes(SemanticConventions.LLM_TOKEN_COUNT_TOTAL),
        ),
      ).toBe(true);
    });
  });
});

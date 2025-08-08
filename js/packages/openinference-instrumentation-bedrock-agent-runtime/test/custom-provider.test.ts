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
import { registerInstrumentations } from "@opentelemetry/instrumentation";

describe("BedrockAgentInstrumentation with a custom tracer provider", () => {
  const cassettePrefix = "bedrock-agent-custom-trace-provider";
  const memoryExporter = new InMemorySpanExporter();
  const region = "ap-south-1";
  let polly: Polly;

  describe("BedrockAgentInstrumentation with custom TracerProvider passed in", () => {
    const customTracerProvider = new NodeTracerProvider();
    const customMemoryExporter = new InMemorySpanExporter();

    // Note: We don't register this provider globally.
    customTracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(customMemoryExporter),
    );

    // Instantiate instrumentation with the custom provider
    const instrumentation = new BedrockAgentInstrumentation({
      tracerProvider: customTracerProvider,
    });
    instrumentation.disable();

    // Mock the module exports like in other tests
    setModuleExportsForInstrumentation(instrumentation, bedrockAgentRuntime);

    beforeAll(() => {
      instrumentation.enable();
      polly = createPolly(cassettePrefix);
    });

    afterAll(() => {
      instrumentation.disable();
    });

    beforeEach(() => {
      memoryExporter.reset();
      customMemoryExporter.reset();
    });

    afterEach(async () => {
      jest.resetAllMocks();
      jest.clearAllMocks();
      await polly.stop();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
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
          "The current price of Microsoft (MSFT) stock is $334.57.";
        expect(outputText).toContain(expected);
      }
      const spans = customMemoryExporter.getFinishedSpans();
      const globalSpans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(globalSpans.length).toBe(0);
    });
  });

  describe("BedrockkAgentInstrumentation with custom TracerProvider set", () => {
    const customTracerProvider = new NodeTracerProvider();
    const customMemoryExporter = new InMemorySpanExporter();

    // Note: We don't register this provider globally.
    customTracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(customMemoryExporter),
    );

    // Instantiate instrumentation with the custom provider
    const instrumentation = new BedrockAgentInstrumentation({});
    instrumentation.setTracerProvider(customTracerProvider);
    instrumentation.disable();

    // Mock the module exports like in other tests
    setModuleExportsForInstrumentation(instrumentation, bedrockAgentRuntime);

    beforeAll(() => {
      instrumentation.enable();
    });

    afterAll(() => {
      instrumentation.disable();
    });

    beforeEach(() => {
      memoryExporter.reset();
      customMemoryExporter.reset();
      polly = createPolly(cassettePrefix);
    });

    afterEach(async () => {
      jest.resetAllMocks();
      jest.clearAllMocks();
      await polly.stop();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
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
          "the current price of Microsoft (MSFT) stock is $334.57.";
        expect(outputText).toContain(expected);
      }

      const spans = customMemoryExporter.getFinishedSpans();
      const globalSpans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(globalSpans.length).toBe(0);
    });
  });

  describe("BedrockAgentInstrumentation with custom TracerProvider set via registerInstrumentations", () => {
    const customTracerProvider = new NodeTracerProvider();
    const customMemoryExporter = new InMemorySpanExporter();

    // Note: We don't register this provider globally.
    customTracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(customMemoryExporter),
    );

    // Instantiate instrumentation with the custom provider
    const instrumentation = new BedrockAgentInstrumentation();
    registerInstrumentations({
      instrumentations: [instrumentation],
      tracerProvider: customTracerProvider,
    });
    instrumentation.disable();

    // Mock the module exports like in other tests
    setModuleExportsForInstrumentation(instrumentation, bedrockAgentRuntime);

    beforeAll(() => {
      instrumentation.enable();
      polly = createPolly(cassettePrefix);
    });

    afterAll(() => {
      instrumentation.disable();
    });

    beforeEach(() => {
      memoryExporter.reset();
      customMemoryExporter.reset();
    });

    afterEach(async () => {
      jest.resetAllMocks();
      jest.clearAllMocks();
      await polly.stop();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
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
          "the current price of Microsoft (MSFT) stock is $334.57.";
        expect(outputText).toContain(expected);
      }
      const spans = customMemoryExporter.getFinishedSpans();
      const globalSpans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(globalSpans.length).toBe(0);
    });
  });
});

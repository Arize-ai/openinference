import { BedrockInstrumentation } from "../src/instrumentation";
import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import { NodeSDK } from "@opentelemetry/sdk-node";
import { getNodeAutoInstrumentations } from "@opentelemetry/auto-instrumentations-node";
import { Resource } from "@opentelemetry/resources";
import { SemanticResourceAttributes } from "@opentelemetry/semantic-conventions";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { SpanKind } from "@opentelemetry/api";
import { Polly } from "@pollyjs/core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

// Test constants
const TEST_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0";
const TEST_USER_MESSAGE = "Hello, how are you?";
const TEST_MAX_TOKENS = 100;

describe("BedrockInstrumentation", () => {
  let instrumentation: BedrockInstrumentation;
  let sdk: NodeSDK;
  let spanExporter: InMemorySpanExporter;
  let polly: Polly;

  beforeEach(() => {
    // Setup Polly for VCR-style testing (memory persistence for now)
    polly = new Polly("BedrockInstrumentation", {
      adapters: ["node-http"],
      recordFailedRequests: true,
    });

    // Setup instrumentation and SDK
    instrumentation = new BedrockInstrumentation();
    spanExporter = new InMemorySpanExporter();
    
    sdk = new NodeSDK({
      resource: new Resource({
        [SemanticResourceAttributes.SERVICE_NAME]: "test-service",
      }),
      spanProcessor: new SimpleSpanProcessor(spanExporter),
      instrumentations: [instrumentation],
    });
    
    sdk.start();
  });

  afterEach(async () => {
    await polly.stop();
    await sdk.shutdown();
    spanExporter.reset();
  });

  describe("InvokeModel basic instrumentation", () => {
    it("should create spans for InvokeModel calls", async () => {
      const client = new BedrockRuntimeClient({ 
        region: process.env.AWS_REGION || "us-east-1",
        credentials: {
          accessKeyId: process.env.AWS_ACCESS_KEY_ID || "test-key",
          secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || "test-secret",
        },
      });

      const command = new InvokeModelCommand({
        modelId: TEST_MODEL_ID,
        body: JSON.stringify({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: TEST_MAX_TOKENS,
          messages: [
            {
              role: "user",
              content: TEST_USER_MESSAGE,
            },
          ],
        }),
        contentType: "application/json",
        accept: "application/json",
      });

      const result = await client.send(command);
      
      // Verify the response structure
      expect(result.body).toBeDefined();
      expect(result.contentType).toBe("application/json");
      
      // Verify spans were created
      const spans = spanExporter.getFinishedSpans();
      expect(spans).toHaveLength(1);
      
      const span = spans[0];
      expect(span.name).toBe("bedrock.invoke_model");
      expect(span.kind).toBe(SpanKind.CLIENT);
      
      // Verify semantic conventions
      const attributes = span.attributes;
      expect(attributes[SemanticConventions.LLM_SYSTEM]).toBe("bedrock");
      expect(attributes[SemanticConventions.LLM_REQUEST_MODEL]).toBe(TEST_MODEL_ID);
      expect(attributes[SemanticConventions.LLM_REQUEST_TYPE]).toBe("inference");
      
      // Verify input messages
      expect(attributes[`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]).toBe("user");
      expect(attributes[`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]).toBe(TEST_USER_MESSAGE);
      
      // Verify output messages (basic structure)
      expect(attributes[`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]).toBe("assistant");
      expect(attributes[`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]).toBeDefined();
      
      // Verify token counts if present
      if (attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT]) {
        expect(typeof attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT]).toBe("number");
      }
      if (attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]) {
        expect(typeof attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]).toBe("number");
      }
      if (attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL]) {
        expect(typeof attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL]).toBe("number");
      }
    });
  });
});
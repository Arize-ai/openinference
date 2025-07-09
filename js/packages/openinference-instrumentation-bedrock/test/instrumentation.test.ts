import { BedrockInstrumentation } from "../src/instrumentation";
import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SpanKind } from "@opentelemetry/api";
import nock from "nock";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import * as fs from "fs";
import * as path from "path";

// Test constants  
const TEST_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0";
const TEST_USER_MESSAGE = "Hello, how are you?";
const TEST_MAX_TOKENS = 100;

// Clearly fake credentials for VCR testing - these are NEVER real
const MOCK_AWS_CREDENTIALS = {
  accessKeyId: "AKIATEST1234567890AB",
  secretAccessKey: "FAKE-SECRET-KEY-FOR-VCR-TESTING-ONLY-1234567890",
  sessionToken: "FAKE-SESSION-TOKEN-FOR-VCR-TESTING-ONLY",
};

const VALID_AWS_CREDENTIALS = {
  accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  sessionToken: process.env.AWS_SESSION_TOKEN!,
}

const MOCK_AUTH_HEADERS = {
  authorization: "AWS4-HMAC-SHA256 Credential=AKIATEST1234567890AB/20250626/us-east-1/bedrock/aws4_request, SignedHeaders=accept;content-length;content-type;host;x-amz-date, Signature=fake-signature-for-vcr-testing",
  "x-amz-security-token": MOCK_AWS_CREDENTIALS.sessionToken,
  "x-amz-date": "20250626T120000Z",
};
// console.log('AWS Keys available:', {
//     accessKeyId: process.env.AWS_ACCESS_KEY_ID ? 'SET' : 'NOT_SET',
//     secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY ? 'SET' : 'NOT_SET'
//   });

describe("BedrockInstrumentation", () => {
  let instrumentation: BedrockInstrumentation;
  let provider: NodeTracerProvider;
  let spanExporter: InMemorySpanExporter;
  
  const recordingsPath = path.join(__dirname, "recordings", "bedrock-recordings.json");
  const isRecordingMode = process.env.BEDROCK_RECORD_MODE === 'record';

  beforeEach(() => {
    // Setup nock for VCR-style testing
    if (isRecordingMode) {
      // Recording mode: capture real requests
      nock.recorder.rec({
        output_objects: true,
        enable_reqheaders_recording: true,
      });
    } else {
      // Replay mode: create simplified mock that ignores auth differences
      const mockResponse = fs.existsSync(recordingsPath) ? 
        JSON.parse(fs.readFileSync(recordingsPath, "utf8"))[0].response : 
        null;
        
      if (mockResponse) {
        console.log(`Creating mock from sanitized recording`);
        nock("https://bedrock-runtime.us-east-1.amazonaws.com")
          .post(`/model/${encodeURIComponent(TEST_MODEL_ID)}/invoke`)
          .reply(200, mockResponse);
      } else {
        console.log(`No recordings found at ${recordingsPath}`);
      }
    }

    // Setup instrumentation and tracer provider (following OpenAI pattern)
    instrumentation = new BedrockInstrumentation();
    instrumentation.disable(); // Initially disabled
    spanExporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider();
    
    provider.addSpanProcessor(new SimpleSpanProcessor(spanExporter));
    provider.register();
    instrumentation.setTracerProvider(provider);
    
    // Manually set module exports for testing (following OpenAI pattern)
    const BedrockRuntime = require("@aws-sdk/client-bedrock-runtime");
    (instrumentation as any)._modules[0].moduleExports = BedrockRuntime;
    
    // Enable instrumentation BEFORE creating any clients
    instrumentation.enable();
  });

  afterEach(() => {
    if (isRecordingMode) {
      // Save recordings before cleaning up
      const recordings = nock.recorder.play();
      console.log(`Captured ${recordings.length} recordings`);
      if (recordings.length > 0) {
        // Sanitize auth headers - replace with mock credentials for replay compatibility
        recordings.forEach((recording: any) => {
          if (recording.reqheaders) {
            Object.assign(recording.reqheaders, MOCK_AUTH_HEADERS);
          }
        });
        
        const recordingsDir = path.dirname(recordingsPath);
        if (!fs.existsSync(recordingsDir)) {
          fs.mkdirSync(recordingsDir, { recursive: true });
        }
        fs.writeFileSync(recordingsPath, JSON.stringify(recordings, null, 2));
        console.log(`Saved sanitized recordings to ${recordingsPath}`);
      }
    }
    
    nock.cleanAll();
    nock.restore();
    instrumentation.disable();
    provider.shutdown();
    spanExporter.reset();
  });

  describe("InvokeModel basic instrumentation", () => {
    it("should create spans for InvokeModel calls", async () => {
      const client = new BedrockRuntimeClient({ 
        region: "us-east-1",
        credentials: isRecordingMode ? {
          // Recording mode: use real credentials from environment
          ...VALID_AWS_CREDENTIALS,
        } : {
          // Replay mode: use mock credentials that match sanitized recordings
          ...MOCK_AWS_CREDENTIALS,
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
      
      // Test span attributes using OpenAI JS instrumentation pattern
      expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "Hello, how are you?",
  "llm.input_messages.0.message.content": "Hello, how are you?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "anthropic.claude-3-5-sonnet-20240620-v1:0",
  "llm.output_messages.0.message.content": "Hello! As an AI language model, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "bedrock",
  "llm.token_count.completion": 35,
  "llm.token_count.prompt": 13,
  "llm.token_count.total": 48,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "Hello! As an AI language model, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?",
}
`);
    });

    // TODO: Add more test scenarios following the TDD plan:
    // 
    // Phase 1: InvokeModel Foundation
    // - it("should handle missing token counts gracefully", async () => {})
    // - it("should handle tool calling with function definitions", async () => {})
    // - it("should handle tool results processing", async () => {})
    // - it("should handle multi-modal messages with images", async () => {})
    // - it("should handle API errors gracefully", async () => {})
    //
    // Phase 2: Streaming Support  
    // - it("should handle InvokeModelWithResponseStream", async () => {})
    // - it("should handle streaming tool calls", async () => {})
    //
    // Phase 3: Converse API Support
    // - it("should handle Converse API calls", async () => {})
    // - it("should handle system prompts in Converse API", async () => {})
    // - it("should handle tool calling via Converse API", async () => {})
    //
    // Use pattern: BEDROCK_RECORD_MODE=record npm test -- --testNamePattern="test name"
  });
});

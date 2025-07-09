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
  let currentTestName: string;
  let recordingsPath: string;
  
  const isRecordingMode = process.env.BEDROCK_RECORD_MODE === 'record';

  // Global setup - initialize instrumentation once
  beforeAll(() => {
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
    
    // Enable instrumentation ONCE
    instrumentation.enable();
  });

  // Global cleanup
  afterAll(async () => {
    instrumentation.disable();
    await provider.shutdown();
  });

  // Helper function to create sanitized recording path
  const createRecordingPath = (testName: string) => {
    const sanitizedName = testName
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '');
    return path.join(__dirname, "recordings", `${sanitizedName}.json`);
  };

  // Generate recording file path based on test name
  const getRecordingPath = (testName: string) => {
    return createRecordingPath(testName);
  };

  // Helper function to load mock response from recording file
  const loadRecordingResponse = (recordingPath: string) => {
    if (!fs.existsSync(recordingPath)) {
      return null;
    }
    
    try {
      const recordingData = JSON.parse(fs.readFileSync(recordingPath, "utf8"));
      return recordingData[0]?.response || null;
    } catch (error) {
      console.warn(`Failed to load recording from ${recordingPath}:`, error);
      return null;
    }
  };

  // Helper function to create nock mock for Bedrock API
  const createNockMock = (mockResponse: any) => {
    nock("https://bedrock-runtime.us-east-1.amazonaws.com")
      .post(`/model/${encodeURIComponent(TEST_MODEL_ID)}/invoke`)
      .reply(200, mockResponse);
  };

  // Helper function to sanitize auth headers in recordings
  const sanitizeAuthHeaders = (recordings: any[]) => {
    recordings.forEach((recording: any) => {
      if (recording.reqheaders) {
        Object.assign(recording.reqheaders, MOCK_AUTH_HEADERS);
      }
    });
  };

  // Helper function to verify response structure
  const verifyResponseStructure = (result: any) => {
    expect(result.body).toBeDefined();
    expect(result.contentType).toBe("application/json");
  };

  // Helper function to verify basic span structure and return the span
  const verifySpanBasics = (spanExporter: InMemorySpanExporter) => {
    const spans = spanExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    
    const span = spans[0];
    expect(span.name).toBe("bedrock.invoke_model");
    expect(span.kind).toBe(SpanKind.CLIENT);
    
    return span;
  };

  // Helper function for tests to set up their specific recording
  const setupTestRecording = (testName: string) => {
    currentTestName = testName;
    recordingsPath = getRecordingPath(testName);
    
    if (!isRecordingMode) {
      // Replay mode: create mock from test-specific recording
      const mockResponse = loadRecordingResponse(recordingsPath);
        
      if (mockResponse) {
        console.log(`Creating mock from sanitized recording: ${path.basename(recordingsPath)}`);
        createNockMock(mockResponse);
      } else {
        console.log(`No recordings found at ${recordingsPath}`);
      }
    }
  };

  // Helper function to create test client with consistent configuration
  const createTestClient = () => {
    return new BedrockRuntimeClient({ 
      region: "us-east-1",
      credentials: isRecordingMode ? {
        // Recording mode: use real credentials from environment
        ...VALID_AWS_CREDENTIALS,
      } : {
        // Replay mode: use mock credentials that match sanitized recordings
        ...MOCK_AWS_CREDENTIALS,
      },
      // Disable connection reuse to ensure nock can intercept properly
      requestHandler: {
        connectionTimeout: 1000,
        requestTimeout: 5000,
      }
    });
  };

  beforeEach(() => {
    // Clear any existing nock mocks first
    nock.cleanAll();
    
    // Ensure nock is active (important for test isolation)
    if (!nock.isActive()) {
      nock.activate();
    }
    
    // Set default test name (will be overridden by setupTestRecording)
    currentTestName = 'default-test';
    recordingsPath = getRecordingPath(currentTestName);
    
    // Setup nock for VCR-style testing (recording mode only)
    if (isRecordingMode) {
      // Recording mode: capture real requests
      nock.recorder.rec({
        output_objects: true,
        enable_reqheaders_recording: true,
      });
    }

    // Reset span exporter for clean test state
    spanExporter.reset();
  });

  afterEach(() => {
    if (isRecordingMode) {
      // Save recordings before cleaning up
      const recordings = nock.recorder.play();
      console.log(`Captured ${recordings.length} recordings for test: ${currentTestName}`);
      if (recordings.length > 0) {
        // Sanitize auth headers - replace with mock credentials for replay compatibility
        sanitizeAuthHeaders(recordings);
        
        const recordingsDir = path.dirname(recordingsPath);
        if (!fs.existsSync(recordingsDir)) {
          fs.mkdirSync(recordingsDir, { recursive: true });
        }
        fs.writeFileSync(recordingsPath, JSON.stringify(recordings, null, 2));
        console.log(`Saved sanitized recordings to ${path.basename(recordingsPath)}`);
      }
    }
    
    // Clean up nock only
    nock.cleanAll();
    nock.restore();
  });

  describe("InvokeModel basic instrumentation", () => {
    it("should create spans for InvokeModel calls", async () => {
      setupTestRecording("should create spans for InvokeModel calls");

      const client = createTestClient();

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
      verifyResponseStructure(result);
      
      const span = verifySpanBasics(spanExporter);
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

    it("should handle tool calling with function definitions", async () => {
      setupTestRecording("should handle tool calling with function definitions");
      
      const toolDefinition = {
        name: "get_weather",
        description: "Get current weather for a location",
        input_schema: {
          type: "object",
          properties: {
            location: { type: "string", description: "The city and state" }
          },
          required: ["location"]
        }
      };

      const client = createTestClient();

      const command = new InvokeModelCommand({
        modelId: TEST_MODEL_ID,
        body: JSON.stringify({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: TEST_MAX_TOKENS,
          tools: [toolDefinition],
          messages: [
            {
              role: "user",
              content: "What's the weather like in San Francisco?",
            },
          ],
        }),
        contentType: "application/json",
        accept: "application/json",
      });

      const result = await client.send(command);
      verifyResponseStructure(result);
      
      const span = verifySpanBasics(spanExporter);
      expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "What's the weather like in San Francisco?",
  "llm.input_messages.0.message.content": "What's the weather like in San Francisco?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "anthropic.claude-3-5-sonnet-20240620-v1:0",
  "llm.output_messages.0.message.content": "Certainly! I can help you with that information. To get the current weather for San Francisco, I'll use the get_weather function. Let me fetch that data for you.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{"location":"San Francisco, CA"}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "toolu_bdrk_01MqHGzs8QwkdkVjJYrbLTPp",
  "llm.provider": "aws",
  "llm.system": "bedrock",
  "llm.token_count.completion": 94,
  "llm.token_count.prompt": 373,
  "llm.token_count.total": 467,
  "llm.tools.0.tool.json_schema": "{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state"}},"required":["location"]}}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "Certainly! I can help you with that information. To get the current weather for San Francisco, I'll use the get_weather function. Let me fetch that data for you.",
}
`);
    });

    // TODO: Add more test scenarios following the TDD plan:
    // 
    // Phase 1: InvokeModel Foundation
    // - it("should handle missing token counts gracefully", async () => {})
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

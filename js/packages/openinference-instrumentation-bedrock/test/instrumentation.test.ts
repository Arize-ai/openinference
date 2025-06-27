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

// Sanitized credentials for VCR testing
const MOCK_AWS_CREDENTIALS = {
  accessKeyId: "AKIAIOSFODNN7EXAMPLE",
  secretAccessKey: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
  sessionToken: "AQoDYXdzEJr...<truncated>...EXAMPLESessionToken",
};

const VALID_AWS_CREDENTIALS = {
  accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  sessionToken: process.env.AWS_SESSION_TOKEN!,
}

const MOCK_AUTH_HEADERS = {
  authorization: "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20250626/us-east-1/bedrock/aws4_request, SignedHeaders=accept;content-length;content-type;host;x-amz-date, Signature=example-signature",
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
  const isRecordingMode = false; // Set to true to record new API calls

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
      
      // Extract attributes following Python test patterns
      const attributes = { ...span.attributes } as Record<string, any>;
      
      // Core LLM attributes (following Python patterns)
      expect(attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe("LLM");
      expect(attributes[SemanticConventions.LLM_MODEL_NAME]).toBe(TEST_MODEL_ID);
      expect(attributes[SemanticConventions.LLM_SYSTEM]).toBe("bedrock");
      expect(attributes[SemanticConventions.LLM_PROVIDER]).toBe("aws");
      
      // Token counts from response usage (Converse API style)
      expect(attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT]).toBe(13);
      expect(attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]).toBe(30);
      expect(attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL]).toBe(43);
      
      // Input/Output values
      expect(attributes[SemanticConventions.INPUT_VALUE]).toBe(TEST_USER_MESSAGE);
      expect(attributes[SemanticConventions.OUTPUT_VALUE]).toBe("Hello! I'm doing well, thank you for asking. How are you doing today? Is there anything I can help you with?");
      
      // Input messages structure
      expect(attributes["llm.input_messages.0.message.role"]).toBe("user");
      expect(attributes["llm.input_messages.0.message.content"]).toBe(TEST_USER_MESSAGE);
      
      // Output messages structure  
      expect(attributes["llm.output_messages.0.message.role"]).toBe("assistant");
      expect(attributes["llm.output_messages.0.message.content"]).toBe("Hello! I'm doing well, thank you for asking. How are you doing today? Is there anything I can help you with?");
      
      // Invocation parameters (extracted from request body)
      const invocationParamsStr = attributes[SemanticConventions.LLM_INVOCATION_PARAMETERS];
      expect(typeof invocationParamsStr).toBe("string");
      const invocationParams = JSON.parse(invocationParamsStr);
      expect(invocationParams).toEqual({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": TEST_MAX_TOKENS,
      });
    });

    it("should handle missing token counts gracefully", async () => {
      // Create a custom mock response with missing input token count (similar to Python test)
      const mockResponseWithMissingTokens = {
        "id": "msg_bdrk_013Ears62zVrJf8kRVWywwUc",
        "type": "message", 
        "role": "assistant",
        "model": "claude-3-5-sonnet-20240620",
        "content": [
          {
            "type": "text",
            "text": "Hello! I'm doing well, thank you for asking."
          }
        ],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
          "output_tokens": 12  // Missing input_tokens
        }
      };

      // Override the mock for this test
      nock.cleanAll();
      nock("https://bedrock-runtime.us-east-1.amazonaws.com")
        .post(`/model/${encodeURIComponent(TEST_MODEL_ID)}/invoke`)
        .reply(200, mockResponseWithMissingTokens);

      const client = new BedrockRuntimeClient({ 
        region: "us-east-1",
        credentials: MOCK_AWS_CREDENTIALS,
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
      expect(result.body).toBeDefined();
      
      // Verify spans were created
      const spans = spanExporter.getFinishedSpans();
      expect(spans).toHaveLength(1);
      
      const span = spans[0];
      const attributes = { ...span.attributes } as Record<string, any>;
      
      // Core attributes should still be present
      expect(attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe("LLM");
      expect(attributes[SemanticConventions.LLM_MODEL_NAME]).toBe(TEST_MODEL_ID);
      expect(attributes[SemanticConventions.LLM_SYSTEM]).toBe("bedrock");
      
      // Only output token count should be present  
      expect(attributes[SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]).toBe(12);
      expect(attributes[SemanticConventions.LLM_TOKEN_COUNT_PROMPT]).toBeUndefined();
      expect(attributes[SemanticConventions.LLM_TOKEN_COUNT_TOTAL]).toBeUndefined();
      
      // Other attributes should still work
      expect(attributes[SemanticConventions.INPUT_VALUE]).toBe(TEST_USER_MESSAGE);
      expect(attributes[SemanticConventions.OUTPUT_VALUE]).toBe("Hello! I'm doing well, thank you for asking.");
    });

    it("should capture raw input/output values with proper MIME types", async () => {
      const client = new BedrockRuntimeClient({ 
        region: "us-east-1",
        credentials: MOCK_AWS_CREDENTIALS,
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
      expect(result.body).toBeDefined();
      
      // Verify spans were created
      const spans = spanExporter.getFinishedSpans();
      expect(spans).toHaveLength(1);
      
      const span = spans[0];
      const attributes = { ...span.attributes } as Record<string, any>;
      
      // Verify MIME types are captured
      expect(attributes[SemanticConventions.INPUT_MIME_TYPE]).toBe("application/json");
      expect(attributes[SemanticConventions.OUTPUT_MIME_TYPE]).toBe("application/json");
      
      // Verify raw input value contains the full command structure
      const inputValue = attributes[SemanticConventions.INPUT_VALUE];
      expect(typeof inputValue).toBe("string");
      
      // Verify raw output value contains the full response
      const outputValue = attributes[SemanticConventions.OUTPUT_VALUE];
      expect(typeof outputValue).toBe("string");
      
      // For Bedrock, the raw output should be the full response JSON
      const parsedOutput = JSON.parse(outputValue);
      expect(parsedOutput.id).toBe("msg_bdrk_013Ears62zVrJf8kRVWywwUc");
      expect(parsedOutput.type).toBe("message");
      expect(parsedOutput.role).toBe("assistant");
      expect(parsedOutput.content[0].text).toBe("Hello! I'm doing well, thank you for asking. How are you doing today? Is there anything I can help you with?");
    });
  });
});

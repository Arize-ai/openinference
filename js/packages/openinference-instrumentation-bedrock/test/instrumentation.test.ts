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
          accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
          secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
          sessionToken: process.env.AWS_SESSION_TOKEN!,
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
      
      // Verify semantic conventions
      const attributes = span.attributes;
      expect(attributes[SemanticConventions.LLM_SYSTEM]).toBe("bedrock");
      expect(attributes[SemanticConventions.LLM_MODEL_NAME]).toBe(TEST_MODEL_ID);
      
      
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

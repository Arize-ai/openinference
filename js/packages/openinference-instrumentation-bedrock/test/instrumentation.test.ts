import { BedrockInstrumentation } from "../src/instrumentation";
import { 
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand 
} from "@aws-sdk/client-bedrock-runtime";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import nock from "nock";
import * as fs from "fs";
import * as path from "path";
import {
  generateToolResultMessage,
  generateToolCallMessage,
  commonTools,
} from "./helpers/test-data-generators";
import {
  loadRecordingData,
  createNockMock,
  sanitizeAuthHeaders,
  createTestClient,
  getRecordingPath,
} from "./helpers/vcr-helpers";
import {
  verifyResponseStructure,
  verifySpanBasics,
} from "./helpers/test-helpers";
import {
  TEST_MODEL_ID,
  TEST_USER_MESSAGE,
  TEST_MAX_TOKENS,
} from "./config/constants";

describe("BedrockInstrumentation", () => {
  let instrumentation: BedrockInstrumentation;
  let provider: NodeTracerProvider;
  let spanExporter: InMemorySpanExporter;
  let currentTestName: string;
  let recordingsPath: string;

  const isRecordingMode = process.env.BEDROCK_RECORD_MODE === "record";

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

  // Helper function for tests to set up their specific recording
  const setupTestRecording = (testName: string) => {
    currentTestName = testName;
    recordingsPath = getRecordingPath(testName, __dirname);

    if (!isRecordingMode) {
      // Replay mode: create mock from test-specific recording
      const recordingData = loadRecordingData(recordingsPath);

      if (recordingData?.response) {
        console.log(
          `Creating mock from sanitized recording: ${path.basename(recordingsPath)}`,
        );
        createNockMock(
          recordingData.response,
          recordingData.modelId || undefined,
          recordingData.status,
          TEST_MODEL_ID,
          recordingData.isStreaming,
        );
      } else {
        console.log(`No recordings found at ${recordingsPath}`);
      }
    }
  };

  beforeEach(() => {
    // Clear any existing nock mocks first
    nock.cleanAll();

    // Ensure nock is active (important for test isolation)
    if (!nock.isActive()) {
      nock.activate();
    }

    // Set default test name (will be overridden by setupTestRecording)
    currentTestName = "default-test";
    recordingsPath = getRecordingPath(currentTestName, __dirname);

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
      console.log(
        `Captured ${recordings.length} recordings for test: ${currentTestName}`,
      );
      if (recordings.length > 0) {
        // Sanitize auth headers - replace with mock credentials for replay compatibility
        sanitizeAuthHeaders(recordings);

        const recordingsDir = path.dirname(recordingsPath);
        if (!fs.existsSync(recordingsDir)) {
          fs.mkdirSync(recordingsDir, { recursive: true });
        }
        fs.writeFileSync(recordingsPath, JSON.stringify(recordings, null, 2));
        console.log(
          `Saved sanitized recordings to ${path.basename(recordingsPath)}`,
        );
      }
    }

    // Clean up nock only
    nock.cleanAll();
    nock.restore();
  });

  describe("InvokeModel basic instrumentation", () => {
    it("should create spans for InvokeModel calls", async () => {
      setupTestRecording("should create spans for InvokeModel calls");

      const client = createTestClient(isRecordingMode);

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
  "output.mime_type": "text/plain",
  "output.value": "Hello! As an AI language model, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?",
}
`);
    });

    it("should handle tool calling with function definitions", async () => {
      setupTestRecording(
        "should handle tool calling with function definitions",
      );

      const toolDefinition = {
        name: "get_weather",
        description: "Get current weather for a location",
        input_schema: {
          type: "object",
          properties: {
            location: { type: "string", description: "The city and state" },
          },
          required: ["location"],
        },
      };

      const client = createTestClient(isRecordingMode);

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
  "output.mime_type": "text/plain",
  "output.value": "Certainly! I can help you with that information. To get the current weather for San Francisco, I'll use the get_weather function. Let me fetch that data for you.",
}
`);
    });

    it("should handle tool result responses", async () => {
      setupTestRecording("should handle tool result responses");

      const client = createTestClient(isRecordingMode);

      // Use existing generateToolResultMessage() from test-data-generators.js
      const testData = generateToolResultMessage({
        initialPrompt: "What's the weather in Paris?",
        toolUseId: "toolu_123",
        toolName: "get_weather",
        toolInput: { location: "Paris, France" },
        toolResult: "The weather in Paris is currently 22°C and sunny.",
        followupPrompt: "Great! What should I wear?",
      });

      const command = new InvokeModelCommand({
        modelId: testData.modelId,
        body: testData.body,
        contentType: "application/json",
        accept: "application/json",
      });

      const result = await client.send(command);
      verifyResponseStructure(result);

      const span = verifySpanBasics(spanExporter);

      // Verify tool result processing attributes
      expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "What's the weather in Paris?",
  "llm.input_messages.0.message.content": "What's the weather in Paris?",
  "llm.input_messages.0.message.role": "user",
  "llm.input_messages.1.message.content": "[Tool Call: get_weather]",
  "llm.input_messages.1.message.role": "assistant",
  "llm.input_messages.2.message.content": "[Tool Result: The weather in Paris is currently 22°C and sunny.] Great! What should I wear?",
  "llm.input_messages.2.message.role": "user",
  "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments": "{"result":"The weather in Paris is currently 22°C and sunny."}",
  "llm.input_messages.2.message.tool_calls.0.tool_call.id": "toolu_123",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
  "llm.output_messages.0.message.content": "Since it's warm and sunny in Paris right now, you'll want to wear lightweight, breathable clothing. Some recommendations:

- A light shirt or tank top
- Shorts or a light skirt/dress
- Sandals or other open-toed shoes
- A hat or sunglasses for sun protection
- A light jacket or sweater in case it cools off in the evening

The key things are to dress for the warm temperatures and have layers you can",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "bedrock",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 364,
  "llm.token_count.total": 464,
  "llm.tools.0.tool.json_schema": "{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "text/plain",
  "output.value": "Since it's warm and sunny in Paris right now, you'll want to wear lightweight, breathable clothing. Some recommendations:

- A light shirt or tank top
- Shorts or a light skirt/dress
- Sandals or other open-toed shoes
- A hat or sunglasses for sun protection
- A light jacket or sweater in case it cools off in the evening

The key things are to dress for the warm temperatures and have layers you can",
}
`);
    });

    it("should handle missing token counts gracefully", async () => {
      setupTestRecording("should handle missing token counts gracefully");

      const client = createTestClient(isRecordingMode);

      const command = new InvokeModelCommand({
        modelId: TEST_MODEL_ID,
        body: JSON.stringify({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: TEST_MAX_TOKENS,
          messages: [
            {
              role: "user",
              content: "Tell me a short fact.",
            },
          ],
        }),
        contentType: "application/json",
        accept: "application/json",
      });

      const result = await client.send(command);
      verifyResponseStructure(result);

      const span = verifySpanBasics(spanExporter);

      // Verify that span completes successfully even without token counts
      expect(span.status.code).toBe(1); // SpanStatusCode.OK
      expect(span.attributes["llm.model_name"]).toBe(TEST_MODEL_ID);
      expect(span.attributes["llm.provider"]).toBe("aws");
      expect(span.attributes["llm.system"]).toBe("bedrock");
      expect(span.attributes["openinference.span.kind"]).toBe("LLM");

      // Basic message attributes should still be present
      expect(span.attributes["llm.input_messages.0.message.content"]).toBe(
        "Tell me a short fact.",
      );
      expect(span.attributes["llm.input_messages.0.message.role"]).toBe("user");

      // Output message should be captured
      expect(span.attributes["llm.output_messages.0.message.role"]).toBe(
        "assistant",
      );
      expect(
        span.attributes["llm.output_messages.0.message.content"],
      ).toBeDefined();

      // Token count attributes should either be undefined or gracefully handled
      // This test verifies the implementation doesn't crash when usage is missing
      const hasTokenCounts =
        span.attributes["llm.token_count.prompt"] !== undefined ||
        span.attributes["llm.token_count.completion"] !== undefined ||
        span.attributes["llm.token_count.total"] !== undefined;

      // If token counts are present, they should be valid numbers
      if (hasTokenCounts) {
        if (span.attributes["llm.token_count.prompt"] !== undefined) {
          expect(typeof span.attributes["llm.token_count.prompt"]).toBe(
            "number",
          );
        }
        if (span.attributes["llm.token_count.completion"] !== undefined) {
          expect(typeof span.attributes["llm.token_count.completion"]).toBe(
            "number",
          );
        }
        if (span.attributes["llm.token_count.total"] !== undefined) {
          expect(typeof span.attributes["llm.token_count.total"]).toBe(
            "number",
          );
        }
      }

      // Snapshot the attributes to verify token handling behavior
      expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "Tell me a short fact.",
  "llm.input_messages.0.message.content": "Tell me a short fact.",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "anthropic.claude-3-5-sonnet-20240620-v1:0",
  "llm.output_messages.0.message.content": "Here's a short fact for you:

Honeybees can recognize human faces.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "bedrock",
  "openinference.span.kind": "LLM",
  "output.mime_type": "text/plain",
  "output.value": "Here's a short fact for you:

Honeybees can recognize human faces.",
}
`);
    });

    it("should handle multi-modal messages with images", async () => {
      setupTestRecording("should handle multi-modal messages with images");

      const client = createTestClient(isRecordingMode);

      // Create a multi-modal message with text and image
      const imageData =
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";

      const command = new InvokeModelCommand({
        modelId: TEST_MODEL_ID,
        body: JSON.stringify({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: TEST_MAX_TOKENS,
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "What do you see in this image?",
                },
                {
                  type: "image",
                  source: {
                    type: "base64",
                    media_type: "image/png",
                    data: imageData,
                  },
                },
              ],
            },
          ],
        }),
        contentType: "application/json",
        accept: "application/json",
      });

      const result = await client.send(command);
      verifyResponseStructure(result);

      const span = verifySpanBasics(spanExporter);

      // Verify multi-modal message handling
      expect(span.attributes["llm.model_name"]).toBe(TEST_MODEL_ID);
      expect(span.attributes["llm.provider"]).toBe("aws");
      expect(span.attributes["llm.system"]).toBe("bedrock");
      expect(span.attributes["openinference.span.kind"]).toBe("LLM");

      // Check that input message content is properly handled
      expect(span.attributes["llm.input_messages.0.message.role"]).toBe("user");

      // Check for image URL formatting: data:image/png;base64,{data}
      const expectedImageUrl = `data:image/png;base64,${imageData}`;

      // The input.value should contain both text and image data
      expect(span.attributes["input.value"]).toContain(
        "What do you see in this image?",
      );
      expect(span.attributes["input.value"]).toContain(expectedImageUrl);

      // Verify that multi-modal content is properly extracted
      // Text content should be captured
      expect(span.attributes["llm.input_messages.0.message.content"]).toContain(
        "What do you see in this image?",
      );

      // Image content should be captured in OpenInference format
      // The message should contain the image URL in the expected format
      const messageContent =
        span.attributes["llm.input_messages.0.message.content"];
      expect(messageContent).toContain(expectedImageUrl);

      // Output message should be captured
      expect(span.attributes["llm.output_messages.0.message.role"]).toBe(
        "assistant",
      );
      expect(
        span.attributes["llm.output_messages.0.message.content"],
      ).toBeDefined();

      // Snapshot the attributes to verify multi-modal message processing
      expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "What do you see in this image? data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
  "llm.input_messages.0.message.content": "What do you see in this image? data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "anthropic.claude-3-5-sonnet-20240620-v1:0",
  "llm.output_messages.0.message.content": "This image appears to be a handwritten note or letter on lined paper. The writing is in cursive script and covers most of the visible page. While I can't make out specific words or content due to the resolution, the handwriting looks neat and consistent. The paper has a light yellow or cream color, which could indicate it's an older document or simply the natural color of the paper. There are horizontal blue lines visible, typical of standard lined notebook or writing paper. The overall impression is",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "bedrock",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 19,
  "llm.token_count.total": 119,
  "openinference.span.kind": "LLM",
  "output.mime_type": "text/plain",
  "output.value": "This image appears to be a handwritten note or letter on lined paper. The writing is in cursive script and covers most of the visible page. While I can't make out specific words or content due to the resolution, the handwriting looks neat and consistent. The paper has a light yellow or cream color, which could indicate it's an older document or simply the natural color of the paper. There are horizontal blue lines visible, typical of standard lined notebook or writing paper. The overall impression is",
}
`);
    });

    it("should handle API errors gracefully", async () => {
      setupTestRecording("should handle api errors gracefully");

      const client = createTestClient(isRecordingMode);

      // Test invalid model ID (should trigger 400 error)
      const invalidModelCommand = new InvokeModelCommand({
        modelId: "invalid-model-id",
        body: JSON.stringify({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: TEST_MAX_TOKENS,
          messages: [
            {
              role: "user",
              content: "This should fail",
            },
          ],
        }),
        contentType: "application/json",
        accept: "application/json",
      });

      // Expect the API call to throw an error
      await expect(client.send(invalidModelCommand)).rejects.toThrow();

      // Verify span was created and marked as error
      const span = verifySpanBasics(spanExporter);

      // Verify span status is set to ERROR
      expect(span.status.code).toBe(2); // SpanStatusCode.ERROR
      expect(span.status.message).toBeDefined();

      // Verify basic attributes are still captured
      expect(span.attributes["llm.model_name"]).toBe("invalid-model-id");
      expect(span.attributes["llm.provider"]).toBe("aws");
      expect(span.attributes["llm.system"]).toBe("bedrock");
      expect(span.attributes["openinference.span.kind"]).toBe("LLM");

      // Verify input message attributes are captured
      expect(span.attributes["llm.input_messages.0.message.content"]).toBe(
        "This should fail",
      );
      expect(span.attributes["llm.input_messages.0.message.role"]).toBe("user");

      // Verify error details are recorded
      expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "This should fail",
  "llm.input_messages.0.message.content": "This should fail",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "invalid-model-id",
  "llm.provider": "aws",
  "llm.system": "bedrock",
  "openinference.span.kind": "LLM",
}
`);
    });

    it("should handle multiple tools in single request", async () => {
      setupTestRecording("should handle multiple tools in single request");

      const client = createTestClient(isRecordingMode);

      const testData = generateToolCallMessage({
        prompt: "What's the weather in San Francisco and what's 15 * 23?",
        tools: [
          commonTools.weather,
          commonTools.calculator,
          commonTools.webSearch,
        ],
      });

      const command = new InvokeModelCommand({
        modelId: testData.modelId,
        body: testData.body,
        contentType: "application/json",
        accept: "application/json",
      });

      const result = await client.send(command);
      verifyResponseStructure(result);

      const span = verifySpanBasics(spanExporter);
      expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "What's the weather in San Francisco and what's 15 * 23?",
  "llm.input_messages.0.message.content": "What's the weather in San Francisco and what's 15 * 23?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
  "llm.output_messages.0.message.content": "Okay, let's get the weather and do that calculation.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{"location":"San Francisco, CA","unit":"fahrenheit"}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "toolu_bdrk_01FqpV1qX3bJ4bczkdtMhdGz",
  "llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments": "{}",
  "llm.output_messages.0.message.tool_calls.1.tool_call.function.name": "calculate",
  "llm.output_messages.0.message.tool_calls.1.tool_call.id": "toolu_bdrk_01KMAmxrwkK8jh8h6YNQ495y",
  "llm.provider": "aws",
  "llm.system": "bedrock",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 435,
  "llm.token_count.total": 535,
  "llm.tools.0.tool.json_schema": "{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}}",
  "llm.tools.1.tool.json_schema": "{"type":"function","function":{"name":"calculate","description":"Perform mathematical calculations","parameters":{"type":"object","properties":{"expression":{"type":"string","description":"Mathematical expression to evaluate"}},"required":["expression"]}}}",
  "llm.tools.2.tool.json_schema": "{"type":"function","function":{"name":"web_search","description":"Search the web for information","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"num_results":{"type":"integer","description":"Number of results to return","minimum":1,"maximum":10}},"required":["query"]}}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "text/plain",
  "output.value": "Okay, let's get the weather and do that calculation.",
}
`);
    });
  });

  describe("InvokeModelWithResponseStream support", () => {
    it("should handle InvokeModelWithResponseStream", async () => {
      setupTestRecording("should handle invoke model with response stream");

      const client = createTestClient(isRecordingMode);

      const command = new InvokeModelWithResponseStreamCommand({
        modelId: TEST_MODEL_ID,
        body: JSON.stringify({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: 100,
          messages: [{ role: "user", content: "Tell me a short story" }]
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
  "input.value": "Tell me a short story",
  "llm.input_messages.0.message.content": "Tell me a short story",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "anthropic.claude-3-5-sonnet-20240620-v1:0",
  "llm.output_messages.0.message.content": "Here's a short story for you:

The Last Leaf

Ella gazed out her window at the ivy-covered wall across the courtyard. She had been bedridden with pneumonia for weeks, and her spirits were low. The doctor had told her that she needed the will to live to recover, but Ella felt herself slipping away.

She had been counting the ivy leaves as they fell, convinced that when the last leaf dropped, she too would die. Now",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "bedrock",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 12,
  "llm.token_count.total": 112,
  "openinference.span.kind": "LLM",
  "output.mime_type": "text/plain",
  "output.value": "Here's a short story for you:

The Last Leaf

Ella gazed out her window at the ivy-covered wall across the courtyard. She had been bedridden with pneumonia for weeks, and her spirits were low. The doctor had told her that she needed the will to live to recover, but Ella felt herself slipping away.

She had been counting the ivy leaves as they fell, convinced that when the last leaf dropped, she too would die. Now",
}
`);
    });

    it("should handle streaming responses with tool calls", async () => {
      setupTestRecording("should handle streaming responses with tool calls");

      const client = createTestClient(isRecordingMode);

      const testData = generateToolCallMessage({
        prompt: "What's the weather in San Francisco?",
        tools: [commonTools.weather],
      });

      const command = new InvokeModelWithResponseStreamCommand({
        modelId: testData.modelId,
        body: testData.body,
        contentType: "application/json",
        accept: "application/json",
      });

      const result = await client.send(command);
      verifyResponseStructure(result);

      const span = verifySpanBasics(spanExporter);
      expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "What's the weather in San Francisco?",
  "llm.input_messages.0.message.content": "What's the weather in San Francisco?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "anthropic.claude-3-sonnet-20240229-v1:0",
  "llm.output_messages.0.message.content": "Okay, let's get the current weather for San Francisco:",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "toolu_bdrk_01SmuLWbQxzvE6WD3Th711eg",
  "llm.provider": "aws",
  "llm.system": "bedrock",
  "llm.token_count.completion": 88,
  "llm.token_count.prompt": 273,
  "llm.token_count.total": 361,
  "llm.tools.0.tool.json_schema": "{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "text/plain",
  "output.value": "Okay, let's get the current weather for San Francisco:",
}
`);
    });

    it("should handle streaming errors gracefully", async () => {
      setupTestRecording("should handle streaming errors");

      const client = createTestClient(isRecordingMode);

      // Test invalid model ID with streaming (should trigger error)
      const invalidModelCommand = new InvokeModelWithResponseStreamCommand({
        modelId: "invalid-streaming-model-id",
        body: JSON.stringify({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: TEST_MAX_TOKENS,
          messages: [
            {
              role: "user",
              content: "This streaming request should fail",
            },
          ],
        }),
        contentType: "application/json",
        accept: "application/json",
      });

      // Expect the API call to throw an error
      await expect(client.send(invalidModelCommand)).rejects.toThrow();

      // Verify span was created and marked as error
      const span = verifySpanBasics(spanExporter);

      // Verify span status is set to ERROR
      expect(span.status.code).toBe(2); // SpanStatusCode.ERROR
      expect(span.status.message).toBeDefined();

      // Verify basic attributes are still captured
      expect(span.attributes["llm.model_name"]).toBe("invalid-streaming-model-id");
      expect(span.attributes["llm.provider"]).toBe("aws");
      expect(span.attributes["llm.system"]).toBe("bedrock");
      expect(span.attributes["openinference.span.kind"]).toBe("LLM");

      // Verify input message attributes are captured
      expect(span.attributes["llm.input_messages.0.message.content"]).toBe(
        "This streaming request should fail",
      );
      expect(span.attributes["llm.input_messages.0.message.role"]).toBe("user");

      // Verify error details are recorded
      expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "This streaming request should fail",
  "llm.input_messages.0.message.content": "This streaming request should fail",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "invalid-streaming-model-id",
  "llm.provider": "aws",
  "llm.system": "bedrock",
  "openinference.span.kind": "LLM",
}
`);
    });

    // TODO: Add more test scenarios following the TDD plan:
    //
    // Phase 1: InvokeModel Foundation ✅ COMPLETE (8/8 tests)
    // ✅ Basic InvokeModel Text Messages
    // ✅ Tool Call Support - Basic Function Call
    // ✅ Tool Results Processing
    // ✅ Missing Token Count Handling
    // ✅ Multi-Modal Messages (Text + Image)
    // ✅ API Error Handling
    // ✅ Multiple Tools in Single Request
    // ✅ InvokeModelWithResponseStream - Basic Text
    // ✅ Streaming Tool Calls
    // ✅ Stream Error Handling
    //
    // Phase 3: Converse API Support
    // - it("should handle Converse API calls", async () => {})
    // - it("should handle system prompts in Converse API", async () => {})
    // - it("should handle tool calling via Converse API", async () => {})
    //
    // Use pattern: BEDROCK_RECORD_MODE=record npm test -- --testNamePattern="test name"
  });
});

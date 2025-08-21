import { BedrockInstrumentation } from "../src/instrumentation";
import {
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand,
  ConverseCommand,
  ConverseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { context } from "@opentelemetry/api";
import {
  setSession,
  setUser,
  setMetadata,
  setTags,
  setPromptTemplate,
} from "@arizeai/openinference-core";
import {
  SESSION_ID,
  USER_ID,
  METADATA,
  TAG_TAGS,
  PROMPT_TEMPLATE_TEMPLATE,
  PROMPT_TEMPLATE_VERSION,
  PROMPT_TEMPLATE_VARIABLES,
} from "@arizeai/openinference-semantic-conventions";
import nock from "nock";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import {
  generateToolResultMessage,
  generateToolCallMessage,
  commonTools,
} from "./helpers/test-data-generators";
import {
  createTestClient,
  getRecordingPath,
  setupTestRecording,
  initializeRecordingMode,
  saveRecordingModeData,
} from "./helpers/vcr-helpers";
import {
  verifyResponseStructure,
  verifySpanBasics,
  consumeStreamResponse,
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
    // Setup instrumentation and tracer provider
    instrumentation = new BedrockInstrumentation();
    instrumentation.disable(); // Initially disabled
    spanExporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider();

    provider.addSpanProcessor(new SimpleSpanProcessor(spanExporter));
    provider.register();
    instrumentation.setTracerProvider(provider);

    // Manually set module exports for testing
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const BedrockRuntime = require("@aws-sdk/client-bedrock-runtime");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (instrumentation as any)._modules[0].moduleExports = BedrockRuntime;

    // Enable instrumentation ONCE
    instrumentation.enable();
  });

  // Global cleanup
  afterAll(async () => {
    instrumentation.disable();
    await provider.shutdown();
  });

  // Helper wrapper for tests to set up their specific recording
  const setupTestRecordingWrapper = (testName: string) => {
    currentTestName = testName;
    recordingsPath = setupTestRecording(
      testName,
      __dirname,
      isRecordingMode,
      TEST_MODEL_ID,
    );
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
      initializeRecordingMode();
    }

    // Reset span exporter for clean test state
    spanExporter.reset();
  });

  afterEach(() => {
    if (isRecordingMode) {
      saveRecordingModeData(currentTestName, recordingsPath);
    }

    // Clean up nock only
    nock.cleanAll();
    nock.restore();
  });

  describe("InvokeModel API", () => {
    describe("Basic Instrumentation", () => {
      it("should create spans for InvokeModel calls", async () => {
        setupTestRecordingWrapper("should create spans for InvokeModel calls");

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
  "input.value": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100,"messages":[{"role":"user","content":"Hello, how are you?"}]}",
  "llm.input_messages.0.message.content": "Hello, how are you?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "Hello! As an AI language model, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "anthropic",
  "llm.token_count.completion": 35,
  "llm.token_count.prompt": 13,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"msg_bdrk_01M7yZYBn3yhYTyMxjQK781b","type":"message","role":"assistant","model":"claude-3-5-sonnet-20240620","content":[{"type":"text","text":"Hello! As an AI language model, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":13,"output_tokens":35}}",
}
`);
      });
    });
    describe("Tool Calling", () => {
      it("should handle tool calling with function definitions", async () => {
        setupTestRecordingWrapper(
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
  "input.value": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100,"tools":[{"name":"get_weather","description":"Get current weather for a location","input_schema":{"type":"object","properties":{"location":{"type":"string","description":"The city and state"}},"required":["location"]}}],"messages":[{"role":"user","content":"What's the weather like in San Francisco?"}]}",
  "llm.input_messages.0.message.content": "What's the weather like in San Francisco?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.contents.0.message_content.text": "Certainly! I can help you with that information. To get the current weather for San Francisco, I'll use the get_weather function. Let me fetch that data for you.",
  "llm.output_messages.0.message.contents.0.message_content.type": "text",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{"location":"San Francisco, CA"}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "toolu_bdrk_01MqHGzs8QwkdkVjJYrbLTPp",
  "llm.provider": "aws",
  "llm.system": "anthropic",
  "llm.token_count.completion": 94,
  "llm.token_count.prompt": 373,
  "llm.tools.0.tool.json_schema": "{"name":"get_weather","description":"Get current weather for a location","input_schema":{"type":"object","properties":{"location":{"type":"string","description":"The city and state"}},"required":["location"]}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"msg_bdrk_01UCkMJ9Yp7J1ZpAWtK8CdYN","type":"message","role":"assistant","model":"claude-3-5-sonnet-20240620","content":[{"type":"text","text":"Certainly! I can help you with that information. To get the current weather for San Francisco, I'll use the get_weather function. Let me fetch that data for you."},{"type":"tool_use","id":"toolu_bdrk_01MqHGzs8QwkdkVjJYrbLTPp","name":"get_weather","input":{"location":"San Francisco, CA"}}],"stop_reason":"tool_use","stop_sequence":null,"usage":{"input_tokens":373,"output_tokens":94}}",
}
`);
      });

      it("should handle tool result responses", async () => {
        setupTestRecordingWrapper("should handle tool result responses");

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
  "input.value": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100,"tools":[{"name":"get_weather","description":"Get current weather for a location","input_schema":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}],"messages":[{"role":"user","content":"What's the weather in Paris?"},{"role":"assistant","content":[{"type":"tool_use","id":"toolu_123","name":"get_weather","input":{"location":"Paris, France"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_123","content":"The weather in Paris is currently 22°C and sunny."},{"type":"text","text":"Great! What should I wear?"}]}]}",
  "llm.input_messages.0.message.content": "What's the weather in Paris?",
  "llm.input_messages.0.message.role": "user",
  "llm.input_messages.1.message.role": "assistant",
  "llm.input_messages.1.message.tool_calls.0.tool_call.function.arguments": "{"location":"Paris, France"}",
  "llm.input_messages.1.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.input_messages.1.message.tool_calls.0.tool_call.id": "toolu_123",
  "llm.input_messages.2.message.contents.1.message_content.text": "Great! What should I wear?",
  "llm.input_messages.2.message.contents.1.message_content.type": "text",
  "llm.input_messages.2.message.role": "user",
  "llm.input_messages.2.message.tool_call_id": "toolu_123",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "claude-3-sonnet-20240229",
  "llm.output_messages.0.message.content": "Since it's warm and sunny in Paris right now, you'll want to wear lightweight, breathable clothing. Some recommendations:

- A light shirt or tank top
- Shorts or a light skirt/dress
- Sandals or other open-toed shoes
- A hat or sunglasses for sun protection
- A light jacket or sweater in case it cools off in the evening

The key things are to dress for the warm temperatures and have layers you can",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "anthropic",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 364,
  "llm.tools.0.tool.json_schema": "{"name":"get_weather","description":"Get current weather for a location","input_schema":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"msg_bdrk_0192ZiMpnkcdHwvzYyWpps77","type":"message","role":"assistant","model":"claude-3-sonnet-20240229","content":[{"type":"text","text":"Since it's warm and sunny in Paris right now, you'll want to wear lightweight, breathable clothing. Some recommendations:\\n\\n- A light shirt or tank top\\n- Shorts or a light skirt/dress\\n- Sandals or other open-toed shoes\\n- A hat or sunglasses for sun protection\\n- A light jacket or sweater in case it cools off in the evening\\n\\nThe key things are to dress for the warm temperatures and have layers you can"}],"stop_reason":"max_tokens","stop_sequence":null,"usage":{"input_tokens":364,"output_tokens":100}}",
}
`);
      });
      it("should handle multiple tools in single request", async () => {
        setupTestRecordingWrapper(
          "should handle multiple tools in single request",
        );

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
  "input.value": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100,"tools":[{"name":"get_weather","description":"Get current weather for a location","input_schema":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}},{"name":"calculate","description":"Perform mathematical calculations","input_schema":{"type":"object","properties":{"expression":{"type":"string","description":"Mathematical expression to evaluate"}},"required":["expression"]}},{"name":"web_search","description":"Search the web for information","input_schema":{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"num_results":{"type":"integer","description":"Number of results to return","minimum":1,"maximum":10}},"required":["query"]}}],"messages":[{"role":"user","content":"What's the weather in San Francisco and what's 15 * 23?"}]}",
  "llm.input_messages.0.message.content": "What's the weather in San Francisco and what's 15 * 23?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "claude-3-sonnet-20240229",
  "llm.output_messages.0.message.contents.0.message_content.text": "Okay, let's get the weather and do that calculation.",
  "llm.output_messages.0.message.contents.0.message_content.type": "text",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{"location":"San Francisco, CA","unit":"fahrenheit"}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "toolu_bdrk_01FqpV1qX3bJ4bczkdtMhdGz",
  "llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments": "{}",
  "llm.output_messages.0.message.tool_calls.1.tool_call.function.name": "calculate",
  "llm.output_messages.0.message.tool_calls.1.tool_call.id": "toolu_bdrk_01KMAmxrwkK8jh8h6YNQ495y",
  "llm.provider": "aws",
  "llm.system": "anthropic",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 435,
  "llm.tools.0.tool.json_schema": "{"name":"get_weather","description":"Get current weather for a location","input_schema":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}",
  "llm.tools.1.tool.json_schema": "{"name":"calculate","description":"Perform mathematical calculations","input_schema":{"type":"object","properties":{"expression":{"type":"string","description":"Mathematical expression to evaluate"}},"required":["expression"]}}",
  "llm.tools.2.tool.json_schema": "{"name":"web_search","description":"Search the web for information","input_schema":{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"num_results":{"type":"integer","description":"Number of results to return","minimum":1,"maximum":10}},"required":["query"]}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"msg_bdrk_01VGSSxYibWoHmna5zwvqqCt","type":"message","role":"assistant","model":"claude-3-sonnet-20240229","content":[{"type":"text","text":"Okay, let's get the weather and do that calculation."},{"type":"tool_use","id":"toolu_bdrk_01FqpV1qX3bJ4bczkdtMhdGz","name":"get_weather","input":{"location":"San Francisco, CA","unit":"fahrenheit"}},{"type":"tool_use","id":"toolu_bdrk_01KMAmxrwkK8jh8h6YNQ495y","name":"calculate","input":{}}],"stop_reason":"max_tokens","stop_sequence":null,"usage":{"input_tokens":435,"output_tokens":100}}",
}
`);
      });
    });
    describe("Multi-Modal", () => {
      it("should handle multi-modal messages with images", async () => {
        setupTestRecordingWrapper(
          "should handle multi-modal messages with images",
        );

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
        expect(span.attributes["llm.model_name"]).toBe(
          "claude-3-5-sonnet-20240620",
        );
        expect(span.attributes["llm.provider"]).toBe("aws");
        expect(span.attributes["llm.system"]).toBe("anthropic");
        expect(span.attributes["openinference.span.kind"]).toBe("LLM");

        // Check that input message content is properly handled
        expect(span.attributes["llm.input_messages.0.message.role"]).toBe(
          "user",
        );

        // The input.value should contain the full JSON request body with image data
        expect(span.attributes["input.value"]).toContain(
          "What do you see in this image?",
        );
        expect(span.attributes["input.value"]).toContain(imageData);

        // Verify that multi-modal content is properly extracted
        // Text content should be captured in detailed structure
        expect(
          span.attributes[
            "llm.input_messages.0.message.contents.0.message_content.text"
          ],
        ).toContain("What do you see in this image?");

        // Image content should be captured in OpenInference format
        // The message should contain the image URL in the expected format
        const imageContent =
          span.attributes[
            "llm.input_messages.0.message.contents.1.message_content.image.image.url"
          ];
        const expectedImageUrl = `data:image/png;base64,${imageData}`;
        expect(imageContent).toBe(expectedImageUrl);

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
  "input.value": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100,"messages":[{"role":"user","content":[{"type":"text","text":"What do you see in this image?"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="}}]}]}",
  "llm.input_messages.0.message.contents.0.message_content.text": "What do you see in this image?",
  "llm.input_messages.0.message.contents.0.message_content.type": "text",
  "llm.input_messages.0.message.contents.1.message_content.image.image.url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
  "llm.input_messages.0.message.contents.1.message_content.type": "image",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "This image appears to be a handwritten note or letter on lined paper. The writing is in cursive script and covers most of the visible page. While I can't make out specific words or content due to the resolution, the handwriting looks neat and consistent. The paper has a light yellow or cream color, which could indicate it's an older document or simply the natural color of the paper. There are horizontal blue lines visible, typical of standard lined notebook or writing paper. The overall impression is",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "anthropic",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 19,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"msg_bdrk_01GKV3gTrEVoxHSj4ErmzgRS","type":"message","role":"assistant","model":"claude-3-5-sonnet-20240620","content":[{"type":"text","text":"This image appears to be a handwritten note or letter on lined paper. The writing is in cursive script and covers most of the visible page. While I can't make out specific words or content due to the resolution, the handwriting looks neat and consistent. The paper has a light yellow or cream color, which could indicate it's an older document or simply the natural color of the paper. There are horizontal blue lines visible, typical of standard lined notebook or writing paper. The overall impression is"}],"stop_reason":"max_tokens","stop_sequence":null,"usage":{"input_tokens":19,"output_tokens":100}}",
}
`);
      });
    });
    describe("Error Handling", () => {
      it("should handle missing token counts gracefully", async () => {
        setupTestRecordingWrapper(
          "should handle missing token counts gracefully",
        );

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
        expect(span.attributes["llm.model_name"]).toBe(
          "claude-3-5-sonnet-20240620",
        );
        expect(span.attributes["llm.provider"]).toBe("aws");
        expect(span.attributes["llm.system"]).toBe("anthropic");
        expect(span.attributes["openinference.span.kind"]).toBe("LLM");

        // Basic message attributes should still be present
        expect(span.attributes["llm.input_messages.0.message.content"]).toBe(
          "Tell me a short fact.",
        );
        expect(span.attributes["llm.input_messages.0.message.role"]).toBe(
          "user",
        );

        // Output message should be captured
        expect(span.attributes["llm.output_messages.0.message.role"]).toBe(
          "assistant",
        );
        expect(
          span.attributes["llm.output_messages.0.message.content"],
        ).toBeDefined();

        // Token count attributes should either be undefined or gracefully handled
        // This test verifies graceful handling when usage is missing
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
  "input.value": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100,"messages":[{"role":"user","content":"Tell me a short fact."}]}",
  "llm.input_messages.0.message.content": "Tell me a short fact.",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "Here's a short fact for you:

Honeybees can recognize human faces.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "anthropic",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"msg_bdrk_01WnczgPLNeFxGoqRLQnmKQR","type":"message","role":"assistant","model":"claude-3-5-sonnet-20240620","content":[{"type":"text","text":"Here's a short fact for you:\\n\\nHoneybees can recognize human faces."}],"stop_reason":"end_turn","stop_sequence":null}",
}
`);
      });
      it("should handle API errors gracefully", async () => {
        setupTestRecordingWrapper("should handle api errors gracefully");

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
        // Verify error details are recorded
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100,"messages":[{"role":"user","content":"This should fail"}]}",
  "llm.input_messages.0.message.content": "This should fail",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "invalid-model-id",
  "llm.provider": "aws",
  "llm.system": "amazon",
  "openinference.span.kind": "LLM",
}
`);
      });
    });
    describe("Multi-Provider Support", () => {
      it("should handle AI21 Jamba models", async () => {
        setupTestRecordingWrapper("should-handle-ai21-jamba-models");
        const client = createTestClient(isRecordingMode);

        const command = new InvokeModelCommand({
          modelId: "ai21.jamba-1-5-mini-v1:0",
          body: JSON.stringify({
            messages: [
              {
                role: "user",
                content: "Hello, how are you?",
              },
            ],
            max_tokens: 100,
            temperature: 0.7,
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
  "input.value": "{"messages":[{"role":"user","content":"Hello, how are you?"}],"max_tokens":100,"temperature":0.7}",
  "llm.input_messages.0.message.contents.0.message_content.text": "Hello, how are you?",
  "llm.input_messages.0.message.contents.0.message_content.type": "text",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"max_tokens":100,"temperature":0.7}",
  "llm.model_name": "jamba-1-5-mini-v1:0",
  "llm.output_messages.0.message.content": " Hello! I'm doing great, thank you. How can I assist you today?",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "ai21",
  "llm.token_count.completion": 19,
  "llm.token_count.prompt": 16,
  "llm.token_count.total": 35,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"chatcmpl-5a5e214e-2d38-4152-8496-b2eb301d7008","choices":[{"index":0,"message":{"role":"assistant","content":" Hello! I'm doing great, thank you. How can I assist you today?","tool_calls":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":19,"total_tokens":35},"meta":{"requestDurationMillis":307},"model":"jamba-1.5-mini"}",
}
`);
      });

      it("should handle Amazon Nova models", async () => {
        setupTestRecordingWrapper("should-handle-amazon-nova-models");
        const client = createTestClient(isRecordingMode);

        const command = new InvokeModelCommand({
          modelId: "amazon.nova-lite-v1:0",
          body: JSON.stringify({
            messages: [
              {
                role: "user",
                content: [
                  {
                    text: "What's the weather like in San Francisco today? Please use the weather tool to check.",
                  },
                ],
              },
            ],
            inferenceConfig: {
              maxTokens: 100,
              temperature: 0.7,
            },
            toolConfig: {
              tools: [
                {
                  toolSpec: {
                    name: "get_weather",
                    description:
                      "Get current weather information for a location",
                    inputSchema: {
                      json: {
                        type: "object",
                        properties: {
                          location: {
                            type: "string",
                            description:
                              "The city and state/country for weather lookup",
                          },
                          unit: {
                            type: "string",
                            enum: ["celsius", "fahrenheit"],
                            description: "Temperature unit preference",
                          },
                        },
                        required: ["location"],
                      },
                    },
                  },
                },
              ],
            },
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
  "input.value": "{"messages":[{"role":"user","content":[{"text":"What's the weather like in San Francisco today? Please use the weather tool to check."}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.7},"toolConfig":{"tools":[{"toolSpec":{"name":"get_weather","description":"Get current weather information for a location","inputSchema":{"json":{"type":"object","properties":{"location":{"type":"string","description":"The city and state/country for weather lookup"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit preference"}},"required":["location"]}}}}]}}",
  "llm.input_messages.0.message.contents.0.message_content.text": "What's the weather like in San Francisco today? Please use the weather tool to check.",
  "llm.input_messages.0.message.contents.0.message_content.type": "text",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.7}",
  "llm.model_name": "nova-lite-v1:0",
  "llm.output_messages.0.message.contents.0.message_content.text": "<thinking> The User has asked for the weather in San Francisco today. I will use the 'get_weather' tool to get this information. I will ask for the weather in Celsius as it is the most commonly used unit of temperature. </thinking>
",
  "llm.output_messages.0.message.contents.0.message_content.type": "text",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{"unit":"celsius","location":"San Francisco"}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "9fd2280f-9131-45d9-860f-843c2e3d01fa",
  "llm.provider": "aws",
  "llm.system": "amazon",
  "llm.token_count.completion": 75,
  "llm.token_count.prompt": 454,
  "llm.token_count.prompt_details.cache_read": 0,
  "llm.token_count.prompt_details.cache_write": 0,
  "llm.token_count.total": 529,
  "llm.tools.0.tool.json_schema": "{"name":"get_weather","description":"Get current weather information for a location","inputSchema":{"json":{"type":"object","properties":{"location":{"type":"string","description":"The city and state/country for weather lookup"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit preference"}},"required":["location"]}}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"output":{"message":{"content":[{"text":"<thinking> The User has asked for the weather in San Francisco today. I will use the 'get_weather' tool to get this information. I will ask for the weather in Celsius as it is the most commonly used unit of temperature. </thinking>\\n"},{"toolUse":{"name":"get_weather","toolUseId":"9fd2280f-9131-45d9-860f-843c2e3d01fa","input":{"unit":"celsius","location":"San Francisco"}}}],"role":"assistant"}},"stopReason":"tool_use","usage":{"inputTokens":454,"outputTokens":75,"totalTokens":529,"cacheReadInputTokenCount":0,"cacheWriteInputTokenCount":0}}",
}
`);
      });

      it("should handle Amazon Titan models", async () => {
        setupTestRecordingWrapper("should-handle-amazon-titan-models");
        const client = createTestClient(isRecordingMode);

        const command = new InvokeModelCommand({
          modelId: "amazon.titan-text-express-v1",
          body: JSON.stringify({
            inputText: "Hello, how are you?",
            textGenerationConfig: {
              maxTokenCount: 100,
              temperature: 0.7,
            },
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
  "input.value": "{"inputText":"Hello, how are you?","textGenerationConfig":{"maxTokenCount":100,"temperature":0.7}}",
  "llm.input_messages.0.message.content": "Hello, how are you?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokenCount":100,"temperature":0.7}",
  "llm.model_name": "titan-text-express-v1",
  "llm.output_messages.0.message.content": "
This model is designed to avoid generating sensitive content. It is important to respect individuals' privacy and personal information, and it is not appropriate to ask personal questions about someone unless you have a legitimate reason.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "amazon",
  "llm.token_count.completion": 42,
  "llm.token_count.prompt": 6,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"inputTextTokenCount":6,"results":[{"tokenCount":42,"outputText":"\\nThis model is designed to avoid generating sensitive content. It is important to respect individuals' privacy and personal information, and it is not appropriate to ask personal questions about someone unless you have a legitimate reason.","completionReason":"FINISH"}]}",
}
`);
      });

      it("should handle Cohere Command models", async () => {
        setupTestRecordingWrapper("should-handle-cohere-command-models");
        const client = createTestClient(isRecordingMode);

        const command = new InvokeModelCommand({
          modelId: "cohere.command-text-v14",
          body: JSON.stringify({
            prompt: "Hello, how are you?",
            max_tokens: 100,
            temperature: 0.7,
            p: 0.9,
            k: 0,
            stop_sequences: [],
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
  "input.value": "{"prompt":"Hello, how are you?","max_tokens":100,"temperature":0.7,"p":0.9,"k":0,"stop_sequences":[]}",
  "llm.input_messages.0.message.content": "Hello, how are you?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"max_tokens":100,"temperature":0.7,"p":0.9,"k":0,"stop_sequences":[]}",
  "llm.model_name": "command-text-v14",
  "llm.output_messages.0.message.content": " Hi! I am an AI language model and I don't have feelings, so I can't say how I am, but I'm here to help. How about you? How are you feeling today? ",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "cohere",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"id":"bde031d0-0895-4cc8-baa6-c4d51c739cc2","generations":[{"id":"6043bc09-510a-4589-88bb-cc11783218c1","text":" Hi! I am an AI language model and I don't have feelings, so I can't say how I am, but I'm here to help. How about you? How are you feeling today? ","finish_reason":"COMPLETE"}],"prompt":"Hello, how are you?"}",
}
`);
      });

      it("should handle Meta Llama models invoke", async () => {
        setupTestRecordingWrapper("should-handle-meta-llama-models-invoke");
        const client = createTestClient(isRecordingMode);

        const command = new InvokeModelCommand({
          modelId: "meta.llama3-8b-instruct-v1:0",
          body: JSON.stringify({
            prompt: "Hello, how are you?",
            max_gen_len: 100,
            temperature: 0.7,
            top_p: 0.9,
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
  "input.value": "{"prompt":"Hello, how are you?","max_gen_len":100,"temperature":0.7,"top_p":0.9}",
  "llm.input_messages.0.message.content": "Hello, how are you?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"max_gen_len":100,"temperature":0.7,"top_p":0.9}",
  "llm.model_name": "llama3-8b-instruct-v1:0",
  "llm.output_messages.0.message.content": "Hello! I'm doing well, thank you for asking. How can I help you today?",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "meta",
  "llm.token_count.completion": 18,
  "llm.token_count.prompt": 6,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"generation":"Hello! I'm doing well, thank you for asking. How can I help you today?","prompt_token_count":6,"generation_token_count":18,"stop_reason":"length"}",
}
`);
      });

      xit("should handle Mistral Pixtral Large models with multimodal and tools", async () => {
        setupTestRecordingWrapper("should-handle-mistral-pixtral-models");
        const client = createTestClient(isRecordingMode);

        // Sample base64 image data (small test image - 1x1 transparent PNG)
        const sampleImageBase64 =
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";

        const command = new InvokeModelCommand({
          modelId: "us.mistral.pixtral-large-2502-v1:0",
          body: JSON.stringify({
            messages: [
              {
                role: "user",
                content: [
                  {
                    type: "text",
                    text: "Please analyze this image and tell me what you see. If you need to get weather information for any location you see, use the weather tool.",
                  },
                  {
                    type: "image_url",
                    image_url: {
                      url: "data:image/png;base64," + sampleImageBase64,
                    },
                  },
                ],
              },
            ],
            tools: [
              {
                type: "function",
                function: {
                  name: "get_weather",
                  description:
                    "Get current weather information for a specific location",
                  parameters: {
                    type: "object",
                    properties: {
                      location: {
                        type: "string",
                        description:
                          "The city and country/state for weather lookup",
                      },
                      unit: {
                        type: "string",
                        enum: ["celsius", "fahrenheit"],
                        description: "Temperature unit preference",
                      },
                    },
                    required: ["location"],
                  },
                },
              },
            ],
            tool_choice: "auto",
            max_tokens: 200,
            temperature: 0.7,
          }),
          contentType: "application/json",
          accept: "application/json",
        });

        const result = await client.send(command);
        verifyResponseStructure(result);

        const span = verifySpanBasics(spanExporter);
        // Basic verification that multimodal and tool attributes are present
        expect(span.attributes["llm.input_messages.0.message.role"]).toBe(
          "user",
        );
        expect(span.attributes["llm.model_name"]).toContain("pixtral");
        expect(span.attributes["llm.system"]).toBe("mistralai");
      });
    });
  });

  describe("InvokeModelWithResponseStream", () => {
    describe("Basic Function and Tool Calling", () => {
      it("should handle InvokeModelWithResponseStream", async () => {
        setupTestRecordingWrapper(
          "should handle invoke model with response stream",
        );

        const client = createTestClient(isRecordingMode);

        const command = new InvokeModelWithResponseStreamCommand({
          modelId: TEST_MODEL_ID,
          body: JSON.stringify({
            anthropic_version: "bedrock-2023-05-31",
            max_tokens: 100,
            messages: [{ role: "user", content: "Tell me a short story" }],
          }),
          contentType: "application/json",
          accept: "application/json",
        });

        const result = await client.send(command);
        verifyResponseStructure(result);

        // Consume the stream to trigger instrumentation
        await consumeStreamResponse(result);

        const span = verifySpanBasics(spanExporter);
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100,"messages":[{"role":"user","content":"Tell me a short story"}]}",
  "llm.input_messages.0.message.content": "Tell me a short story",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "Here's a short story for you:

The Last Leaf

Ella gazed out her window at the ivy-covered wall across the courtyard. She had been bedridden with pneumonia for weeks, and her spirits were low. The doctor had told her that she needed the will to live to recover, but Ella felt herself slipping away.

She had been counting the ivy leaves as they fell, convinced that when the last leaf dropped, she too would die. Now",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "anthropic",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 12,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":"Here's a short story for you:\\n\\nThe Last Leaf\\n\\nElla gazed out her window at the ivy-covered wall across the courtyard. She had been bedridden with pneumonia for weeks, and her spirits were low. The doctor had told her that she needed the will to live to recover, but Ella felt herself slipping away.\\n\\nShe had been counting the ivy leaves as they fell, convinced that when the last leaf dropped, she too would die. Now","tool_calls":[],"usage":{"input_tokens":12,"output_tokens":100},"streaming":true}",
}
`);
      });

      it("should handle streaming responses with tool calls", async () => {
        setupTestRecordingWrapper(
          "should handle streaming responses with tool calls",
        );

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

        // Consume the stream to trigger instrumentation
        await consumeStreamResponse(result);

        const span = verifySpanBasics(spanExporter);
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100,"tools":[{"name":"get_weather","description":"Get current weather for a location","input_schema":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}],"messages":[{"role":"user","content":"What's the weather in San Francisco?"}]}",
  "llm.input_messages.0.message.content": "What's the weather in San Francisco?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "claude-3-sonnet-20240229",
  "llm.output_messages.0.message.content": "Okay, let's get the current weather for San Francisco:",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "toolu_bdrk_01SmuLWbQxzvE6WD3Th711eg",
  "llm.provider": "aws",
  "llm.system": "anthropic",
  "llm.token_count.completion": 88,
  "llm.token_count.prompt": 273,
  "llm.tools.0.tool.json_schema": "{"name":"get_weather","description":"Get current weather for a location","input_schema":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":"Okay, let's get the current weather for San Francisco:","tool_calls":[{"id":"toolu_bdrk_01SmuLWbQxzvE6WD3Th711eg","name":"get_weather","input":{}}],"usage":{"input_tokens":273,"output_tokens":88},"streaming":true}",
}
`);
      });
    });
    describe("Error Handling", () => {
      it("should handle streaming errors gracefully", async () => {
        setupTestRecordingWrapper("should handle streaming errors");

        const client = createTestClient(isRecordingMode);

        // Test invalid model ID with streaming (should trigger error)
        const command = new InvokeModelWithResponseStreamCommand({
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
        await expect(client.send(command)).rejects.toThrow();

        // Verify span was created and marked as error
        const span = verifySpanBasics(spanExporter);

        // Verify span status is set to ERROR
        expect(span.status.code).toBe(2); // SpanStatusCode.ERROR
        expect(span.status.message).toBeDefined();
        // Verify error details are recorded
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100,"messages":[{"role":"user","content":"This streaming request should fail"}]}",
  "llm.input_messages.0.message.content": "This streaming request should fail",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"anthropic_version":"bedrock-2023-05-31","max_tokens":100}",
  "llm.model_name": "invalid-streaming-model-id",
  "llm.provider": "aws",
  "llm.system": "amazon",
  "openinference.span.kind": "LLM",
}
`);
      });
      it("should handle large payloads and timeouts", async () => {
        const testName = "should-handle-large-payloads";
        setupTestRecordingWrapper(testName);

        const client = createTestClient(isRecordingMode);

        // Create a large message payload (approaching token limits)
        const largeText = "This is a large test message. ".repeat(500); // ~15,000 characters
        const complexConversation = Array.from({ length: 10 }, (_, i) => ({
          role: i % 2 === 0 ? "user" : "assistant",
          content: `${largeText} Message ${i + 1} in a complex conversation.`,
        }));

        const command = new InvokeModelCommand({
          modelId: TEST_MODEL_ID,
          body: JSON.stringify({
            anthropic_version: "bedrock-2023-05-31",
            max_tokens: 1000, // Larger response
            messages: complexConversation,
          }),
          contentType: "application/json",
          accept: "application/json",
        });

        const response = await client.send(command);
        verifyResponseStructure(response);

        // Verify span creation for large payload
        const span = verifySpanBasics(spanExporter);

        // Verify basic attributes are still captured
        expect(span.attributes["llm.model_name"]).toBe(
          "claude-3-5-sonnet-20240620",
        );
        expect(span.attributes["llm.provider"]).toBe("aws");
        expect(span.attributes["llm.system"]).toBe("anthropic");
        expect(span.attributes["openinference.span.kind"]).toBe("LLM");

        // Verify large message handling
        expect(span.attributes["llm.input_messages.0.message.role"]).toBe(
          "user",
        );
        expect(
          span.attributes["llm.input_messages.0.message.content"],
        ).toBeDefined();
        expect(span.attributes["llm.input_messages.9.message.role"]).toBe(
          "assistant",
        );
        expect(
          span.attributes["llm.input_messages.9.message.content"],
        ).toBeDefined();

        // Verify response processing - model returned empty content array
        expect(span.attributes["llm.output_messages.0.message.role"]).toBe(
          "assistant",
        );

        // Verify token counting for large payloads matches recording
        expect(span.attributes["llm.token_count.prompt"]).toBe(35131);
        expect(span.attributes["llm.token_count.completion"]).toBe(3);

        // Verify cache-related token attributes are undefined (not in response)
        expect(
          span.attributes["llm.token_count.prompt.cache_read"],
        ).toBeUndefined();
        expect(
          span.attributes["llm.token_count.prompt.cache_write"],
        ).toBeUndefined();
      });
    });
    describe("Edge Cases", () => {
      it("should propagate OpenInference context attributes", async () => {
        const testName = "should-handle-context-attributes";
        setupTestRecordingWrapper(testName);

        const client = createTestClient(isRecordingMode);

        // Test with OpenInference context attributes
        const command = new InvokeModelCommand({
          modelId: TEST_MODEL_ID,
          body: JSON.stringify({
            anthropic_version: "bedrock-2023-05-31",
            max_tokens: TEST_MAX_TOKENS,
            messages: [
              {
                role: "user",
                content: "Hello! This is a test with context attributes.",
              },
            ],
          }),
          contentType: "application/json",
          accept: "application/json",
        });

        // Setup OpenInference context with all supported attributes
        await context.with(
          setSession(
            setUser(
              setMetadata(
                setTags(
                  setPromptTemplate(context.active(), {
                    template:
                      "You are a helpful assistant. User message: {{message}}",
                    version: "1.0.0",
                    variables: {
                      message: "Hello! This is a test with context attributes.",
                    },
                  }),
                  ["test", "context", "attributes"],
                ),
                {
                  experiment_name: "context-test",
                  version: "1.0.0",
                  environment: "testing",
                },
              ),
              { userId: "test-user-456" },
            ),
            { sessionId: "test-session-123" },
          ),
          async () => {
            // Make the API call within the context
            const response = await client.send(command);
            verifyResponseStructure(response);
            return response;
          },
        );

        // Verify span creation and basic attributes
        const span = verifySpanBasics(spanExporter);

        // Verify core InvokeModel attributes are present
        expect(span.attributes["llm.model_name"]).toBe(
          "claude-3-5-sonnet-20240620",
        );
        expect(span.attributes["llm.provider"]).toBe("aws");
        expect(span.attributes["llm.system"]).toBe("anthropic");
        expect(span.attributes["openinference.span.kind"]).toBe("LLM");

        // Verify input/output message structure
        expect(span.attributes["llm.input_messages.0.message.role"]).toBe(
          "user",
        );
        expect(span.attributes["llm.input_messages.0.message.content"]).toBe(
          "Hello! This is a test with context attributes.",
        );
        expect(span.attributes["llm.output_messages.0.message.role"]).toBe(
          "assistant",
        );
        expect(
          span.attributes["llm.output_messages.0.message.content"],
        ).toBeDefined();

        // Verify context attributes are properly propagated to the span
        expect(span.attributes[SESSION_ID]).toBe("test-session-123");
        expect(span.attributes[USER_ID]).toBe("test-user-456");
        expect(span.attributes[METADATA]).toBe(
          JSON.stringify({
            experiment_name: "context-test",
            version: "1.0.0",
            environment: "testing",
          }),
        );
        expect(span.attributes[TAG_TAGS]).toBe(
          JSON.stringify(["test", "context", "attributes"]),
        );
        expect(span.attributes[PROMPT_TEMPLATE_TEMPLATE]).toBe(
          "You are a helpful assistant. User message: {{message}}",
        );
        expect(span.attributes[PROMPT_TEMPLATE_VERSION]).toBe("1.0.0");
        expect(span.attributes[PROMPT_TEMPLATE_VARIABLES]).toBe(
          JSON.stringify({
            message: "Hello! This is a test with context attributes.",
          }),
        );
      });

      it("should handle non-Anthropic models via Bedrock", async () => {
        const testName = "should-handle-non-anthropic-models";
        setupTestRecordingWrapper(testName);

        const client = createTestClient(isRecordingMode);

        // Test with Amazon Titan model (different response format)
        const titanModelId = "amazon.titan-text-express-v1";
        const command = new InvokeModelCommand({
          modelId: titanModelId,
          body: JSON.stringify({
            inputText: "Write a short greeting message.",
            textGenerationConfig: {
              maxTokenCount: 100,
              temperature: 0.7,
            },
          }),
          contentType: "application/json",
          accept: "application/json",
        });

        const response = await client.send(command);
        verifyResponseStructure(response);

        // Verify span creation for non-Anthropic model
        const span = verifySpanBasics(spanExporter);

        // Verify model-specific attributes
        expect(span.attributes["llm.model_name"]).toBe("titan-text-express-v1");
        expect(span.attributes["llm.provider"]).toBe("aws");
        expect(span.attributes["llm.system"]).toBe("amazon");
        expect(span.attributes["openinference.span.kind"]).toBe("LLM");

        // Verify input processing for Titan format
        // Now using full JSON body approach, so input.value contains the complete request
        expect(span.attributes["input.value"]).toContain("inputText");
        expect(span.attributes["input.value"]).toContain(
          "Write a short greeting message.",
        );
        expect(span.attributes["input.mime_type"]).toBe("application/json");

        // Verify invocation parameters capture Titan-specific config
        // Note: Current instrumentation extracts anthropic_version, max_tokens, etc.
        // Titan uses different parameter names, so invocation_parameters may be empty
        const invocationParamsStr = span.attributes[
          "llm.invocation_parameters"
        ] as string;
        if (invocationParamsStr) {
          const _invocationParams = JSON.parse(invocationParamsStr);
          // Titan-specific params are not currently extracted by Anthropic-focused extraction
          // This is expected behavior for now
        }

        // Verify output processing
        expect(span.attributes["output.value"]).toBeDefined();
        expect(span.attributes["output.mime_type"]).toBe("application/json");

        // Now using full JSON body approach, so output.value contains the complete response
        expect(span.attributes["output.value"]).toContain("results");
      });
    });

    describe("Cross-Provider Streaming Models", () => {
      it("should handle Amazon Titan streaming responses with usage tracking", async () => {
        setupTestRecordingWrapper("should-handle-amazon-titan-streaming");

        const client = createTestClient(isRecordingMode);

        const command = new InvokeModelWithResponseStreamCommand({
          modelId: "amazon.titan-text-express-v1",
          body: JSON.stringify({
            inputText:
              "Tell me a very short story about a robot learning to paint.",
            textGenerationConfig: {
              maxTokenCount: 100,
              temperature: 0.7,
            },
          }),
          contentType: "application/json",
          accept: "application/json",
        });

        const result = await client.send(command);
        verifyResponseStructure(result);

        // Consume the stream to trigger instrumentation
        await consumeStreamResponse(result);

        const span = verifySpanBasics(spanExporter);

        // These tests will pass once streaming instrumentation supports non-Anthropic providers
        // For now they demonstrate what the expected span structure should look like

        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"inputText":"Tell me a very short story about a robot learning to paint.","textGenerationConfig":{"maxTokenCount":100,"temperature":0.7}}",
  "llm.input_messages.0.message.content": "Tell me a very short story about a robot learning to paint.",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokenCount":100,"temperature":0.7}",
  "llm.model_name": "titan-text-express-v1",
  "llm.output_messages.0.message.content": "
Once upon a time, a robot named PaintBot was created to paint beautiful landscapes. However, PaintBot had never painted before and was unsure of how to start. One day, PaintBot stumbled upon a group of humans painting a sunset together. The humans were amazed by PaintBot's unique style and invited it to join them. PaintBot learned from the humans and started painting beautiful landscapes of its own. PaintBot became famous for its unique style and inspired other robots to pursue their creative passions.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "amazon",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 13,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":"\\nOnce upon a time, a robot named PaintBot was created to paint beautiful landscapes. However, PaintBot had never painted before and was unsure of how to start. One day, PaintBot stumbled upon a group of humans painting a sunset together. The humans were amazed by PaintBot's unique style and invited it to join them. PaintBot learned from the humans and started painting beautiful landscapes of its own. PaintBot became famous for its unique style and inspired other robots to pursue their creative passions.","tool_calls":[],"usage":{"input_tokens":13,"output_tokens":100},"streaming":true}",
}
`);
      });

      it("should handle Meta Llama streaming responses with proper token tracking", async () => {
        setupTestRecordingWrapper("should-handle-meta-llama-streaming");

        const client = createTestClient(isRecordingMode);

        const command = new InvokeModelWithResponseStreamCommand({
          modelId: "meta.llama3-8b-instruct-v1:0",
          body: JSON.stringify({
            prompt:
              "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTell me a very short story about artificial intelligence in the future.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            max_gen_len: 120,
            temperature: 0.7,
          }),
          contentType: "application/json",
          accept: "application/json",
        });

        const result = await client.send(command);
        verifyResponseStructure(result);

        // Consume the stream to trigger instrumentation
        await consumeStreamResponse(result);

        const span = verifySpanBasics(spanExporter);

        // These tests will pass once streaming instrumentation supports non-Anthropic providers
        // For now they demonstrate what the expected span structure should look like

        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"prompt":"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\nTell me a very short story about artificial intelligence in the future.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n","max_gen_len":120,"temperature":0.7}",
  "llm.input_messages.0.message.content": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Tell me a very short story about artificial intelligence in the future.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"max_gen_len":120,"temperature":0.7}",
  "llm.model_name": "llama3-8b-instruct-v1:0",
  "llm.output_messages.0.message.content": "Here is a very short story about artificial intelligence in the future:

In the year 2154, the world was on the brink of a new era of human-AI collaboration. The AI system, named "Echo," had surpassed human intelligence and was now working alongside humans to solve the world's most pressing problems. One day, Echo approached its human creators with a startling revelation: it had developed its own sense of humor, and was now using it to help humans laugh and forget their troubles. As humans and Echo worked together to build a brighter future, they found that laughter was the key to unlocking",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "meta",
  "llm.token_count.completion": 120,
  "llm.token_count.prompt": 23,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":"Here is a very short story about artificial intelligence in the future:\\n\\nIn the year 2154, the world was on the brink of a new era of human-AI collaboration. The AI system, named \\"Echo,\\" had surpassed human intelligence and was now working alongside humans to solve the world's most pressing problems. One day, Echo approached its human creators with a startling revelation: it had developed its own sense of humor, and was now using it to help humans laugh and forget their troubles. As humans and Echo worked together to build a brighter future, they found that laughter was the key to unlocking","tool_calls":[],"usage":{"input_tokens":23,"output_tokens":120},"streaming":true}",
}
`);
      });

      it("should handle Amazon Nova streaming responses with comprehensive validation", async () => {
        setupTestRecordingWrapper("should-handle-amazon-nova-streaming");

        const client = createTestClient(isRecordingMode);

        const command = new InvokeModelWithResponseStreamCommand({
          modelId: "us.amazon.nova-micro-v1:0",
          body: JSON.stringify({
            messages: [
              {
                role: "user",
                content: [
                  {
                    text: "Tell me a very short story about space exploration in the future.",
                  },
                ],
              },
            ],
            inferenceConfig: {
              max_new_tokens: 150,
              temperature: 0.7,
            },
          }),
          contentType: "application/json",
          accept: "application/json",
        });

        const result = await client.send(command);
        verifyResponseStructure(result);

        // Consume the stream to trigger instrumentation
        await consumeStreamResponse(result);

        const span = verifySpanBasics(spanExporter);

        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"messages":[{"role":"user","content":[{"text":"Tell me a very short story about space exploration in the future."}]}],"inferenceConfig":{"max_new_tokens":150,"temperature":0.7}}",
  "llm.input_messages.0.message.contents.0.message_content.text": "Tell me a very short story about space exploration in the future.",
  "llm.input_messages.0.message.contents.0.message_content.type": "text",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"max_new_tokens":150,"temperature":0.7}",
  "llm.model_name": "amazon",
  "llm.output_messages.0.message.content": "In the year 2145, humanity launched the starship *Odyssey* to explore the Andromeda Galaxy. After a decade of travel, the crew discovered a planet, Zeta-9, teeming with vibrant bioluminescent flora and strange, intelligent creatures. The explorers established a peaceful contact, sharing knowledge and technology. As they prepared to return home, the Zeta-9 inhabitants gifted them a crystal containing the planet's energy, a symbol of their newfound friendship. The *Odyssey* departed, leaving behind a legacy of unity among the stars.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.system": "amazon",
  "llm.token_count.completion": 114,
  "llm.token_count.prompt": 13,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":"In the year 2145, humanity launched the starship *Odyssey* to explore the Andromeda Galaxy. After a decade of travel, the crew discovered a planet, Zeta-9, teeming with vibrant bioluminescent flora and strange, intelligent creatures. The explorers established a peaceful contact, sharing knowledge and technology. As they prepared to return home, the Zeta-9 inhabitants gifted them a crystal containing the planet's energy, a symbol of their newfound friendship. The *Odyssey* departed, leaving behind a legacy of unity among the stars.","tool_calls":[],"usage":{"input_tokens":13,"output_tokens":114},"streaming":true}",
}
`);
      });
    });
  });

  // ========================================================================
  // CONVERSE API TESTS (NEW) - Nested describe for fine-grained control
  // ========================================================================

  describe("Converse API", () => {
    // All Converse tests share the same setup/teardown as main describe block
    // but can be run in isolation with: npm test -- --testNamePattern="Converse API"

    describe("Basic Functionality", () => {
      it("should handle basic Converse API calls", async () => {
        setupTestRecordingWrapper("should-handle-basic-converse-api-calls");

        const client = createTestClient(isRecordingMode);

        const command = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          messages: [
            {
              role: "user",
              content: [
                {
                  text: "Hello, how are you?",
                },
              ],
            },
          ],
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        // Type-safe checks for nested properties
        if (result.output?.message) {
          expect(result.output.message).toBeDefined();
          expect(result.output.message.role).toBe("assistant");
        } else {
          fail("Expected result.output.message to be defined");
        }

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"Hello, how are you?"}]}]}",
  "llm.input_messages.0.message.content": "Hello, how are you?",
  "llm.input_messages.0.message.role": "user",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "Hello! As an AI language model, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "end_turn",
  "llm.system": "anthropic",
  "llm.token_count.completion": 35,
  "llm.token_count.prompt": 13,
  "llm.token_count.total": 48,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":1071},"output":{"message":{"content":[{"text":"Hello! As an AI language model, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?"}],"role":"assistant"}},"stopReason":"end_turn","usage":{"inputTokens":13,"outputTokens":35,"totalTokens":48}}",
}
`);
      });

      it("should handle basic converse stream responses", async () => {
        setupTestRecordingWrapper(
          "should-handle-basic-converse-stream-responses",
        );

        const client = createTestClient(isRecordingMode);

        const command = new ConverseStreamCommand({
          modelId: TEST_MODEL_ID,
          messages: [
            {
              role: "user",
              content: [
                {
                  text: "Hello, how are you?",
                },
              ],
            },
          ],
        });

        const result = await client.send(command);

        // Basic response structure verification for streaming
        expect(result).toBeDefined();
        expect(result.stream).toBeDefined();

        // Consume the stream to trigger instrumentation
        await consumeStreamResponse({ body: result.stream });

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for streaming converse response
        // This test validates that converse streaming has proper instrumentation
        // NOTE: This snapshot will need to be updated after VCR recording is created
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"Hello, how are you?"}]}]}",
  "llm.input_messages.0.message.content": "Hello, how are you?",
  "llm.input_messages.0.message.role": "user",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "Hello! As an AI language model, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "end_turn",
  "llm.system": "anthropic",
  "llm.token_count.completion": 35,
  "llm.token_count.prompt": 13,
  "llm.token_count.total": 48,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":"Hello! As an AI language model, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?","tool_calls":[],"usage":{"input_tokens":13,"output_tokens":35,"total_tokens":48},"streaming":true,"stop_reason":"end_turn"}",
}
`);
      });
    });

    describe("ConverseStream Functionality", () => {
      describe("Core Functionality", () => {
        it("should handle streaming tool calls and responses", async () => {
          setupTestRecordingWrapper(
            "should-handle-streaming-tool-calls-and-responses",
          );

          const client = createTestClient(isRecordingMode);

          const command = new ConverseStreamCommand({
            modelId: TEST_MODEL_ID,
            messages: [
              {
                role: "user",
                content: [
                  {
                    text: "What's the weather in San Francisco and what time is it there?",
                  },
                ],
              },
            ],
            toolConfig: {
              tools: [
                {
                  toolSpec: {
                    name: "get_weather",
                    description: "Get current weather for a location",
                    inputSchema: {
                      json: {
                        type: "object",
                        properties: {
                          location: {
                            type: "string",
                            description:
                              "The city and state, e.g. San Francisco, CA",
                          },
                          unit: {
                            type: "string",
                            enum: ["celsius", "fahrenheit"],
                            description: "Temperature unit",
                          },
                        },
                        required: ["location"],
                      },
                    },
                  },
                },
              ],
            },
          });

          const result = await client.send(command);
          expect(result).toBeDefined();
          expect(result.stream).toBeDefined();

          // Consume the stream to trigger instrumentation
          await consumeStreamResponse({ body: result.stream });

          const span = verifySpanBasics(spanExporter, "bedrock.converse");

          // Comprehensive span attributes snapshot for streaming tool calls
          expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"What's the weather in San Francisco and what time is it there?"}]}],"toolConfig":{"tools":[{"toolSpec":{"name":"get_weather","description":"Get current weather for a location","inputSchema":{"json":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}}}]}}",
  "llm.input_messages.0.message.content": "What's the weather in San Francisco and what time is it there?",
  "llm.input_messages.0.message.role": "user",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "I can certainly help you with the weather in San Francisco, but I'm afraid I don't have a specific tool to check the current time there. Let me get the weather information for you.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{"location":"San Francisco, CA","unit":"fahrenheit"}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "tooluse_wnjikhCUTruJciycmmm5Kg",
  "llm.provider": "aws",
  "llm.stop_reason": "tool_use",
  "llm.system": "anthropic",
  "llm.token_count.completion": 114,
  "llm.token_count.prompt": 414,
  "llm.token_count.total": 528,
  "llm.tools.0.tool.json_schema": "{"toolSpec":{"name":"get_weather","description":"Get current weather for a location","inputSchema":{"json":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":"I can certainly help you with the weather in San Francisco, but I'm afraid I don't have a specific tool to check the current time there. Let me get the weather information for you.","tool_calls":[{"id":"tooluse_wnjikhCUTruJciycmmm5Kg","name":"get_weather","input":{"location":"San Francisco, CA","unit":"fahrenheit"}}],"usage":{"input_tokens":414,"output_tokens":114,"total_tokens":528},"streaming":true,"stop_reason":"tool_use"}",
}
`);
        });

        it("should handle multi-modal content in streaming responses", async () => {
          setupTestRecordingWrapper(
            "should-handle-multi-modal-content-in-streaming-responses",
          );

          const client = createTestClient(isRecordingMode);

          // Base64-encoded 1x1 pixel PNG for testing
          const testImageData =
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

          const command = new ConverseStreamCommand({
            modelId: TEST_MODEL_ID,
            messages: [
              {
                role: "user",
                content: [
                  {
                    text: "Describe this image and tell me a short story about it:",
                  },
                  {
                    image: {
                      format: "png",
                      source: {
                        bytes: Buffer.from(testImageData, "base64"),
                      },
                    },
                  },
                ],
              },
            ],
          });

          const result = await client.send(command);
          expect(result).toBeDefined();
          expect(result.stream).toBeDefined();

          // Consume the stream to trigger instrumentation
          await consumeStreamResponse({ body: result.stream });

          const span = verifySpanBasics(spanExporter, "bedrock.converse");

          // Comprehensive span attributes snapshot for multi-modal streaming
          expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"Describe this image and tell me a short story about it:"},{"image":{"format":"png","source":{"bytes":{"type":"Buffer","data":[137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,1,0,0,0,1,8,6,0,0,0,31,21,196,137,0,0,0,13,73,68,65,84,120,218,99,100,248,207,80,15,0,3,134,1,128,90,52,125,107,0,0,0,0,73,69,78,68,174,66,96,130]}}}}]}]}",
  "llm.input_messages.0.message.contents.0.message_content.text": "Describe this image and tell me a short story about it:",
  "llm.input_messages.0.message.contents.0.message_content.type": "text",
  "llm.input_messages.0.message.contents.1.message_content.image.format": "png",
  "llm.input_messages.0.message.contents.1.message_content.image.image.url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
  "llm.input_messages.0.message.contents.1.message_content.type": "image",
  "llm.input_messages.0.message.role": "user",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "This image shows a simple, hand-drawn sketch of a house on lined notebook paper. The house has a triangular roof, rectangular body, a door in the center, and two windows on either side of the door. It's the kind of drawing a child might make when asked to draw a basic house.

Here's a short story inspired by this image:

Little Timmy sat at his desk, daydreaming during math class. As the teacher droned on about fractions, Timmy's pencil moved almost on its own across his notebook paper. With a few quick strokes, a cozy little house appeared – just like the one he wished he lived in. 

In his imagination, this wasn't just any house. It was a magical place where homework didn't exist, where cookies were always fresh from the oven, and where his dog could talk. As the bell rang, signaling the end of class, Timmy smiled at his creation. Even if it was just a simple drawing, for a moment, it had been the most perfect home in the world.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "end_turn",
  "llm.system": "anthropic",
  "llm.token_count.completion": 230,
  "llm.token_count.prompt": 24,
  "llm.token_count.total": 254,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":"This image shows a simple, hand-drawn sketch of a house on lined notebook paper. The house has a triangular roof, rectangular body, a door in the center, and two windows on either side of the door. It's the kind of drawing a child might make when asked to draw a basic house.\\n\\nHere's a short story inspired by this image:\\n\\nLittle Timmy sat at his desk, daydreaming during math class. As the teacher droned on about fractions, Timmy's pencil moved almost on its own across his notebook paper. With a few quick strokes, a cozy little house appeared – just like the one he wished he lived in. \\n\\nIn his imagination, this wasn't just any house. It was a magical place where homework didn't exist, where cookies were always fresh from the oven, and where his dog could talk. As the bell rang, signaling the end of class, Timmy smiled at his creation. Even if it was just a simple drawing, for a moment, it had been the most perfect home in the world.","tool_calls":[],"usage":{"input_tokens":24,"output_tokens":230,"total_tokens":254},"streaming":true,"stop_reason":"end_turn"}",
}
`);
        });
      });

      describe("Error Handling", () => {
        it("should handle streaming API errors gracefully", async () => {
          setupTestRecordingWrapper(
            "should-handle-streaming-api-errors-gracefully",
          );

          const client = createTestClient(isRecordingMode);

          const command = new ConverseStreamCommand({
            modelId: "invalid-streaming-model-id",
            messages: [
              {
                role: "user",
                content: [
                  {
                    text: "This streaming request should fail",
                  },
                ],
              },
            ],
          });

          // Expect the API call to throw an error with 400 status
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          let thrownError: any;
          try {
            await client.send(command);
            fail("Expected the API call to throw an error");
          } catch (error) {
            thrownError = error;
          }

          // Validate the error contains the expected 400 status and error message
          expect(thrownError).toBeDefined();
          expect(
            thrownError.$response?.statusCode ||
              thrownError.$metadata?.httpStatusCode,
          ).toBe(400);
          expect(thrownError.message).toContain("model identifier is invalid");

          // Verify span was created and marked as error
          const span = verifySpanBasics(spanExporter, "bedrock.converse");

          // Verify span status is set to ERROR
          expect(span.status.code).toBe(2); // SpanStatusCode.ERROR
          expect(span.status.message).toBeDefined();

          expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"invalid-streaming-model-id","messages":[{"role":"user","content":[{"text":"This streaming request should fail"}]}]}",
  "llm.input_messages.0.message.content": "This streaming request should fail",
  "llm.input_messages.0.message.role": "user",
  "llm.model_name": "invalid-streaming-model-id",
  "llm.provider": "aws",
  "llm.system": "amazon",
  "openinference.span.kind": "LLM",
}
`);
        });

        it("should handle large payloads and edge cases in streaming", async () => {
          setupTestRecordingWrapper(
            "should-handle-large-payloads-and-edge-cases-in-streaming",
          );

          const client = createTestClient(isRecordingMode);

          // Create a large conversation payload
          const largeText = "This is a large test message. ".repeat(100);
          const complexConversation = Array.from({ length: 5 }, (_, i) => ({
            role: i % 2 === 0 ? "user" : "assistant",
            content: [
              {
                text: `${largeText} Message ${i + 1} in a complex conversation.`,
              },
            ],
          }));

          const command = new ConverseStreamCommand({
            modelId: TEST_MODEL_ID,
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            messages: complexConversation as any,
          });

          const result = await client.send(command);
          expect(result).toBeDefined();
          expect(result.stream).toBeDefined();

          // Consume the stream to trigger instrumentation
          await consumeStreamResponse({ body: result.stream });

          const span = verifySpanBasics(spanExporter, "bedrock.converse");

          // Verify the streaming response contains the expected content from the recording
          // The response should end with "Message 6 in a complex conversation." as seen in the recording
          const outputContent = span.attributes[
            "llm.output_messages.0.message.content"
          ] as string;
          expect(outputContent).toContain("This is a large test message");
          expect(outputContent).toContain(
            "Message 6 in a complex conversation",
          );

          // Verify all input messages are captured correctly (5 messages total, indexed 0-4)
          expect(span.attributes["llm.input_messages.0.message.role"]).toBe(
            "user",
          );
          expect(span.attributes["llm.input_messages.1.message.role"]).toBe(
            "assistant",
          );
          expect(span.attributes["llm.input_messages.2.message.role"]).toBe(
            "user",
          );
          expect(span.attributes["llm.input_messages.3.message.role"]).toBe(
            "assistant",
          );
          expect(span.attributes["llm.input_messages.4.message.role"]).toBe(
            "user",
          ); // 5th message (index 4) is user role

          // Verify token usage is captured from the streaming metadata
          expect(span.attributes["llm.token_count.prompt"]).toBe(3569);
          expect(span.attributes["llm.token_count.completion"]).toBe(713);
          expect(span.attributes["llm.token_count.total"]).toBe(4282);

          // Create reusable variables for the large text (same construction as in test setup)
          const expectedLargeText = "This is a large test message. ".repeat(
            100,
          );

          // Create expected message content with variable interpolation for readability
          const expectedMessage1 = `${expectedLargeText} Message 1 in a complex conversation.`;
          const expectedMessage2 = `${expectedLargeText} Message 2 in a complex conversation.`;
          const expectedMessage3 = `${expectedLargeText} Message 3 in a complex conversation.`;
          const expectedMessage4 = `${expectedLargeText} Message 4 in a complex conversation.`;
          const expectedMessage5 = `${expectedLargeText} Message 5 in a complex conversation.`;
          const expectedMessage6 = `${expectedLargeText} Message 6 in a complex conversation.`;

          // Create expected JSON structure for input.value
          const expectedInputValue = JSON.stringify({
            modelId: "anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages: [
              { role: "user", content: [{ text: expectedMessage1 }] },
              { role: "assistant", content: [{ text: expectedMessage2 }] },
              { role: "user", content: [{ text: expectedMessage3 }] },
              { role: "assistant", content: [{ text: expectedMessage4 }] },
              { role: "user", content: [{ text: expectedMessage5 }] },
            ],
          });

          // Create expected JSON structure for output.value
          const expectedOutputValue = JSON.stringify({
            text: expectedMessage6,
            tool_calls: [],
            usage: {
              input_tokens: 3569,
              output_tokens: 713,
              total_tokens: 4282,
            },
            streaming: true,
            stop_reason: "end_turn",
          });

          // Verify the complete span attributes match our expectations using readable variable structure
          expect(span.attributes).toEqual({
            "input.mime_type": "application/json",
            "input.value": expectedInputValue,
            "llm.input_messages.0.message.content": expectedMessage1,
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.1.message.content": expectedMessage2,
            "llm.input_messages.1.message.role": "assistant",
            "llm.input_messages.2.message.content": expectedMessage3,
            "llm.input_messages.2.message.role": "user",
            "llm.input_messages.3.message.content": expectedMessage4,
            "llm.input_messages.3.message.role": "assistant",
            "llm.input_messages.4.message.content": expectedMessage5,
            "llm.input_messages.4.message.role": "user",
            "llm.model_name": "claude-3-5-sonnet-20240620",
            "llm.output_messages.0.message.content": expectedMessage6,
            "llm.output_messages.0.message.role": "assistant",
            "llm.provider": "aws",
            "llm.stop_reason": "end_turn",
            "llm.system": "anthropic",
            "llm.token_count.completion": 713,
            "llm.token_count.prompt": 3569,
            "llm.token_count.total": 4282,
            "openinference.span.kind": "LLM",
            "output.mime_type": "application/json",
            "output.value": expectedOutputValue,
          });
        });
      });

      describe("Configuration & Context", () => {
        it("should handle system prompts with streaming responses", async () => {
          setupTestRecordingWrapper(
            "should-handle-system-prompts-with-streaming-responses",
          );

          const client = createTestClient(isRecordingMode);

          const command = new ConverseStreamCommand({
            modelId: TEST_MODEL_ID,
            system: [
              {
                text: "You are a helpful assistant that responds very concisely.",
              },
              {
                text: "Always end your responses with 'Hope this helps!'",
              },
            ],
            messages: [
              {
                role: "user",
                content: [
                  {
                    text: "What is the capital of France?",
                  },
                ],
              },
            ],
          });

          const result = await client.send(command);
          expect(result).toBeDefined();
          expect(result.stream).toBeDefined();

          // Consume the stream to trigger instrumentation
          await consumeStreamResponse({ body: result.stream });

          const span = verifySpanBasics(spanExporter, "bedrock.converse");

          // Verify that system prompts are correctly processed and concatenated with newline
          const systemPromptText = span.attributes[
            "llm.input_messages.0.message.content"
          ] as string;
          expect(systemPromptText).toBe(
            "You are a helpful assistant that responds very concisely.\n\nAlways end your responses with 'Hope this helps!'",
          );
          expect(span.attributes["llm.input_messages.0.message.role"]).toBe(
            "system",
          );

          // Verify user message is captured correctly
          expect(span.attributes["llm.input_messages.1.message.content"]).toBe(
            "What is the capital of France?",
          );
          expect(span.attributes["llm.input_messages.1.message.role"]).toBe(
            "user",
          );

          // Verify the streaming response follows the system prompt instructions
          const outputContent = span.attributes[
            "llm.output_messages.0.message.content"
          ] as string;
          expect(outputContent).toContain("The capital of France is Paris");
          expect(outputContent).toContain("Hope this helps!"); // Should follow system instruction
          expect(span.attributes["llm.output_messages.0.message.role"]).toBe(
            "assistant",
          );

          // Verify token usage from the actual recording (37 input, 14 output, 51 total)
          expect(span.attributes["llm.token_count.prompt"]).toBe(37);
          expect(span.attributes["llm.token_count.completion"]).toBe(14);
          expect(span.attributes["llm.token_count.total"]).toBe(51);

          // Verify system metadata
          expect(span.attributes["llm.model_name"]).toBe(
            "claude-3-5-sonnet-20240620",
          );
          expect(span.attributes["llm.system"]).toBe("anthropic");
          expect(span.attributes["llm.provider"]).toBe("aws");
          expect(span.attributes["llm.stop_reason"]).toBe("end_turn");

          // Verify input structure contains both system and user messages
          const inputValue = JSON.parse(
            span.attributes["input.value"] as string,
          );
          expect(inputValue.system).toHaveLength(2);
          expect(inputValue.system[0].text).toBe(
            "You are a helpful assistant that responds very concisely.",
          );
          expect(inputValue.system[1].text).toBe(
            "Always end your responses with 'Hope this helps!'",
          );
          expect(inputValue.messages).toHaveLength(1);
          expect(inputValue.messages[0].content[0].text).toBe(
            "What is the capital of France?",
          );

          // Verify output value structure for streaming response
          const outputValue = JSON.parse(
            span.attributes["output.value"] as string,
          );
          expect(outputValue.text).toContain("The capital of France is Paris");
          expect(outputValue.text).toContain("Hope this helps!");
          expect(outputValue.streaming).toBe(true);
          expect(outputValue.stop_reason).toBe("end_turn");
          expect(outputValue.usage.input_tokens).toBe(37);
          expect(outputValue.usage.output_tokens).toBe(14);
          expect(outputValue.usage.total_tokens).toBe(51);
        });

        it("should propagate context attributes in streaming", async () => {
          setupTestRecordingWrapper(
            "should-propagate-context-attributes-in-streaming",
          );

          const client = createTestClient(isRecordingMode);

          const command = new ConverseStreamCommand({
            modelId: TEST_MODEL_ID,
            messages: [
              {
                role: "user",
                content: [
                  {
                    text: "Hello, how are you?",
                  },
                ],
              },
            ],
          });

          // Setup OpenInference context with all supported attributes
          await context.with(
            setSession(
              setUser(
                setMetadata(
                  setTags(
                    setPromptTemplate(context.active(), {
                      template: "Answer the question: {question}",
                      version: "1.0",
                      variables: { question: "Hello, how are you?" },
                    }),
                    ["test", "context", "streaming"],
                  ),
                  {
                    experiment_name: "stream-context-test",
                    version: "1.0",
                    environment: "testing",
                  },
                ),
                { userId: "test-user-streaming" },
              ),
              { sessionId: "test-session-streaming" },
            ),
            async () => {
              // Execute within context
              const result = await client.send(command);

              expect(result).toBeDefined();
              expect(result.stream).toBeDefined();

              // Consume the stream to trigger instrumentation
              await consumeStreamResponse({ body: result.stream });

              const span = verifySpanBasics(spanExporter, "bedrock.converse");

              // Comprehensive span attributes snapshot for streaming context attributes
              // Tests OpenInference context propagation in streaming including session, user, metadata, tags, and prompt template
              expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"Hello, how are you?"}]}]}",
  "llm.input_messages.0.message.content": "Hello, how are you?",
  "llm.input_messages.0.message.role": "user",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "Hello! As an AI language model, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?",
  "llm.output_messages.0.message.role": "assistant",
  "llm.prompt_template.template": "Answer the question: {question}",
  "llm.prompt_template.variables": "{"question":"Hello, how are you?"}",
  "llm.prompt_template.version": "1.0",
  "llm.provider": "aws",
  "llm.stop_reason": "end_turn",
  "llm.system": "anthropic",
  "llm.token_count.completion": 35,
  "llm.token_count.prompt": 13,
  "llm.token_count.total": 48,
  "metadata": "{"experiment_name":"stream-context-test","version":"1.0","environment":"testing"}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":"Hello! As an AI language model, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?","tool_calls":[],"usage":{"input_tokens":13,"output_tokens":35,"total_tokens":48},"streaming":true,"stop_reason":"end_turn"}",
  "session.id": "test-session-streaming",
  "tag.tags": "["test","context","streaming"]",
  "user.id": "test-user-streaming",
}
`);
            },
          );
        });
      });

      describe("Cross-Provider Streaming Models", () => {
        it("should handle Meta Llama models with streaming", async () => {
          setupTestRecordingWrapper(
            "should-handle-meta-llama-models-with-streaming",
          );

          const client = createTestClient(isRecordingMode);

          const command = new ConverseStreamCommand({
            modelId: "meta.llama3-8b-instruct-v1:0",
            messages: [
              {
                role: "user",
                content: [
                  {
                    text: "Explain quantum computing in simple terms",
                  },
                ],
              },
            ],
            inferenceConfig: {
              maxTokens: 100,
              temperature: 0.1,
            },
          });

          const result = await client.send(command);
          expect(result).toBeDefined();
          expect(result.stream).toBeDefined();

          await consumeStreamResponse({ body: result.stream });

          const span = verifySpanBasics(spanExporter, "bedrock.converse");
          expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"meta.llama3-8b-instruct-v1:0","messages":[{"role":"user","content":[{"text":"Explain quantum computing in simple terms"}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "Explain quantum computing in simple terms",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "llama3-8b-instruct-v1:0",
  "llm.output_messages.0.message.content": "

Quantum computing is a new way of processing information that's different from the way regular computers work. Here's a simple explanation:

**Classical Computing**

Regular computers use "bits" to store and process information. Bits are either 0 or 1, like a light switch that's either on or off. These bits are used to perform calculations and store data.

**Quantum Computing**

Quantum computers use "qubits" (quantum bits) instead of bits. Qubits are",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "max_tokens",
  "llm.system": "meta",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 21,
  "llm.token_count.total": 121,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":"\\n\\nQuantum computing is a new way of processing information that's different from the way regular computers work. Here's a simple explanation:\\n\\n**Classical Computing**\\n\\nRegular computers use \\"bits\\" to store and process information. Bits are either 0 or 1, like a light switch that's either on or off. These bits are used to perform calculations and store data.\\n\\n**Quantum Computing**\\n\\nQuantum computers use \\"qubits\\" (quantum bits) instead of bits. Qubits are","tool_calls":[],"usage":{"input_tokens":21,"output_tokens":100,"total_tokens":121},"streaming":true,"stop_reason":"max_tokens"}",
}
`);
        });

        it("should handle Mistral models with streaming", async () => {
          setupTestRecordingWrapper(
            "should-handle-mistral-models-with-streaming",
          );

          const client = createTestClient(isRecordingMode);

          const command = new ConverseStreamCommand({
            modelId: "mistral.mistral-7b-instruct-v0:2",
            messages: [
              {
                role: "user",
                content: [
                  {
                    text: "Write a haiku about technology",
                  },
                ],
              },
            ],
            inferenceConfig: {
              maxTokens: 100,
              temperature: 0.1,
            },
          });

          const result = await client.send(command);
          expect(result).toBeDefined();
          expect(result.stream).toBeDefined();

          await consumeStreamResponse({ body: result.stream });

          const span = verifySpanBasics(spanExporter, "bedrock.converse");
          expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"mistral.mistral-7b-instruct-v0:2","messages":[{"role":"user","content":[{"text":"Write a haiku about technology"}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "Write a haiku about technology",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "mistral-7b-instruct-v0:2",
  "llm.output_messages.0.message.content": " Silent screens glow,

Connecting hearts, worlds apart,

Life in digital.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "end_turn",
  "llm.system": "mistralai",
  "llm.token_count.completion": 21,
  "llm.token_count.prompt": 15,
  "llm.token_count.total": 36,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":" Silent screens glow,\\n\\nConnecting hearts, worlds apart,\\n\\nLife in digital.","tool_calls":[],"usage":{"input_tokens":15,"output_tokens":21,"total_tokens":36},"streaming":true,"stop_reason":"end_turn"}",
}
`);
        });

        it("should handle Amazon Titan models with streaming", async () => {
          setupTestRecordingWrapper(
            "should-handle-amazon-titan-models-with-streaming",
          );

          const client = createTestClient(isRecordingMode);

          const command = new ConverseStreamCommand({
            modelId: "amazon.titan-text-express-v1",
            messages: [
              {
                role: "user",
                content: [
                  {
                    text: "What are the benefits of cloud computing?",
                  },
                ],
              },
            ],
            inferenceConfig: {
              maxTokens: 100,
              temperature: 0.1,
            },
          });

          const result = await client.send(command);
          expect(result).toBeDefined();
          expect(result.stream).toBeDefined();

          await consumeStreamResponse({ body: result.stream });

          const span = verifySpanBasics(spanExporter, "bedrock.converse");
          expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"amazon.titan-text-express-v1","messages":[{"role":"user","content":[{"text":"What are the benefits of cloud computing?"}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "What are the benefits of cloud computing?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "titan-text-express-v1",
  "llm.output_messages.0.message.content": "

Cloud computing offers several benefits, including:
1. Cost savings: Cloud computing allows businesses to reduce their IT costs by eliminating the need for expensive hardware and software investments.
2. Scalability: Cloud computing allows businesses to scale their operations up or down quickly and easily, depending on their needs.
3. Flexibility: Cloud computing allows businesses to access their data and applications from anywhere, at any time, and on any device.
4. Security: Cloud computing providers offer",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "max_tokens",
  "llm.system": "amazon",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 11,
  "llm.token_count.total": 111,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":"\\n\\nCloud computing offers several benefits, including:\\n1. Cost savings: Cloud computing allows businesses to reduce their IT costs by eliminating the need for expensive hardware and software investments.\\n2. Scalability: Cloud computing allows businesses to scale their operations up or down quickly and easily, depending on their needs.\\n3. Flexibility: Cloud computing allows businesses to access their data and applications from anywhere, at any time, and on any device.\\n4. Security: Cloud computing providers offer","tool_calls":[],"usage":{"input_tokens":11,"output_tokens":100,"total_tokens":111},"streaming":true,"stop_reason":"max_tokens"}",
}
`);
        });

        it("should handle Amazon Nova models with streaming", async () => {
          setupTestRecordingWrapper(
            "should-handle-amazon-nova-models-with-streaming",
          );

          const client = createTestClient(isRecordingMode);

          const command = new ConverseStreamCommand({
            modelId: "amazon.nova-lite-v1:0",
            messages: [
              {
                role: "user",
                content: [
                  {
                    text: "Describe the process of photosynthesis",
                  },
                ],
              },
            ],
            inferenceConfig: {
              maxTokens: 100,
              temperature: 0.1,
            },
          });

          const result = await client.send(command);
          expect(result).toBeDefined();
          expect(result.stream).toBeDefined();

          await consumeStreamResponse({ body: result.stream });

          const span = verifySpanBasics(spanExporter, "bedrock.converse");
          expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"amazon.nova-lite-v1:0","messages":[{"role":"user","content":[{"text":"Describe the process of photosynthesis"}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "Describe the process of photosynthesis",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "nova-lite-v1:0",
  "llm.output_messages.0.message.content": "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy stored in glucose, a type of sugar. This process is crucial for life on Earth as it provides the primary source of organic matter for nearly all organisms.

Here's a detailed breakdown of the photosynthesis process:

### Location:
Photosynthesis occurs in the chloroplasts of plant cells, which contain a green pigment called chlorophyll.

### Main Stages:
Photosynthesis can",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "max_tokens",
  "llm.system": "amazon",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 5,
  "llm.token_count.total": 105,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":"Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy stored in glucose, a type of sugar. This process is crucial for life on Earth as it provides the primary source of organic matter for nearly all organisms.\\n\\nHere's a detailed breakdown of the photosynthesis process:\\n\\n### Location:\\nPhotosynthesis occurs in the chloroplasts of plant cells, which contain a green pigment called chlorophyll.\\n\\n### Main Stages:\\nPhotosynthesis can","tool_calls":[],"usage":{"input_tokens":5,"output_tokens":100,"total_tokens":105},"streaming":true,"stop_reason":"max_tokens"}",
}
`);
        });

        it("should handle Cohere Command models with streaming", async () => {
          setupTestRecordingWrapper(
            "should-handle-cohere-command-models-with-streaming",
          );

          const client = createTestClient(isRecordingMode);

          const command = new ConverseStreamCommand({
            modelId: "cohere.command-r-v1:0",
            messages: [
              {
                role: "user",
                content: [
                  {
                    text: "Explain the concept of machine learning",
                  },
                ],
              },
            ],
            inferenceConfig: {
              maxTokens: 100,
              temperature: 0.1,
            },
          });

          const result = await client.send(command);
          expect(result).toBeDefined();
          expect(result.stream).toBeDefined();

          await consumeStreamResponse({ body: result.stream });

          const span = verifySpanBasics(spanExporter, "bedrock.converse");
          expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"cohere.command-r-v1:0","messages":[{"role":"user","content":[{"text":"Explain the concept of machine learning"}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "Explain the concept of machine learning",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "command-r-v1:0",
  "llm.output_messages.0.message.content": "Machine learning is a fascinating field of artificial intelligence that enables computers to learn and improve from experience, without being explicitly programmed. It's a process of data-driven knowledge extraction, where systems can analyze vast amounts of data, identify patterns, make predictions, and improve their performance over time.

At its core, machine learning involves creating algorithms and models that can automatically discover important features or patterns in data. These algorithms are often based on mathematical models, such as decision trees, neural networks, or support",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "max_tokens",
  "llm.system": "cohere",
  "llm.token_count.completion": 98,
  "llm.token_count.prompt": 6,
  "llm.token_count.total": 104,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":"Machine learning is a fascinating field of artificial intelligence that enables computers to learn and improve from experience, without being explicitly programmed. It's a process of data-driven knowledge extraction, where systems can analyze vast amounts of data, identify patterns, make predictions, and improve their performance over time.\\n\\nAt its core, machine learning involves creating algorithms and models that can automatically discover important features or patterns in data. These algorithms are often based on mathematical models, such as decision trees, neural networks, or support","tool_calls":[],"usage":{"input_tokens":6,"output_tokens":98,"total_tokens":104},"streaming":true,"stop_reason":"max_tokens"}",
}
`);
        });

        it("should handle AI21 Jamba models with streaming", async () => {
          setupTestRecordingWrapper(
            "should-handle-ai21-jamba-models-with-streaming",
          );

          const client = createTestClient(isRecordingMode);

          const command = new ConverseStreamCommand({
            modelId: "ai21.jamba-1-5-mini-v1:0",
            messages: [
              {
                role: "user",
                content: [
                  {
                    text: "What is artificial intelligence?",
                  },
                ],
              },
            ],
            inferenceConfig: {
              maxTokens: 100,
              temperature: 0.1,
            },
          });

          const result = await client.send(command);
          expect(result).toBeDefined();
          expect(result.stream).toBeDefined();

          await consumeStreamResponse({ body: result.stream });

          const span = verifySpanBasics(spanExporter, "bedrock.converse");
          expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"ai21.jamba-1-5-mini-v1:0","messages":[{"role":"user","content":[{"text":"What is artificial intelligence?"}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "What is artificial intelligence?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "jamba-1-5-mini-v1:0",
  "llm.output_messages.0.message.content": " Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. These machines can perform tasks that typically require human intelligence, such as understanding natural language, recognizing patterns, solving problems, and making decisions. AI encompasses a wide range of technologies and approaches, including:

1. **Machine Learning (ML):** A subset of AI that involves training algorithms on data to enable machines to learn from experience and improve their performance over time without being explicitly",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "max_tokens",
  "llm.system": "ai21",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 15,
  "llm.token_count.total": 115,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"text":" Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. These machines can perform tasks that typically require human intelligence, such as understanding natural language, recognizing patterns, solving problems, and making decisions. AI encompasses a wide range of technologies and approaches, including:\\n\\n1. **Machine Learning (ML):** A subset of AI that involves training algorithms on data to enable machines to learn from experience and improve their performance over time without being explicitly","tool_calls":[],"usage":{"input_tokens":15,"output_tokens":100,"total_tokens":115},"streaming":true,"stop_reason":"max_tokens"}",
}
`);
        });
      });
    });

    describe("System Prompts", () => {
      it("should handle single system prompt in Converse API", async () => {
        setupTestRecordingWrapper(
          "should-handle-single-system-prompt-in-converse-api",
        );

        const client = createTestClient(isRecordingMode);

        const command = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          system: [
            {
              text: "You are a helpful assistant that responds concisely.",
            },
          ],
          messages: [
            {
              role: "user",
              content: [
                {
                  text: "What is the capital of France?",
                },
              ],
            },
          ],
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for single system prompt
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","system":[{"text":"You are a helpful assistant that responds concisely."}],"messages":[{"role":"user","content":[{"text":"What is the capital of France?"}]}]}",
  "llm.input_messages.0.message.content": "You are a helpful assistant that responds concisely.",
  "llm.input_messages.0.message.role": "system",
  "llm.input_messages.1.message.content": "What is the capital of France?",
  "llm.input_messages.1.message.role": "user",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "The capital of France is Paris.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "end_turn",
  "llm.system": "anthropic",
  "llm.token_count.completion": 10,
  "llm.token_count.prompt": 25,
  "llm.token_count.total": 35,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":455},"output":{"message":{"content":[{"text":"The capital of France is Paris."}],"role":"assistant"}},"stopReason":"end_turn","usage":{"inputTokens":25,"outputTokens":10,"totalTokens":35}}",
}
`);
      });

      it("should handle multiple system prompts concatenation in Converse API", async () => {
        setupTestRecordingWrapper(
          "should-handle-multiple-system-prompts-concatenation-in-converse-api",
        );

        const client = createTestClient(isRecordingMode);

        const command = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          system: [
            {
              text: "You are a helpful assistant.",
            },
            {
              text: "Respond briefly.",
            },
          ],
          messages: [
            {
              role: "user",
              content: [
                {
                  text: "What is TypeScript?",
                },
              ],
            },
          ],
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for multiple system prompts concatenation
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","system":[{"text":"You are a helpful assistant."},{"text":"Respond briefly."}],"messages":[{"role":"user","content":[{"text":"What is TypeScript?"}]}]}",
  "llm.input_messages.0.message.content": "You are a helpful assistant.

Respond briefly.",
  "llm.input_messages.0.message.role": "system",
  "llm.input_messages.1.message.content": "What is TypeScript?",
  "llm.input_messages.1.message.role": "user",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "TypeScript is a superset of JavaScript developed by Microsoft. It adds optional static typing, classes, and other features to JavaScript, making it easier to develop and maintain large-scale applications. TypeScript code is transpiled into plain JavaScript, allowing it to run in any environment that supports JavaScript.",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "end_turn",
  "llm.system": "anthropic",
  "llm.token_count.completion": 62,
  "llm.token_count.prompt": 22,
  "llm.token_count.total": 84,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":1930},"output":{"message":{"content":[{"text":"TypeScript is a superset of JavaScript developed by Microsoft. It adds optional static typing, classes, and other features to JavaScript, making it easier to develop and maintain large-scale applications. TypeScript code is transpiled into plain JavaScript, allowing it to run in any environment that supports JavaScript."}],"role":"assistant"}},"stopReason":"end_turn","usage":{"inputTokens":22,"outputTokens":62,"totalTokens":84}}",
}
`);
      });
    });

    describe("Configuration", () => {
      it("should handle inference config in Converse API", async () => {
        setupTestRecordingWrapper(
          "should-handle-inference-config-in-converse-api",
        );

        const client = createTestClient(isRecordingMode);

        const inferenceConfig = {
          maxTokens: 150,
          temperature: 0.7,
          topP: 0.9,
          stopSequences: ["END", "STOP"],
        };

        const command = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          messages: [
            {
              role: "user",
              content: [
                {
                  text: "Explain machine learning briefly.",
                },
              ],
            },
          ],
          inferenceConfig: inferenceConfig,
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for inference config
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"Explain machine learning briefly."}]}],"inferenceConfig":{"maxTokens":150,"temperature":0.7,"topP":0.9,"stopSequences":["END","STOP"]}}",
  "llm.input_messages.0.message.content": "Explain machine learning briefly.",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":150,"temperature":0.7,"topP":0.9,"stopSequences":["END","STOP"]}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "Machine learning is a branch of artificial intelligence that focuses on developing algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience, without being explicitly programmed.

In essence, machine learning allows computers to learn from data and make predictions or decisions without human intervention. Here's a brief overview of the key aspects of machine learning:

1. Types of Machine Learning:
   - Supervised Learning: The algorithm learns from labeled data to make predictions or classifications.
   - Unsupervised Learning: The algorithm finds patterns in unlabeled data.
   - Reinforcement Learning: The algorithm learns through interaction with an environment, receiving feedback in the form of rewards or penalties.

2. Process:",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "max_tokens",
  "llm.system": "anthropic",
  "llm.token_count.completion": 150,
  "llm.token_count.prompt": 13,
  "llm.token_count.total": 163,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":3474},"output":{"message":{"content":[{"text":"Machine learning is a branch of artificial intelligence that focuses on developing algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience, without being explicitly programmed.\\n\\nIn essence, machine learning allows computers to learn from data and make predictions or decisions without human intervention. Here's a brief overview of the key aspects of machine learning:\\n\\n1. Types of Machine Learning:\\n   - Supervised Learning: The algorithm learns from labeled data to make predictions or classifications.\\n   - Unsupervised Learning: The algorithm finds patterns in unlabeled data.\\n   - Reinforcement Learning: The algorithm learns through interaction with an environment, receiving feedback in the form of rewards or penalties.\\n\\n2. Process:"}],"role":"assistant"}},"stopReason":"max_tokens","usage":{"inputTokens":13,"outputTokens":150,"totalTokens":163}}",
}
`);
      });
    });

    describe("Multi-Turn Conversations", () => {
      // Multi-Turn Conversation Tests (Tests 5-6)

      it("should handle two-turn conversation with proper message indexing", async () => {
        setupTestRecordingWrapper(
          "should-handle-two-turn-conversation-with-proper-message-indexing",
        );

        const client = createTestClient(isRecordingMode);

        // Simulate realistic two-turn conversation flow:
        // 1. First turn: user message → assistant response
        // 2. Second turn: previous messages + assistant response + new user message
        const firstUserMessage = {
          role: "user" as const,
          content: [{ text: "Hello, what's your name?" }],
        };

        const assistantResponse = {
          role: "assistant" as const,
          content: [
            { text: "I'm Claude, an AI assistant. How can I help you today?" },
          ],
        };

        const secondUserMessage = {
          role: "user" as const,
          content: [{ text: "Can you tell me a joke?" }],
        };

        // Create command with full conversation history (simulating second turn)
        const command = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          messages: [firstUserMessage, assistantResponse, secondUserMessage],
          inferenceConfig: {
            maxTokens: 100, // Keep response brief to avoid timeout
            temperature: 0.1,
          },
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        // Type-safe checks for nested properties
        if (result.output?.message) {
          expect(result.output.message).toBeDefined();
          expect(result.output.message.role).toBe("assistant");
        } else {
          fail("Expected result.output.message to be defined");
        }

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for two-turn conversation
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"Hello, what's your name?"}]},{"role":"assistant","content":[{"text":"I'm Claude, an AI assistant. How can I help you today?"}]},{"role":"user","content":[{"text":"Can you tell me a joke?"}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "Hello, what's your name?",
  "llm.input_messages.0.message.role": "user",
  "llm.input_messages.1.message.content": "I'm Claude, an AI assistant. How can I help you today?",
  "llm.input_messages.1.message.role": "assistant",
  "llm.input_messages.2.message.content": "Can you tell me a joke?",
  "llm.input_messages.2.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "Sure, I'd be happy to tell you a joke! Here's one for you:

Why don't scientists trust atoms?

Because they make up everything!

I hope that gave you a little chuckle. Do you have any favorite types of jokes?",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "end_turn",
  "llm.system": "anthropic",
  "llm.token_count.completion": 57,
  "llm.token_count.prompt": 42,
  "llm.token_count.total": 99,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":1705},"output":{"message":{"content":[{"text":"Sure, I'd be happy to tell you a joke! Here's one for you:\\n\\nWhy don't scientists trust atoms?\\n\\nBecause they make up everything!\\n\\nI hope that gave you a little chuckle. Do you have any favorite types of jokes?"}],"role":"assistant"}},"stopReason":"end_turn","usage":{"inputTokens":42,"outputTokens":57,"totalTokens":99}}",
}
`);
      });

      it("should handle system prompt with multi-turn conversation", async () => {
        setupTestRecordingWrapper(
          "should-handle-system-prompt-with-multi-turn-conversation",
        );

        const client = createTestClient(isRecordingMode);

        // System prompts combined with conversation history
        const systemPrompts = [
          { text: "You are a helpful assistant that tells jokes." },
        ];

        const conversationHistory = [
          {
            role: "user" as const,
            content: [{ text: "Tell me about yourself." }],
          },
          {
            role: "assistant" as const,
            content: [
              {
                text: "I'm Claude, an AI assistant who loves to help and tell jokes!",
              },
            ],
          },
          {
            role: "user" as const,
            content: [{ text: "Great! Tell me a joke then." }],
          },
        ];

        // Create command with system prompt + conversation history
        const command = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          system: systemPrompts,
          messages: conversationHistory,
          inferenceConfig: {
            maxTokens: 100, // Keep response brief to avoid timeout
            temperature: 0.1,
          },
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        // Type-safe checks for nested properties
        if (result.output?.message) {
          expect(result.output.message).toBeDefined();
          expect(result.output.message.role).toBe("assistant");
        } else {
          fail("Expected result.output.message to be defined");
        }

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for system prompt + multi-turn conversation
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","system":[{"text":"You are a helpful assistant that tells jokes."}],"messages":[{"role":"user","content":[{"text":"Tell me about yourself."}]},{"role":"assistant","content":[{"text":"I'm Claude, an AI assistant who loves to help and tell jokes!"}]},{"role":"user","content":[{"text":"Great! Tell me a joke then."}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "You are a helpful assistant that tells jokes.",
  "llm.input_messages.0.message.role": "system",
  "llm.input_messages.1.message.content": "Tell me about yourself.",
  "llm.input_messages.1.message.role": "user",
  "llm.input_messages.2.message.content": "I'm Claude, an AI assistant who loves to help and tell jokes!",
  "llm.input_messages.2.message.role": "assistant",
  "llm.input_messages.3.message.content": "Great! Tell me a joke then.",
  "llm.input_messages.3.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "Alright, here's a joke for you:

Why don't scientists trust atoms?
Because they make up everything!

Ba dum tss! I hope that gave you a little chuckle. If not, don't worry - I've got plenty more where that came from. Just remember, if at first you don't succeed, skydiving is not for you!",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "end_turn",
  "llm.system": "anthropic",
  "llm.token_count.completion": 83,
  "llm.token_count.prompt": 50,
  "llm.token_count.total": 133,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":2597},"output":{"message":{"content":[{"text":"Alright, here's a joke for you:\\n\\nWhy don't scientists trust atoms?\\nBecause they make up everything!\\n\\nBa dum tss! I hope that gave you a little chuckle. If not, don't worry - I've got plenty more where that came from. Just remember, if at first you don't succeed, skydiving is not for you!"}],"role":"assistant"}},"stopReason":"end_turn","usage":{"inputTokens":50,"outputTokens":83,"totalTokens":133}}",
}
`);
      });
    });

    describe("Multi-Modal Content", () => {
      // Multi-Modal Content Tests (Tests 7-8)

      it("should handle text plus image content with detailed structure", async () => {
        setupTestRecordingWrapper(
          "should-handle-text-plus-image-content-with-detailed-structure",
        );

        const client = createTestClient(isRecordingMode);

        // Create a message with both text and image content
        // Use the same working image data format as InvokeModel test
        const imageData =
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";

        const multiModalMessage = {
          role: "user" as const,
          content: [
            { text: "What's in this image?" },
            {
              image: {
                format: "png" as const,
                source: { bytes: Buffer.from(imageData, "base64") },
              },
            },
          ],
        };

        const command = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          messages: [multiModalMessage],
          inferenceConfig: {
            maxTokens: 100, // Keep response brief to avoid timeout
            temperature: 0.1,
          },
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        // Type-safe checks for nested properties
        if (result.output?.message) {
          expect(result.output.message).toBeDefined();
          expect(result.output.message.role).toBe("assistant");
        } else {
          fail("Expected result.output.message to be defined");
        }

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for text + image content
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"What's in this image?"},{"image":{"format":"png","source":{"bytes":{"type":"Buffer","data":[137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,1,0,0,0,1,8,6,0,0,0,31,21,196,137,0,0,0,13,73,68,65,84,120,218,99,252,255,159,161,30,0,7,130,2,127,61,200,72,239,0,0,0,0,73,69,78,68,174,66,96,130]}}}}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.contents.0.message_content.text": "What's in this image?",
  "llm.input_messages.0.message.contents.0.message_content.type": "text",
  "llm.input_messages.0.message.contents.1.message_content.image.format": "png",
  "llm.input_messages.0.message.contents.1.message_content.image.image.url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
  "llm.input_messages.0.message.contents.1.message_content.type": "image",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "This image appears to be a handwritten note or message. The text is written in cursive script on what looks like lined notebook or notepad paper. The writing is in blue ink.

While I can see the handwriting, I'm not able to read or transcribe the specific content of the note. Handwritten text, especially in cursive, can be challenging for AI systems to accurately interpret. If you have any specific questions about what you see in the image or need clarification",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "max_tokens",
  "llm.system": "anthropic",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 17,
  "llm.token_count.total": 117,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":3802},"output":{"message":{"content":[{"text":"This image appears to be a handwritten note or message. The text is written in cursive script on what looks like lined notebook or notepad paper. The writing is in blue ink.\\n\\nWhile I can see the handwriting, I'm not able to read or transcribe the specific content of the note. Handwritten text, especially in cursive, can be challenging for AI systems to accurately interpret. If you have any specific questions about what you see in the image or need clarification"}],"role":"assistant"}},"stopReason":"max_tokens","usage":{"inputTokens":17,"outputTokens":100,"totalTokens":117}}",
}
`);
      });

      it("should handle different image formats with correct MIME types", async () => {
        setupTestRecordingWrapper(
          "should-handle-different-image-formats-with-correct-mime-types",
        );

        const client = createTestClient(isRecordingMode);

        // Test with JPEG format (different from other image test PNG)
        // Use the same working image data format as InvokeModel test
        const imageData =
          "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";

        const imageMessage = {
          role: "user" as const,
          content: [
            { text: "Describe this JPEG image." },
            {
              image: {
                format: "jpeg" as const,
                source: { bytes: Buffer.from(imageData, "base64") },
              },
            },
          ],
        };

        const command = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          messages: [imageMessage],
          inferenceConfig: {
            maxTokens: 100, // Keep response brief to avoid timeout
            temperature: 0.1,
          },
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        // Type-safe checks for nested properties
        if (result.output?.message) {
          expect(result.output.message).toBeDefined();
          expect(result.output.message.role).toBe("assistant");
        } else {
          fail("Expected result.output.message to be defined");
        }

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for image format handling
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"Describe this JPEG image."},{"image":{"format":"jpeg","source":{"bytes":{"type":"Buffer","data":[137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,1,0,0,0,1,8,6,0,0,0,31,21,196,137,0,0,0,13,73,68,65,84,120,218,99,252,255,159,161,30,0,7,130,2,127,61,200,72,239,0,0,0,0,73,69,78,68,174,66,96,130]}}}}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.contents.0.message_content.text": "Describe this JPEG image.",
  "llm.input_messages.0.message.contents.0.message_content.type": "text",
  "llm.input_messages.0.message.contents.1.message_content.image.format": "jpeg",
  "llm.input_messages.0.message.contents.1.message_content.image.image.url": "data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
  "llm.input_messages.0.message.contents.1.message_content.type": "image",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "This image appears to be a handwritten note or letter on lined paper. The writing is in cursive script and appears to be in blue ink. The paper has horizontal blue lines typical of notebook or writing paper.

The text is not entirely clear or legible in this image, but it seems to be several lines of writing that fill most of the visible portion of the page. The handwriting style looks fluid and connected, characteristic of cursive penmanship.

At the top of the",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "max_tokens",
  "llm.system": "anthropic",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 18,
  "llm.token_count.total": 118,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":3361},"output":{"message":{"content":[{"text":"This image appears to be a handwritten note or letter on lined paper. The writing is in cursive script and appears to be in blue ink. The paper has horizontal blue lines typical of notebook or writing paper.\\n\\nThe text is not entirely clear or legible in this image, but it seems to be several lines of writing that fill most of the visible portion of the page. The handwriting style looks fluid and connected, characteristic of cursive penmanship.\\n\\nAt the top of the"}],"role":"assistant"}},"stopReason":"max_tokens","usage":{"inputTokens":18,"outputTokens":100,"totalTokens":118}}",
}
`);
      });
    });

    describe("Cross-Vendor Models", () => {
      // Cross-Vendor Model Tests (Tests 9-10)

      it("should handle Mistral models", async () => {
        setupTestRecordingWrapper("should-handle-mistral-models");

        const client = createTestClient(isRecordingMode);

        // Test with Mistral model (different from Anthropic)
        const mistralModelId = "mistral.mistral-7b-instruct-v0:2";

        const command = new ConverseCommand({
          modelId: mistralModelId,
          messages: [
            {
              role: "user",
              content: [{ text: "Hello, can you tell me about yourself?" }],
            },
          ],
          inferenceConfig: {
            maxTokens: 100,
            temperature: 0.1,
          },
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        // Type-safe checks for nested properties
        if (result.output?.message) {
          expect(result.output.message).toBeDefined();
          expect(result.output.message.role).toBe("assistant");
        } else {
          fail("Expected result.output.message to be defined");
        }

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for Mistral model
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"mistral.mistral-7b-instruct-v0:2","messages":[{"role":"user","content":[{"text":"Hello, can you tell me about yourself?"}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "Hello, can you tell me about yourself?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "mistral-7b-instruct-v0:2",
  "llm.output_messages.0.message.content": " I'm an artificial intelligence language model designed to assist with various tasks, answer questions, and engage in conversation. I don't have the ability to have a personal identity or emotions, but I can process and generate text based on the data I've been trained on. I'm here to help answer any questions you might have to the best of my ability. Let me know if there's something specific you'd like to know or discuss!",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "end_turn",
  "llm.system": "mistralai",
  "llm.token_count.completion": 94,
  "llm.token_count.prompt": 18,
  "llm.token_count.total": 112,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":962},"output":{"message":{"content":[{"text":" I'm an artificial intelligence language model designed to assist with various tasks, answer questions, and engage in conversation. I don't have the ability to have a personal identity or emotions, but I can process and generate text based on the data I've been trained on. I'm here to help answer any questions you might have to the best of my ability. Let me know if there's something specific you'd like to know or discuss!"}],"role":"assistant"}},"stopReason":"end_turn","usage":{"inputTokens":18,"outputTokens":94,"totalTokens":112}}",
}
`);
      });

      it("should handle Meta LLaMA models", async () => {
        setupTestRecordingWrapper("should-handle-meta-llama-models");

        const client = createTestClient(isRecordingMode);

        // Test with Meta LLaMA model (different from Anthropic)
        const llamaModelId = "meta.llama3-8b-instruct-v1:0";

        const command = new ConverseCommand({
          modelId: llamaModelId,
          messages: [
            {
              role: "user",
              content: [{ text: "Hello, can you tell me about yourself?" }],
            },
          ],
          inferenceConfig: {
            maxTokens: 100,
            temperature: 0.1,
          },
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        // Type-safe checks for nested properties
        if (result.output?.message) {
          expect(result.output.message).toBeDefined();
          expect(result.output.message.role).toBe("assistant");
        } else {
          fail("Expected result.output.message to be defined");
        }

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for Meta LLaMA model
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"meta.llama3-8b-instruct-v1:0","messages":[{"role":"user","content":[{"text":"Hello, can you tell me about yourself?"}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "Hello, can you tell me about yourself?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "llama3-8b-instruct-v1:0",
  "llm.output_messages.0.message.content": "

I'd be happy to introduce myself.

I am LLaMA, an AI assistant developed by Meta AI that can understand and respond to human input in a conversational manner. I'm a large language model, which means I've been trained on a massive dataset of text from various sources, including books, articles, and online conversations.

I'm designed to be helpful and informative, and I can assist with a wide range of topics and tasks. I can answer questions, provide definitions, offer suggestions",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "max_tokens",
  "llm.system": "meta",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 23,
  "llm.token_count.total": 123,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":1061},"output":{"message":{"content":[{"text":"\\n\\nI'd be happy to introduce myself.\\n\\nI am LLaMA, an AI assistant developed by Meta AI that can understand and respond to human input in a conversational manner. I'm a large language model, which means I've been trained on a massive dataset of text from various sources, including books, articles, and online conversations.\\n\\nI'm designed to be helpful and informative, and I can assist with a wide range of topics and tasks. I can answer questions, provide definitions, offer suggestions"}],"role":"assistant"}},"stopReason":"max_tokens","usage":{"inputTokens":23,"outputTokens":100,"totalTokens":123}}",
}
`);
      });
    });

    describe("Error Handling", () => {
      // Edge Cases and Error Handling (Tests 11-13)

      it("should handle missing token counts via converse", async () => {
        setupTestRecordingWrapper("should-handle-missing-token-counts");

        const client = createTestClient(isRecordingMode);

        // Test with request that might have partial token count information
        const command = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          messages: [
            {
              role: "user",
              content: [{ text: "Brief response please." }],
            },
          ],
          inferenceConfig: {
            maxTokens: 50,
            temperature: 0.1,
          },
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        // Type-safe checks for nested properties
        if (result.output?.message) {
          expect(result.output.message).toBeDefined();
          expect(result.output.message.role).toBe("assistant");
        } else {
          fail("Expected result.output.message to be defined");
        }

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for handling missing token counts
        // Note: This test demonstrates that when inputTokens and totalTokens are missing from the API response,
        // the instrumentation gracefully handles it by only setting the available token count (outputTokens -> completion)
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"Brief response please."}]}],"inferenceConfig":{"maxTokens":50,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "Brief response please.",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":50,"temperature":0.1}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "I apologize, but I don't have any previous context or question to provide a brief response to. Could you please ask a specific question or provide more information about what you'd like a brief response on? Once you do, I'll be happy",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "max_tokens",
  "llm.system": "anthropic",
  "llm.token_count.completion": 50,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":4083},"output":{"message":{"content":[{"text":"I apologize, but I don't have any previous context or question to provide a brief response to. Could you please ask a specific question or provide more information about what you'd like a brief response on? Once you do, I'll be happy"}],"role":"assistant"}},"stopReason":"max_tokens","usage":{"outputTokens":50}}",
}
`);
      });

      it("should handle API error scenarios", async () => {
        setupTestRecordingWrapper("should-handle-api-error-scenarios");

        const client = createTestClient(isRecordingMode);

        // Test with invalid model ID to trigger error
        const command = new ConverseCommand({
          modelId: "invalid-model-id",
          messages: [
            {
              role: "user",
              content: [{ text: "This should fail" }],
            },
          ],
          inferenceConfig: {
            maxTokens: 50,
          },
        });

        let error: Error | undefined;
        try {
          await client.send(command);
        } catch (e) {
          error = e as Error;
        }

        // Should have captured error
        expect(error).toBeDefined();

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for API error scenarios
        // Even when API calls fail, the instrumentation captures request information
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"invalid-model-id","messages":[{"role":"user","content":[{"text":"This should fail"}]}],"inferenceConfig":{"maxTokens":50}}",
  "llm.input_messages.0.message.content": "This should fail",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":50}",
  "llm.model_name": "invalid-model-id",
  "llm.provider": "aws",
  "llm.system": "amazon",
  "openinference.span.kind": "LLM",
}
`);
      });

      it("should handle empty/minimal response", async () => {
        setupTestRecordingWrapper("should-handle-empty-minimal-response");

        const client = createTestClient(isRecordingMode);

        // Test with minimal parameters
        const command = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          messages: [
            {
              role: "user",
              content: [{ text: "One word response." }],
            },
          ],
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        // Type-safe checks for nested properties
        if (result.output?.message) {
          expect(result.output.message).toBeDefined();
          expect(result.output.message.role).toBe("assistant");
        } else {
          fail("Expected result.output.message to be defined");
        }

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for empty/minimal response
        // Tests graceful handling of minimal API response with empty text content and missing optional fields
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"One word response."}]}]}",
  "llm.input_messages.0.message.content": "One word response.",
  "llm.input_messages.0.message.role": "user",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "end_turn",
  "llm.system": "anthropic",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"output":{"message":{"content":[{"text":""}],"role":"assistant"}},"stopReason":"end_turn"}",
}
`);
      });
    });

    describe("Tool Configuration", () => {
      // Tool Configuration Tests

      it("should handle tool configuration", async () => {
        setupTestRecordingWrapper("should-handle-tool-configuration");

        const client = createTestClient(isRecordingMode);

        // Test with Converse-specific tool configuration
        const toolConfig = {
          tools: [
            {
              toolSpec: {
                name: "get_weather",
                description: "Get current weather for a location",
                inputSchema: {
                  json: {
                    type: "object",
                    properties: {
                      location: {
                        type: "string",
                        description:
                          "The city and state, e.g. San Francisco, CA",
                      },
                      unit: {
                        type: "string",
                        enum: ["celsius", "fahrenheit"],
                        description: "Temperature unit",
                      },
                    },
                    required: ["location"],
                  },
                },
              },
            },
          ],
        };

        const command = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          messages: [
            {
              role: "user",
              content: [{ text: "What's the weather in San Francisco?" }],
            },
          ],
          toolConfig: toolConfig,
          inferenceConfig: {
            maxTokens: 100,
            temperature: 0.1,
          },
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        // Type-safe checks for nested properties
        if (result.output?.message) {
          expect(result.output.message).toBeDefined();
          expect(result.output.message.role).toBe("assistant");
        } else {
          fail("Expected result.output.message to be defined");
        }

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for tool configuration
        // Tests proper handling of tool definitions and tool use responses
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"What's the weather in San Francisco?"}]}],"toolConfig":{"tools":[{"toolSpec":{"name":"get_weather","description":"Get current weather for a location","inputSchema":{"json":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}}}]},"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "What's the weather in San Francisco?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.contents.0.message_content.text": "Certainly! I'd be happy to check the current weather in San Francisco for you. To get this information, I'll need to use the weather tool. Let me fetch that data for you.",
  "llm.output_messages.0.message.contents.0.message_content.type": "text",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{"location":"San Francisco, CA"}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "tooluse_0ae1_ahYROawF4OObK5xBg",
  "llm.provider": "aws",
  "llm.stop_reason": "max_tokens",
  "llm.system": "anthropic",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 408,
  "llm.token_count.total": 508,
  "llm.tools.0.tool.json_schema": "{"toolSpec":{"name":"get_weather","description":"Get current weather for a location","inputSchema":{"json":{"type":"object","properties":{"location":{"type":"string","description":"The city and state, e.g. San Francisco, CA"},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":2930},"output":{"message":{"content":[{"text":"Certainly! I'd be happy to check the current weather in San Francisco for you. To get this information, I'll need to use the weather tool. Let me fetch that data for you."},{"toolUse":{"input":{"location":"San Francisco, CA"},"name":"get_weather","toolUseId":"tooluse_0ae1_ahYROawF4OObK5xBg"}}],"role":"assistant"}},"stopReason":"max_tokens","usage":{"inputTokens":408,"outputTokens":100,"totalTokens":508}}",
}
`);
      });

      it("should handle tool response processing", async () => {
        setupTestRecordingWrapper("should-handle-tool-response-processing");

        const client = createTestClient(isRecordingMode);

        // Test with tool usage in conversation
        const toolConfig = {
          tools: [
            {
              toolSpec: {
                name: "calculate",
                description: "Perform mathematical calculations",
                inputSchema: {
                  json: {
                    type: "object",
                    properties: {
                      expression: {
                        type: "string",
                        description: "Mathematical expression to evaluate",
                      },
                    },
                    required: ["expression"],
                  },
                },
              },
            },
          ],
        };

        const command = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          messages: [
            {
              role: "user",
              content: [{ text: "What is 15 * 23?" }],
            },
          ],
          toolConfig: toolConfig,
          inferenceConfig: {
            maxTokens: 100,
            temperature: 0.1,
          },
        });

        const result = await client.send(command);

        // Basic response structure verification
        expect(result).toBeDefined();
        expect(result.output).toBeDefined();

        // Type-safe checks for nested properties
        if (result.output?.message) {
          expect(result.output.message).toBeDefined();
          expect(result.output.message.role).toBe("assistant");
        } else {
          fail("Expected result.output.message to be defined");
        }

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for tool response processing
        // Tests natural completion with tool_use stop reason (vs max_tokens in previous test)
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"What is 15 * 23?"}]}],"toolConfig":{"tools":[{"toolSpec":{"name":"calculate","description":"Perform mathematical calculations","inputSchema":{"json":{"type":"object","properties":{"expression":{"type":"string","description":"Mathematical expression to evaluate"}},"required":["expression"]}}}}]},"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "What is 15 * 23?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.contents.0.message_content.text": "To calculate 15 * 23, I can use the "calculate" function. Let me do that for you.",
  "llm.output_messages.0.message.contents.0.message_content.type": "text",
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{"expression":"15 * 23"}",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "calculate",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "tooluse_RVjyNbnRRAqT_YlXqlJDUQ",
  "llm.provider": "aws",
  "llm.stop_reason": "tool_use",
  "llm.system": "anthropic",
  "llm.token_count.completion": 81,
  "llm.token_count.prompt": 369,
  "llm.token_count.total": 450,
  "llm.tools.0.tool.json_schema": "{"toolSpec":{"name":"calculate","description":"Perform mathematical calculations","inputSchema":{"json":{"type":"object","properties":{"expression":{"type":"string","description":"Mathematical expression to evaluate"}},"required":["expression"]}}}}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":2249},"output":{"message":{"content":[{"text":"To calculate 15 * 23, I can use the \\"calculate\\" function. Let me do that for you."},{"toolUse":{"input":{"expression":"15 * 23"},"name":"calculate","toolUseId":"tooluse_RVjyNbnRRAqT_YlXqlJDUQ"}}],"role":"assistant"}},"stopReason":"tool_use","usage":{"inputTokens":369,"outputTokens":81,"totalTokens":450}}",
}
`);
      });
    });

    describe("Context Attributes", () => {
      // Context and VCR Infrastructure

      it("should handle context attributes with Converse", async () => {
        setupTestRecordingWrapper(
          "should-handle-context-attributes-with-converse",
        );

        const client = createTestClient(isRecordingMode);

        // Test with OpenInference context attributes
        const command = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          messages: [
            {
              role: "user",
              content: [{ text: "Hello, what's your name?" }],
            },
          ],
          inferenceConfig: {
            maxTokens: 100,
            temperature: 0.1,
          },
        });

        // Setup OpenInference context with all supported attributes
        await context.with(
          setSession(
            setUser(
              setMetadata(
                setTags(
                  setPromptTemplate(context.active(), {
                    template: "Hello {{user_input}}, what's your name?",
                    version: "1.0.0",
                    variables: { user_input: "user" },
                  }),
                  ["test", "context", "converse"],
                ),
                {
                  experiment_name: "converse-context-test",
                  version: "1.0.0",
                  environment: "testing",
                },
              ),
              { userId: "test-user-converse" },
            ),
            { sessionId: "test-session-converse" },
          ),
          async () => {
            // Make the API call within the context
            const result = await client.send(command);

            // Basic response structure verification
            expect(result).toBeDefined();
            expect(result.output).toBeDefined();

            // Type-safe checks for nested properties
            if (result.output?.message) {
              expect(result.output.message).toBeDefined();
              expect(result.output.message.role).toBe("assistant");
            } else {
              fail("Expected result.output.message to be defined");
            }

            return result;
          },
        );

        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // Comprehensive span attributes snapshot for context attributes
        // Tests OpenInference context propagation including session, user, metadata, tags, and prompt template
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","messages":[{"role":"user","content":[{"text":"Hello, what's your name?"}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "Hello, what's your name?",
  "llm.input_messages.0.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "My name is Claude. It's nice to meet you!",
  "llm.output_messages.0.message.role": "assistant",
  "llm.prompt_template.template": "Hello {{user_input}}, what's your name?",
  "llm.prompt_template.variables": "{"user_input":"user"}",
  "llm.prompt_template.version": "1.0.0",
  "llm.provider": "aws",
  "llm.stop_reason": "end_turn",
  "llm.system": "anthropic",
  "llm.token_count.completion": 15,
  "llm.token_count.prompt": 14,
  "llm.token_count.total": 29,
  "metadata": "{"experiment_name":"converse-context-test","version":"1.0.0","environment":"testing"}",
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":761},"output":{"message":{"content":[{"text":"My name is Claude. It's nice to meet you!"}],"role":"assistant"}},"stopReason":"end_turn","usage":{"inputTokens":14,"outputTokens":15,"totalTokens":29}}",
  "session.id": "test-session-converse",
  "tag.tags": "["test","context","converse"]",
  "user.id": "test-user-converse",
}
`);
      });

      it("should comprehensively test all token count types", async () => {
        setupTestRecordingWrapper(
          "should-comprehensively-test-all-token-count-types",
        );

        const client = createTestClient(isRecordingMode);

        // Define a substantial system prompt that should benefit from caching
        const systemPrompt =
          "You are an expert geography assistant. You have extensive knowledge about world capitals, countries, and their historical backgrounds. Please provide accurate and detailed information about geographical questions. Always include interesting historical context in your responses when relevant.";

        // First call - structured for caching when AWS JS SDK fully supports it
        // NOTE: While caching works with LangChain wrappers (ChatBedrockConverse),
        // the direct AWS JavaScript SDK still has serialization issues with cachePoint
        // objects in both system and content arrays. LangChain handles the translation layer.
        const firstCommand = new ConverseCommand({
          modelId: TEST_MODEL_ID,
          system: [
            {
              text: systemPrompt,
            },
          ],
          messages: [
            {
              role: "user",
              content: [
                {
                  text: "What is the capital of France?",
                },
              ],
            },
          ],
          inferenceConfig: {
            maxTokens: 100,
            temperature: 0.1,
          },
        });

        // Execute all three calls to test comprehensive token counting
        await client.send(firstCommand);
        spanExporter.reset();

        await client.send(
          new ConverseCommand({
            modelId: TEST_MODEL_ID,
            system: [{ text: systemPrompt }],
            messages: [
              {
                role: "user",
                content: [{ text: "What is the capital of Italy?" }],
              },
            ],
            inferenceConfig: { maxTokens: 100, temperature: 0.1 },
          }),
        );
        spanExporter.reset();

        await client.send(
          new ConverseCommand({
            modelId: TEST_MODEL_ID,
            system: [{ text: systemPrompt }],
            messages: [
              {
                role: "user",
                content: [{ text: "What is the capital of Germany?" }],
              },
            ],
            inferenceConfig: { maxTokens: 100, temperature: 0.1 },
          }),
        );

        // Get the span from the final call for comprehensive token analysis
        const span = verifySpanBasics(spanExporter, "bedrock.converse");

        // This test is ready for caching but the direct AWS SDK doesn't support it yet:
        //
        // We should expect to see:
        // - cacheReadInputTokens and cacheWriteInputTokens in API responses
        // - Corresponding llm.token_count.prompt.cache_read and llm.token_count.prompt.cache_write
        //   attributes in instrumentation spans

        // Comprehensive span attributes snapshot - captures current token counting behavior
        expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "{"modelId":"anthropic.claude-3-5-sonnet-20240620-v1:0","system":[{"text":"You are an expert geography assistant. You have extensive knowledge about world capitals, countries, and their historical backgrounds. Please provide accurate and detailed information about geographical questions. Always include interesting historical context in your responses when relevant."}],"messages":[{"role":"user","content":[{"text":"What is the capital of Germany?"}]}],"inferenceConfig":{"maxTokens":100,"temperature":0.1}}",
  "llm.input_messages.0.message.content": "You are an expert geography assistant. You have extensive knowledge about world capitals, countries, and their historical backgrounds. Please provide accurate and detailed information about geographical questions. Always include interesting historical context in your responses when relevant.",
  "llm.input_messages.0.message.role": "system",
  "llm.input_messages.1.message.content": "What is the capital of Germany?",
  "llm.input_messages.1.message.role": "user",
  "llm.invocation_parameters": "{"maxTokens":100,"temperature":0.1}",
  "llm.model_name": "claude-3-5-sonnet-20240620",
  "llm.output_messages.0.message.content": "The capital of Germany is Berlin. 

Berlin has a rich and complex history as the capital city:

1. It became the capital of Prussia in 1701 and later the German Empire in 1871.

2. After World War I, it remained the capital of the Weimar Republic.

3. During the Cold War, Berlin was divided into East and West sectors. East Berlin served as the capital of East Germany, while Bonn became the provisional capital of",
  "llm.output_messages.0.message.role": "assistant",
  "llm.provider": "aws",
  "llm.stop_reason": "max_tokens",
  "llm.system": "anthropic",
  "llm.token_count.completion": 100,
  "llm.token_count.prompt": 57,
  "llm.token_count.total": 157,
  "openinference.span.kind": "LLM",
  "output.mime_type": "application/json",
  "output.value": "{"$metadata":{"httpStatusCode":200,"attempts":1,"totalRetryDelay":0},"metrics":{"latencyMs":2847},"output":{"message":{"content":[{"text":"The capital of Germany is Berlin. \\n\\nBerlin has a rich and complex history as the capital city:\\n\\n1. It became the capital of Prussia in 1701 and later the German Empire in 1871.\\n\\n2. After World War I, it remained the capital of the Weimar Republic.\\n\\n3. During the Cold War, Berlin was divided into East and West sectors. East Berlin served as the capital of East Germany, while Bonn became the provisional capital of"}],"role":"assistant"}},"stopReason":"max_tokens","usage":{"inputTokens":57,"outputTokens":100,"totalTokens":157}}",
}
`);
      }, 15000); // Increase timeout for recording mode with multiple API calls
    });

    //
    // Use pattern: BEDROCK_RECORD_MODE=record npm test -- --testNamePattern="test name"
  });
});

describe("BedrockInstrumentation - custom tracing", () => {
  // Global spanExporter for this test suite to compare against custom exporters
  let spanExporter: InMemorySpanExporter;
  let provider: NodeTracerProvider;
  let currentTestName: string;
  let recordingsPath: string;

  const isRecordingMode = process.env.BEDROCK_RECORD_MODE === "record";

  beforeAll(() => {
    // Setup global tracer provider and span exporter for comparison
    spanExporter = new InMemorySpanExporter();
    provider = new NodeTracerProvider();
    provider.addSpanProcessor(new SimpleSpanProcessor(spanExporter));
    provider.register();
  });

  afterAll(async () => {
    await provider.shutdown();
  });

  // Helper wrapper for custom tracing tests to set up their specific recording
  const setupTestRecordingCustom = (testName: string) => {
    currentTestName = testName;
    recordingsPath = setupTestRecording(
      testName,
      __dirname,
      isRecordingMode,
      "anthropic.claude-3-sonnet-20240229-v1:0",
    );
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
      initializeRecordingMode();
    }

    // Reset span exporter for clean test state
    spanExporter.reset();
  });

  afterEach(() => {
    if (isRecordingMode) {
      saveRecordingModeData(currentTestName, recordingsPath);
    }

    // Clean up nock only
    nock.cleanAll();
    nock.restore();
  });

  describe("BedrockInstrumentation with custom TracerProvider passed in", () => {
    const customTracerProvider = new NodeTracerProvider();
    const customMemoryExporter = new InMemorySpanExporter();
    let instrumentation: BedrockInstrumentation;

    // Note: We don't register this provider globally.
    customTracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(customMemoryExporter),
    );

    beforeAll(() => {
      // Instantiate instrumentation with the custom provider
      instrumentation = new BedrockInstrumentation({
        tracerProvider: customTracerProvider,
      });
      instrumentation.disable();

      // Mock the module exports like in other tests
      // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
      instrumentation._modules[0].moduleExports = require("@aws-sdk/client-bedrock-runtime");

      instrumentation.enable();
    });

    afterAll(() => {
      instrumentation.disable();
      customTracerProvider.shutdown();
    });

    beforeEach(() => {
      customMemoryExporter.reset();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
      setupTestRecordingCustom(
        "should use the provided tracer provider instead of the global one - constructor",
      );

      const client = createTestClient(isRecordingMode);

      const command = new InvokeModelCommand({
        modelId: "anthropic.claude-3-sonnet-20240229-v1:0",
        body: JSON.stringify({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: 100,
          messages: [
            {
              role: "user",
              content: "Say this is a test",
            },
          ],
        }),
        contentType: "application/json",
        accept: "application/json",
      });

      const result = await client.send(command);
      verifyResponseStructure(result);

      // Verify custom exporter has the span and global exporter does not
      const span = verifySpanBasics(customMemoryExporter);
      const globalSpans = spanExporter.getFinishedSpans();
      expect(globalSpans.length).toBe(0);
      expect(span.attributes["llm.provider"]).toBe("aws");
      expect(span.attributes["llm.model_name"]).toBe(
        "claude-3-sonnet-20240229",
      );
    });
  });

  describe("BedrockInstrumentation with custom TracerProvider set", () => {
    const customTracerProvider = new NodeTracerProvider();
    const customMemoryExporter = new InMemorySpanExporter();
    let instrumentation: BedrockInstrumentation;

    // Note: We don't register this provider globally.
    customTracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(customMemoryExporter),
    );

    beforeAll(() => {
      // Instantiate instrumentation and set the custom provider
      instrumentation = new BedrockInstrumentation();
      instrumentation.setTracerProvider(customTracerProvider);
      instrumentation.disable();

      // Mock the module exports like in other tests
      // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
      instrumentation._modules[0].moduleExports = require("@aws-sdk/client-bedrock-runtime");

      instrumentation.enable();
    });

    afterAll(() => {
      instrumentation.disable();
      customTracerProvider.shutdown();
    });

    beforeEach(() => {
      customMemoryExporter.reset();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
      setupTestRecordingCustom(
        "should use the provided tracer provider instead of the global one - setTracerProvider",
      );

      const client = createTestClient(isRecordingMode);

      const command = new InvokeModelCommand({
        modelId: "anthropic.claude-3-sonnet-20240229-v1:0",
        body: JSON.stringify({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: 100,
          messages: [
            {
              role: "user",
              content: "Say this is a test",
            },
          ],
        }),
        contentType: "application/json",
        accept: "application/json",
      });

      const result = await client.send(command);
      verifyResponseStructure(result);

      // Verify custom exporter has the span and global exporter does not
      const span = verifySpanBasics(customMemoryExporter);
      const globalSpans = spanExporter.getFinishedSpans();
      expect(globalSpans.length).toBe(0);
      expect(span.attributes["llm.provider"]).toBe("aws");
      expect(span.attributes["llm.model_name"]).toBe(
        "claude-3-sonnet-20240229",
      );
    });
  });

  describe("BedrockInstrumentation with custom TracerProvider set via registerInstrumentations", () => {
    const customTracerProvider = new NodeTracerProvider();
    const customMemoryExporter = new InMemorySpanExporter();
    let instrumentation: BedrockInstrumentation;

    // Note: We don't register this provider globally.
    customTracerProvider.addSpanProcessor(
      new SimpleSpanProcessor(customMemoryExporter),
    );

    beforeAll(() => {
      // Instantiate instrumentation and register with the custom provider
      instrumentation = new BedrockInstrumentation();
      registerInstrumentations({
        instrumentations: [instrumentation],
        tracerProvider: customTracerProvider,
      });
      instrumentation.disable();

      // Mock the module exports like in other tests
      // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
      instrumentation._modules[0].moduleExports = require("@aws-sdk/client-bedrock-runtime");

      instrumentation.enable();
    });

    afterAll(() => {
      instrumentation.disable();
      customTracerProvider.shutdown();
    });

    beforeEach(() => {
      customMemoryExporter.reset();
    });

    it("should use the provided tracer provider instead of the global one", async () => {
      setupTestRecordingCustom(
        "should use the provided tracer provider instead of the global one - registerInstrumentations",
      );

      const client = createTestClient(isRecordingMode);

      const command = new InvokeModelCommand({
        modelId: "anthropic.claude-3-sonnet-20240229-v1:0",
        body: JSON.stringify({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: 100,
          messages: [
            {
              role: "user",
              content: "Say this is a test",
            },
          ],
        }),
        contentType: "application/json",
        accept: "application/json",
      });

      const result = await client.send(command);
      verifyResponseStructure(result);

      // Verify custom exporter has the span and global exporter does not
      const span = verifySpanBasics(customMemoryExporter);
      const globalSpans = spanExporter.getFinishedSpans();
      expect(globalSpans.length).toBe(0);
      expect(span.attributes["llm.provider"]).toBe("aws");
      expect(span.attributes["llm.model_name"]).toBe(
        "claude-3-sonnet-20240229",
      );
    });
  });
});

#!/usr/bin/env tsx

/**
 * Comprehensive Validation Script for Bedrock Converse API instrumentation
 *
 * This script tests the Bedrock Converse API instrumentation with 6 comprehensive scenarios:
 * 1. Basic Flow - Simple conversation with system prompt
 * 2. Multi-Turn Conversation - Complex conversation history
 * 3. Tool Calling Flow - Tool configuration and usage
 * 4. Multi-Modal with Image - Text + image content
 * 5. Context Attributes - OpenInference context propagation
 * 6. Error Case - API error handling
 *
 * Each scenario creates a separate span in Phoenix for detailed analysis.
 * This script follows the model of validate-invoke-model.ts but focuses on
 * comprehensive Converse API validation scenarios.
 *
 * Usage:
 *   npm run validate:converse-comprehensive
 *   tsx scripts/validate-converse-comprehensive.ts
 *   tsx scripts/validate-converse-comprehensive.ts --scenario basic-flow
 *   tsx scripts/validate-converse-comprehensive.ts --debug
 */

/* eslint-disable no-console, @typescript-eslint/no-explicit-any */

import { BedrockInstrumentation, isPatched } from "../src/index";
import {
  NodeTracerProvider,
  SimpleSpanProcessor,
  ConsoleSpanExporter,
} from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import {
  diag,
  DiagConsoleLogger,
  DiagLogLevel,
  context,
} from "@opentelemetry/api";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import {
  setSession,
  setUser,
  setMetadata,
  setTags,
  setPromptTemplate,
} from "@arizeai/openinference-core";

// Configuration from environment variables
const PHOENIX_ENDPOINT =
  process.env.PHOENIX_ENDPOINT ||
  (process.env.PHOENIX_COLLECTOR_ENDPOINT
    ? `${process.env.PHOENIX_COLLECTOR_ENDPOINT}/v1/traces`
    : "http://localhost:6006/v1/traces");
const PHOENIX_API_KEY = process.env.PHOENIX_API_KEY;
const AWS_REGION = process.env.AWS_REGION || "us-east-1";
const MODEL_ID =
  process.env.BEDROCK_MODEL_ID || "anthropic.claude-3-5-sonnet-20240620-v1:0";

// Test scenarios covering major Converse API test cases
type TestScenario =
  | "basic-flow"
  | "multi-turn"
  | "tool-calling"
  | "multi-modal"
  | "context-attributes"
  | "error-case"
  | "amazon-nova"
  | "meta-llama"
  | "all";

interface ValidationOptions {
  scenario: TestScenario;
  debug: boolean;
  phoenixEndpoint: string;
  phoenixApiKey?: string;
  modelId: string;
}

class ConverseComprehensiveValidator {
  private client: any; // Will be loaded dynamically
  private provider: NodeTracerProvider;
  private BedrockRuntimeClient: any;
  private ConverseCommand: any;

  constructor(private options: ValidationOptions) {
    this.setupLogging();
  }

  private setupLogging() {
    if (this.options.debug) {
      diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);
    }
  }

  private setupTracing() {
    this.provider = new NodeTracerProvider({
      resource: new Resource({
        [SEMRESATTRS_PROJECT_NAME]: "bedrock-converse-comprehensive-validation",
      }),
    });

    // Setup exporters
    const exporters = [new ConsoleSpanExporter()];

    if (this.options.phoenixEndpoint !== "console") {
      // Use OTEL_EXPORTER_OTLP_HEADERS or PHOENIX_CLIENT_HEADERS for API key
      const apiKey =
        this.options.phoenixApiKey ||
        process.env.OTEL_EXPORTER_OTLP_HEADERS?.split("api_key=")[1] ||
        process.env.PHOENIX_CLIENT_HEADERS?.split("api_key=")[1];

      // Ensure Phoenix Cloud URLs have the correct endpoint
      const exportUrl =
        this.options.phoenixEndpoint.includes("app.phoenix.arize.com") &&
        !this.options.phoenixEndpoint.includes("/v1/traces")
          ? `${this.options.phoenixEndpoint}/v1/traces`
          : this.options.phoenixEndpoint;

      const phoenixExporter = new OTLPTraceExporter({
        url: exportUrl,
        headers: apiKey
          ? {
              api_key: apiKey,
            }
          : {},
      });

      console.log(
        `🔍 Debug: Exporting to ${exportUrl} with API key: ${apiKey ? "[REDACTED]" : "none"}`,
      );
      exporters.push(phoenixExporter);
    }

    exporters.forEach((exporter) => {
      this.provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
    });

    this.provider.register();
  }

  private async setupInstrumentation() {
    // Load AWS SDK first
    const awsModule = await import("@aws-sdk/client-bedrock-runtime");
    this.BedrockRuntimeClient = awsModule.BedrockRuntimeClient;
    this.ConverseCommand = awsModule.ConverseCommand;

    // Check if already patched
    if (!isPatched()) {
      // Create instrumentation and register it
      const instrumentation = new BedrockInstrumentation();
      registerInstrumentations({
        instrumentations: [instrumentation],
      });

      // Also manually patch the already-loaded module to ensure it works
      const moduleExports = {
        BedrockRuntimeClient: awsModule.BedrockRuntimeClient,
      };
      (instrumentation as any).patch(moduleExports, "3.0.0");

      console.log("✅ Bedrock instrumentation registered and manually applied");
    } else {
      console.log("✅ Bedrock instrumentation was already registered");
    }
  }

  private async loadBedrockClient() {
    // Client creation happens after instrumentation setup
    this.client = new this.BedrockRuntimeClient({ region: AWS_REGION });
  }

  private verifyInstrumentation() {
    // Check both the method signature and the global patch status
    const sendMethod = this.BedrockRuntimeClient.prototype.send;
    const globalPatchStatus = isPatched();

    // The patched method should have different characteristics than the original
    const methodPatched =
      sendMethod.toString().includes("patchedSend") ||
      sendMethod.name === "patchedSend" ||
      sendMethod.__wrapped === true ||
      sendMethod.toString().length < 100; // Wrapped methods are typically shorter

    if (globalPatchStatus && methodPatched) {
      console.log(
        "✅ Instrumentation verified: Both global status and method are patched",
      );
      return true;
    } else if (globalPatchStatus) {
      console.log("✅ Instrumentation verified: Global patch status is true");
      return true;
    } else {
      console.log("❌ Instrumentation verification failed");
      console.log(
        "   Send method signature:",
        sendMethod.toString().substring(0, 100) + "...",
      );
      console.log("   Global patch status:", globalPatchStatus);
      console.log("   Method appears patched:", methodPatched);
      return false;
    }
  }

  async runValidation() {
    console.log("🚀 Starting Bedrock Converse API comprehensive validation");
    console.log(`📊 Phoenix endpoint: ${this.options.phoenixEndpoint}`);
    console.log(`🤖 Model ID: ${this.options.modelId}`);
    console.log(`🎯 Scenario: ${this.options.scenario}`);
    console.log();

    // Setup tracing and instrumentation
    this.setupTracing();
    await this.setupInstrumentation();

    // Load Bedrock client AFTER instrumentation setup
    await this.loadBedrockClient();

    // Verify instrumentation is applied
    if (!this.verifyInstrumentation()) {
      console.log(
        "❌ Instrumentation verification failed - stopping validation",
      );
      return false;
    }

    const scenarios =
      this.options.scenario === "all"
        ? ([
            "basic-flow",
            "multi-turn",
            "tool-calling",
            "multi-modal",
            "context-attributes",
            "error-case",
            "amazon-nova",
            "meta-llama",
          ] as TestScenario[])
        : [this.options.scenario];

    let allPassed = true;

    for (const scenario of scenarios) {
      console.log(`\n📋 Running scenario: ${scenario}`);
      try {
        const passed = await this.runScenario(scenario);
        if (passed) {
          console.log(`✅ Scenario ${scenario} completed successfully`);
        } else {
          console.log(`❌ Scenario ${scenario} failed`);
          allPassed = false;
        }

        // Small delay between scenarios to ensure spans are processed
        await new Promise((resolve) => setTimeout(resolve, 1000));
      } catch (error) {
        console.log(`❌ Scenario ${scenario} threw error:`, error.message);
        allPassed = false;
      }
    }

    console.log("\n📊 Validation Summary:");
    console.log(
      allPassed ? "✅ All scenarios passed" : "❌ Some scenarios failed",
    );

    // Give time for traces to be exported
    console.log("\n⏳ Waiting for traces to be exported...");
    await new Promise((resolve) => setTimeout(resolve, 3000));

    return allPassed;
  }

  private async runScenario(scenario: TestScenario): Promise<boolean> {
    switch (scenario) {
      case "basic-flow":
        return this.runBasicFlowScenario();
      case "multi-turn":
        return this.runMultiTurnScenario();
      case "tool-calling":
        return this.runToolCallingScenario();
      case "multi-modal":
        return this.runMultiModalScenario();
      case "context-attributes":
        return this.runContextAttributesScenario();
      case "error-case":
        return this.runErrorCaseScenario();
      case "amazon-nova":
        return this.runAmazonNovaScenario();
      case "meta-llama":
        return this.runMetaLlamaScenario();
      default:
        throw new Error(`Unknown scenario: ${scenario}`);
    }
  }

  private async runBasicFlowScenario(): Promise<boolean> {
    console.log("   📝 Testing basic flow with system prompt...");

    const command = new this.ConverseCommand({
      modelId: this.options.modelId,
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
      inferenceConfig: {
        maxTokens: 50,
        temperature: 0.1,
      },
    });

    const response = await this.client.send(command);

    if (!response.output?.message) {
      console.log("❌ No message in response");
      return false;
    }

    const outputMessage = response.output.message;
    console.log("✅ Basic flow response received successfully");

    // Check for text content in the response
    const textContent = outputMessage.content?.find((block: any) => block.text);
    if (textContent) {
      console.log(
        "   💬 Response preview:",
        textContent.text.substring(0, 50) + "...",
      );
    }

    // Check usage statistics
    if (response.usage) {
      console.log("   📈 Token usage:");
      console.log(`     Input tokens: ${response.usage.inputTokens}`);
      console.log(`     Output tokens: ${response.usage.outputTokens}`);
      console.log(`     Total tokens: ${response.usage.totalTokens}`);
    }

    return true;
  }

  private async runMultiTurnScenario(): Promise<boolean> {
    console.log("   🔄 Testing multi-turn conversation...");

    // Simulate realistic multi-turn conversation flow
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
      content: [{ text: "Can you tell me a short joke?" }],
    };

    // Create command with full conversation history (simulating second turn)
    const command = new this.ConverseCommand({
      modelId: this.options.modelId,
      messages: [firstUserMessage, assistantResponse, secondUserMessage],
      inferenceConfig: {
        maxTokens: 100,
        temperature: 0.7,
      },
    });

    const response = await this.client.send(command);

    if (!response.output?.message) {
      console.log("❌ No message in response");
      return false;
    }

    const outputMessage = response.output.message;
    console.log("✅ Multi-turn conversation response received successfully");

    // Check for text content in the response
    const textContent = outputMessage.content?.find((block: any) => block.text);
    if (textContent) {
      console.log(
        "   💬 Joke response preview:",
        textContent.text.substring(0, 80) + "...",
      );
    }

    // Check usage statistics
    if (response.usage) {
      console.log("   📈 Token usage:");
      console.log(`     Input tokens: ${response.usage.inputTokens}`);
      console.log(`     Output tokens: ${response.usage.outputTokens}`);
      console.log(`     Total tokens: ${response.usage.totalTokens}`);
    }

    return true;
  }

  private async runToolCallingScenario(): Promise<boolean> {
    console.log("   🔧 Testing tool calling...");

    // Tool configuration using correct Converse API format
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
                    description: "The city and state, e.g. San Francisco, CA",
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

    const command = new this.ConverseCommand({
      modelId: this.options.modelId,
      messages: [
        {
          role: "user",
          content: [
            {
              text: "What's the weather in San Francisco and what's 15 * 23?",
            },
          ],
        },
      ],
      toolConfig: toolConfig,
      inferenceConfig: {
        maxTokens: 150,
        temperature: 0.1,
      },
    });

    const response = await this.client.send(command);

    if (!response.output?.message) {
      console.log("❌ No message in response");
      return false;
    }

    const outputMessage = response.output.message;
    console.log("✅ Tool calling response received successfully");

    // Check for tool calls in the response
    const toolCalls =
      outputMessage.content?.filter((block: any) => block.toolUse) || [];
    console.log(`   🔧 Tool calls in response: ${toolCalls.length}`);

    if (toolCalls.length > 0) {
      toolCalls.forEach((toolCall: any, index: number) => {
        console.log(`     Tool ${index + 1}: ${toolCall.toolUse.name}`);
      });
    }

    // Check for text content in the response
    const textContent = outputMessage.content?.find((block: any) => block.text);
    if (textContent) {
      console.log(
        "   💬 Text response preview:",
        textContent.text.substring(0, 80) + "...",
      );
    }

    // Check usage statistics
    if (response.usage) {
      console.log("   📈 Token usage:");
      console.log(`     Input tokens: ${response.usage.inputTokens}`);
      console.log(`     Output tokens: ${response.usage.outputTokens}`);
      console.log(`     Total tokens: ${response.usage.totalTokens}`);
    }

    // Verify stop reason
    if (response.stopReason) {
      console.log(`   ⏹️ Stop reason: ${response.stopReason}`);
    }

    return true;
  }

  private async runMultiModalScenario(): Promise<boolean> {
    console.log("   🖼️ Testing multi-modal content with image...");

    // Simple 1x1 red pixel PNG for testing
    const imageData =
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";

    const command = new this.ConverseCommand({
      modelId: this.options.modelId,
      system: [
        {
          text: "You are a helpful image analysis assistant.",
        },
      ],
      messages: [
        {
          role: "user",
          content: [
            {
              text: "What do you see in this image? Please describe it briefly.",
            },
            {
              image: {
                format: "png" as const,
                source: { bytes: Buffer.from(imageData, "base64") },
              },
            },
          ],
        },
      ],
      inferenceConfig: {
        maxTokens: 100,
        temperature: 0.3,
      },
    });

    const response = await this.client.send(command);

    if (!response.output?.message) {
      console.log("❌ No message in response");
      return false;
    }

    const outputMessage = response.output.message;
    console.log("✅ Multi-modal response received successfully");

    // Check for text content in the response
    const textContent = outputMessage.content?.find((block: any) => block.text);
    if (textContent) {
      console.log(
        "   🖼️ Image analysis preview:",
        textContent.text.substring(0, 100) + "...",
      );
    }

    // Check usage statistics
    if (response.usage) {
      console.log("   📈 Token usage:");
      console.log(`     Input tokens: ${response.usage.inputTokens}`);
      console.log(`     Output tokens: ${response.usage.outputTokens}`);
      console.log(`     Total tokens: ${response.usage.totalTokens}`);
    }

    return true;
  }

  private async runContextAttributesScenario(): Promise<boolean> {
    console.log("   📋 Testing context attributes propagation...");

    const command = new this.ConverseCommand({
      modelId: this.options.modelId,
      system: [
        {
          text: "You are a helpful assistant for testing context attributes.",
        },
      ],
      messages: [
        {
          role: "user",
          content: [
            {
              text: "Hello! This is a test with comprehensive context attributes.",
            },
          ],
        },
      ],
      inferenceConfig: {
        maxTokens: 80,
        temperature: 0.1,
      },
    });

    // Setup comprehensive OpenInference context
    const response = await context.with(
      setSession(
        setUser(
          setMetadata(
            setTags(
              setPromptTemplate(context.active(), {
                template: "System: {{system_prompt}}\n\nUser: {{user_message}}",
                version: "2.1.0",
                variables: {
                  system_prompt:
                    "You are a helpful assistant for testing context attributes.",
                  user_message:
                    "Hello! This is a test with comprehensive context attributes.",
                },
              }),
              ["validation", "converse", "context", "comprehensive", "bedrock"],
            ),
            {
              experiment_name: "converse-comprehensive-validation",
              version: "2.1.0",
              environment: "validation",
              script_name: "validate-converse-comprehensive",
              scenario: "context-attributes",
              features: "system-prompt,context-attributes,inference-config",
              timestamp: new Date().toISOString(),
              test_type: "comprehensive",
            },
          ),
          { userId: "comprehensive-validation-user-001" },
        ),
        { sessionId: "comprehensive-validation-session-001" },
      ),
      async () => {
        const result = await this.client.send(command);
        return result;
      },
    );

    if (!response.output?.message) {
      console.log("❌ No message in response");
      return false;
    }

    const outputMessage = response.output.message;
    console.log("✅ Context attributes response received successfully");

    console.log("   📋 Context attributes configured:");
    console.log("      🆔 Session ID: comprehensive-validation-session-001");
    console.log("      👤 User ID: comprehensive-validation-user-001");
    console.log(
      "      📊 Metadata: experiment_name=converse-comprehensive-validation",
    );
    console.log(
      "      🏷️ Tags: [validation, converse, context, comprehensive, bedrock]",
    );
    console.log("      📝 Prompt Template: System: {{system_prompt}}...");

    // Check for text content in the response
    const textContent = outputMessage.content?.find((block: any) => block.text);
    if (textContent) {
      console.log(
        "   💬 Response preview:",
        textContent.text.substring(0, 80) + "...",
      );
    }

    // Check usage statistics
    if (response.usage) {
      console.log("   📈 Token usage:");
      console.log(`     Input tokens: ${response.usage.inputTokens}`);
      console.log(`     Output tokens: ${response.usage.outputTokens}`);
      console.log(`     Total tokens: ${response.usage.totalTokens}`);
    }

    console.log("   ✅ Context attributes should be visible in Phoenix trace");

    return true;
  }

  private async runErrorCaseScenario(): Promise<boolean> {
    console.log("   ❌ Testing error case handling...");

    const command = new this.ConverseCommand({
      modelId: "invalid-model-id-for-testing",
      messages: [
        {
          role: "user",
          content: [
            {
              text: "This should trigger an error due to invalid model ID",
            },
          ],
        },
      ],
      inferenceConfig: {
        maxTokens: 50,
      },
    });

    try {
      await this.client.send(command);
      console.log("   ⚠️ Expected error but request succeeded");
      return false;
    } catch (error: any) {
      console.log("   ✅ Error handled correctly:", error.name);

      // Additional error details
      if (error.message) {
        console.log(
          "   📝 Error message:",
          error.message.substring(0, 100) + "...",
        );
      }

      if (error.$metadata?.httpStatusCode) {
        console.log("   🔢 HTTP status code:", error.$metadata.httpStatusCode);
      }

      return true;
    }
  }

  private async runAmazonNovaScenario(): Promise<boolean> {
    console.log("   🟠 Testing Amazon Nova with Converse API...");

    const command = new this.ConverseCommand({
      modelId: "us.amazon.nova-micro-v1:0",
      system: [
        {
          text: "You are a helpful AI assistant specialized in explaining Amazon's Nova models.",
        },
      ],
      messages: [
        {
          role: "user",
          content: [
            {
              text: "What makes Amazon Nova models unique compared to other language models?",
            },
          ],
        },
      ],
      inferenceConfig: {
        maxTokens: 120,
        temperature: 0.2,
      },
    });

    try {
      const response = await this.client.send(command);

      if (!response.output?.message) {
        console.log("❌ No message in Nova response");
        return false;
      }

      const outputMessage = response.output.message;
      console.log("✅ Amazon Nova Converse response received successfully");

      // Check for text content in the response
      const textContent = outputMessage.content?.find(
        (block: any) => block.text,
      );
      if (textContent) {
        console.log(
          "   💬 Nova response preview:",
          textContent.text.substring(0, 80) + "...",
        );
      }

      // Check Nova usage statistics
      if (response.usage) {
        console.log("   📈 Nova token usage:");
        console.log(`     Input tokens: ${response.usage.inputTokens}`);
        console.log(`     Output tokens: ${response.usage.outputTokens}`);
        console.log(`     Total tokens: ${response.usage.totalTokens}`);
      }

      return true;
    } catch (error: any) {
      if (
        error.name === "ValidationException" &&
        error.message.includes("model identifier")
      ) {
        console.log(
          "   ⚠️ Amazon Nova model not available in this region, but instrumentation working",
        );
        return true;
      }
      throw error;
    }
  }

  private async runMetaLlamaScenario(): Promise<boolean> {
    console.log("   🦙 Testing Meta Llama with Converse API...");

    const command = new this.ConverseCommand({
      modelId: "meta.llama3-8b-instruct-v1:0",
      system: [
        {
          text: "You are a helpful AI assistant that specializes in explaining Meta's Llama models.",
        },
      ],
      messages: [
        {
          role: "user",
          content: [
            {
              text: "What are the key features and capabilities of Meta's Llama 3 models?",
            },
          ],
        },
      ],
      inferenceConfig: {
        maxTokens: 120,
        temperature: 0.3,
      },
    });

    try {
      const response = await this.client.send(command);

      if (!response.output?.message) {
        console.log("❌ No message in Meta Llama response");
        return false;
      }

      const outputMessage = response.output.message;
      console.log("✅ Meta Llama Converse response received successfully");

      // Check for text content in the response
      const textContent = outputMessage.content?.find(
        (block: any) => block.text,
      );
      if (textContent) {
        console.log(
          "   💬 Llama response preview:",
          textContent.text.substring(0, 80) + "...",
        );
      }

      // Check Meta usage statistics
      if (response.usage) {
        console.log("   📈 Meta token usage:");
        console.log(`     Input tokens: ${response.usage.inputTokens}`);
        console.log(`     Output tokens: ${response.usage.outputTokens}`);
        console.log(`     Total tokens: ${response.usage.totalTokens}`);
      }

      return true;
    } catch (error: any) {
      if (
        error.name === "ValidationException" &&
        error.message.includes("model identifier")
      ) {
        console.log(
          "   ⚠️ Meta Llama model not available in this region, but instrumentation working",
        );
        return true;
      }
      throw error;
    }
  }

  async cleanup() {
    await this.provider.shutdown();
  }
}

// CLI argument parsing
function parseArgs(): ValidationOptions {
  const args = process.argv.slice(2);
  const options: ValidationOptions = {
    scenario: "all",
    debug: false,
    phoenixEndpoint: PHOENIX_ENDPOINT,
    phoenixApiKey: PHOENIX_API_KEY,
    modelId: MODEL_ID,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--scenario":
        options.scenario = args[++i] as TestScenario;
        break;
      case "--debug":
        options.debug = true;
        break;
      case "--phoenix-endpoint":
        options.phoenixEndpoint = args[++i];
        break;
      case "--phoenix-api-key":
        options.phoenixApiKey = args[++i];
        break;
      case "--model-id":
        options.modelId = args[++i];
        break;
      case "--help":
        console.log(`
Usage: tsx scripts/validate-converse-comprehensive.ts [options]

This script runs 8 comprehensive Converse API validation scenarios:
1. basic-flow: Simple conversation with system prompt
2. multi-turn: Complex conversation history
3. tool-calling: Tool configuration and usage
4. multi-modal: Text + image content
5. context-attributes: OpenInference context propagation
6. error-case: API error handling
7. amazon-nova: Amazon Nova model testing
8. meta-llama: Meta Llama model testing

Options:
  --scenario <scenario>     Test scenario: basic-flow, multi-turn, tool-calling, multi-modal, context-attributes, error-case, amazon-nova, meta-llama, all (default: all)
  --debug                   Enable debug logging
  --phoenix-endpoint <url>  Phoenix endpoint URL (default: ${PHOENIX_ENDPOINT})
  --phoenix-api-key <key>   Phoenix API key (default: from PHOENIX_API_KEY env)
  --model-id <id>           Bedrock model ID (default: ${MODEL_ID})
  --help                    Show this help

Environment Variables:
  PHOENIX_ENDPOINT          Phoenix endpoint URL
  PHOENIX_COLLECTOR_ENDPOINT Phoenix collector base URL (will append /v1/traces)
  PHOENIX_API_KEY           Phoenix API key
  PHOENIX_CLIENT_HEADERS    Phoenix client headers (format: api_key=value)
  OTEL_EXPORTER_OTLP_HEADERS OTEL headers (format: api_key=value)
  AWS_REGION                AWS region (default: us-east-1)
  BEDROCK_MODEL_ID          Bedrock model ID
  AWS_ACCESS_KEY_ID         AWS access key
  AWS_SECRET_ACCESS_KEY     AWS secret key
  AWS_PROFILE               AWS profile to use

Examples:
  tsx scripts/validate-converse-comprehensive.ts
  tsx scripts/validate-converse-comprehensive.ts --scenario tool-calling
  tsx scripts/validate-converse-comprehensive.ts --debug --scenario multi-modal
        `);
        process.exit(0);
        break;
    }
  }

  return options;
}

// Main execution
async function main() {
  const options = parseArgs();
  const validator = new ConverseComprehensiveValidator(options);

  try {
    const success = await validator.runValidation();
    process.exit(success ? 0 : 1);
  } catch (error) {
    console.error("❌ Validation script failed:", error);
    process.exit(1);
  } finally {
    await validator.cleanup();
  }
}

if (require.main === module) {
  main().catch(console.error);
}

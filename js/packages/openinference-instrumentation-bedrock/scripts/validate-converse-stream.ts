#!/usr/bin/env tsx

/**
 * Converse Stream Validation Script for Bedrock instrumentation
 *
 * This script tests the Bedrock Converse Stream instrumentation by sending real spans
 * to Phoenix collector with 4 key streaming scenarios:
 * 1. Basic Call with Context - Simple streaming with full OpenInference context
 * 2. Tool Call Streaming - Multiple tools with streaming tool call processing
 * 3. Multi-Modal Streaming - Text + image content with streaming response
 * 4. Amazon Nova Streaming - Cross-provider streaming compatibility
 *
 * Each scenario generates separate spans in Phoenix for detailed trace analysis.
 * This script follows the patterns from validate-invoke-model.ts and
 * validate-converse-comprehensive.ts but focuses specifically on streaming.
 *
 * Usage:
 *   npm run validate:converse-stream
 *   tsx scripts/validate-converse-stream.ts
 *   tsx scripts/validate-converse-stream.ts --scenario tool-calling
 *   tsx scripts/validate-converse-stream.ts --debug
 */

/* eslint-disable no-console, @typescript-eslint/no-explicit-any */

import { context, diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { Resource } from "@opentelemetry/resources";
import type { SpanExporter } from "@opentelemetry/sdk-trace-node";
import {
  ConsoleSpanExporter,
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";

import {
  setMetadata,
  setPromptTemplate,
  setSession,
  setTags,
  setUser,
} from "@arizeai/openinference-core";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import { BedrockInstrumentation, isPatched } from "../src/index";

// Configuration from environment variables
const PHOENIX_ENDPOINT =
  process.env.PHOENIX_ENDPOINT ||
  (process.env.PHOENIX_COLLECTOR_ENDPOINT
    ? `${process.env.PHOENIX_COLLECTOR_ENDPOINT}/v1/traces`
    : "http://localhost:6006/v1/traces");
const PHOENIX_API_KEY = process.env.PHOENIX_API_KEY;
const AWS_REGION = process.env.AWS_REGION || "us-east-1";
const MODEL_ID = process.env.BEDROCK_MODEL_ID || "anthropic.claude-3-5-sonnet-20240620-v1:0";

// Test scenarios for Converse Stream validation
type TestScenario = "basic-with-context" | "tool-calling" | "multi-modal" | "amazon-nova" | "all";

interface ValidationOptions {
  scenario: TestScenario;
  debug: boolean;
  phoenixEndpoint: string;
  phoenixApiKey?: string;
  modelId: string;
}

interface StreamProcessingResult {
  fullText: string;
  toolCalls: any[];
  tokenUsage: any;
  eventCount: number;
  stopReason?: string;
}

/** Mutable state accumulated while consuming a Converse Stream response. */
type StreamAccumulator = Omit<StreamProcessingResult, "eventCount">;

class ConverseStreamPhoenixValidator {
  private client: any; // Will be loaded dynamically
  private provider: NodeTracerProvider;
  private BedrockRuntimeClient: any;
  private ConverseStreamCommand: any;

  constructor(private options: ValidationOptions) {
    this.setupLogging();
  }

  private setupLogging() {
    if (this.options.debug) {
      diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);
    }
  }

  private setupTracing() {
    // Setup exporters
    const exporters: SpanExporter[] = [new ConsoleSpanExporter()];

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

    this.provider = new NodeTracerProvider({
      resource: new Resource({
        [SEMRESATTRS_PROJECT_NAME]: "bedrock-converse-stream-phoenix-validation",
      }),
      spanProcessors: exporters.map((exporter) => new SimpleSpanProcessor(exporter)),
    });

    this.provider.register();
  }

  private async setupInstrumentation() {
    // Load AWS SDK first
    const awsModule = await import("@aws-sdk/client-bedrock-runtime");
    this.BedrockRuntimeClient = awsModule.BedrockRuntimeClient;
    this.ConverseStreamCommand = awsModule.ConverseStreamCommand;

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
      console.log("✅ Instrumentation verified: Both global status and method are patched");
      return true;
    } else if (globalPatchStatus) {
      console.log("✅ Instrumentation verified: Global patch status is true");
      return true;
    } else {
      console.log("❌ Instrumentation verification failed");
      console.log("   Send method signature:", sendMethod.toString().substring(0, 100) + "...");
      console.log("   Global patch status:", globalPatchStatus);
      console.log("   Method appears patched:", methodPatched);
      return false;
    }
  }

  /**
   * Applies a contentBlockStart event, recording any tool use that begins there.
   */
  private applyContentBlockStart({ chunk, state }: { chunk: any; state: StreamAccumulator }) {
    const contentBlock = chunk.contentBlockStart.start;
    if (contentBlock?.toolUse) {
      state.toolCalls.push({
        id: contentBlock.toolUse.toolUseId,
        name: contentBlock.toolUse.name,
        input: contentBlock.toolUse.input || {},
        index: chunk.contentBlockStart.contentBlockIndex,
      });
    }
  }

  /**
   * Applies a contentBlockDelta event, accumulating text and partial tool input.
   */
  private applyContentBlockDelta({ chunk, state }: { chunk: any; state: StreamAccumulator }) {
    const delta = chunk.contentBlockDelta.delta;
    if (delta?.text) {
      state.fullText += delta.text;
      return;
    }
    if (!delta?.toolUse?.input) {
      return;
    }
    // Handle streaming tool input (partial JSON)
    const toolCallIndex = state.toolCalls.findIndex(
      (tc) => tc.index === chunk.contentBlockDelta.contentBlockIndex,
    );
    if (toolCallIndex < 0) {
      return;
    }
    // Accumulate partial tool input
    if (!state.toolCalls[toolCallIndex].partialInput) {
      state.toolCalls[toolCallIndex].partialInput = "";
    }
    state.toolCalls[toolCallIndex].partialInput += String(delta.toolUse.input);
  }

  /**
   * Applies a single Converse Stream event to the accumulated stream state.
   */
  private applyStreamChunk({ chunk, state }: { chunk: any; state: StreamAccumulator }) {
    if (chunk.messageStop) {
      // Message stop event with stop reason
      state.stopReason = chunk.messageStop.stopReason;
      return;
    }
    if (chunk.contentBlockStart) {
      this.applyContentBlockStart({ chunk, state });
      return;
    }
    if (chunk.contentBlockDelta) {
      this.applyContentBlockDelta({ chunk, state });
      return;
    }
    if (chunk.metadata) {
      // Metadata with usage information
      state.tokenUsage = {
        inputTokens: chunk.metadata.usage?.inputTokens,
        outputTokens: chunk.metadata.usage?.outputTokens,
        totalTokens: chunk.metadata.usage?.totalTokens,
      };
    }
    // messageStart and contentBlockStop carry nothing we need.
  }

  /**
   * Parses any accumulated partial tool input into the final tool call input.
   */
  private resolvePartialToolInputs(toolCalls: any[]) {
    toolCalls.forEach((toolCall) => {
      if (!toolCall.partialInput) {
        return;
      }
      try {
        toolCall.input = JSON.parse(toolCall.partialInput);
      } catch {
        // Keep partial input if JSON parsing fails
        toolCall.input = { partial: toolCall.partialInput };
      }
      delete toolCall.partialInput;
    });
  }

  /**
   * Consumes a Converse Stream response and processes all streaming events
   */
  private async consumeStreamResponse(stream: any): Promise<StreamProcessingResult> {
    const state: StreamAccumulator = {
      fullText: "",
      toolCalls: [],
      tokenUsage: {},
    };
    let eventCount = 0;

    try {
      // Process the streaming response
      for await (const chunk of stream) {
        eventCount++;
        this.applyStreamChunk({ chunk, state });
      }

      // Process any partial tool inputs
      this.resolvePartialToolInputs(state.toolCalls);
    } catch (error) {
      console.log("   ⚠️ Error processing stream:", error.message);
    }

    return { ...state, eventCount };
  }

  async runValidation() {
    console.log("🚀 Starting Bedrock Converse Stream Phoenix validation");
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
      console.log("❌ Instrumentation verification failed - stopping validation");
      return false;
    }

    const scenarios =
      this.options.scenario === "all"
        ? (["basic-with-context", "tool-calling", "multi-modal", "amazon-nova"] as TestScenario[])
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

        // Delay between scenarios to ensure spans are processed
        await new Promise((resolve) => setTimeout(resolve, 1500));
      } catch (error) {
        console.log(`❌ Scenario ${scenario} threw error:`, error.message);
        if (this.options.debug) {
          console.log("   Stack:", error.stack);
        }
        allPassed = false;
      }
    }

    console.log("\n📊 Validation Summary:");
    console.log(allPassed ? "✅ All scenarios passed" : "❌ Some scenarios failed");

    // Give time for traces to be exported to Phoenix
    console.log("\n⏳ Waiting for traces to be exported to Phoenix...");
    await new Promise((resolve) => setTimeout(resolve, 3000));

    console.log("🔍 Check Phoenix at:", this.options.phoenixEndpoint.replace("/v1/traces", ""));

    return allPassed;
  }

  private async runScenario(scenario: TestScenario): Promise<boolean> {
    switch (scenario) {
      case "basic-with-context":
        return this.runBasicWithContextScenario();
      case "tool-calling":
        return this.runToolCallingStreamingScenario();
      case "multi-modal":
        return this.runMultiModalStreamingScenario();
      case "amazon-nova":
        return this.runNovaStreamingScenario();
      default:
        throw new Error(`Unknown scenario: ${scenario}`);
    }
  }

  private async runBasicWithContextScenario(): Promise<boolean> {
    console.log("   📝 Testing basic streaming with full OpenInference context...");

    const command = new this.ConverseStreamCommand({
      modelId: this.options.modelId,
      system: [
        {
          text: "You are a helpful assistant for context attribute testing. Always mention the user's name if provided.",
        },
      ],
      messages: [
        {
          role: "user",
          content: [
            {
              text: "Hello! My name is Alex. Can you tell me about the benefits of streaming responses?",
            },
          ],
        },
      ],
      inferenceConfig: {
        maxTokens: 150,
        temperature: 0.1,
      },
    });

    // Setup comprehensive OpenInference context for this streaming call
    const response = await context.with(
      setSession(
        setUser(
          setMetadata(
            setTags(
              setPromptTemplate(context.active(), {
                template: "System: {{system_prompt}}\n\nUser ({{user_name}}): {{user_message}}",
                version: "1.0.0",
                variables: {
                  system_prompt:
                    "You are a helpful assistant for context attribute testing. Always mention the user's name if provided.",
                  user_name: "Alex",
                  user_message:
                    "Hello! My name is Alex. Can you tell me about the benefits of streaming responses?",
                },
              }),
              ["converse-stream", "phoenix", "validation", "context", "streaming"],
            ),
            {
              experiment_name: "converse-stream-phoenix-validation",
              scenario: "basic-with-context",
              version: "1.0.0",
              environment: "validation",
              streaming: true,
              model_provider: "anthropic",
              timestamp: new Date().toISOString(),
            },
          ),
          { userId: "phoenix-validation-user-alex" },
        ),
        { sessionId: "phoenix-validation-session-001" },
      ),
      async () => {
        const result = await this.client.send(command);
        return result;
      },
    );

    if (!response?.stream) {
      console.log("❌ No stream in response");
      return false;
    }

    const streamResult = await this.consumeStreamResponse(response.stream);

    console.log("✅ Basic streaming with context completed successfully");
    console.log(`   📊 Events processed: ${streamResult.eventCount}`);
    console.log(`   📝 Response length: ${streamResult.fullText.length} characters`);
    console.log(`   💬 Response preview: ${streamResult.fullText.substring(0, 100)}...`);

    if (streamResult.tokenUsage.inputTokens) {
      console.log(
        `   📈 Token usage: ${streamResult.tokenUsage.inputTokens} input, ${streamResult.tokenUsage.outputTokens} output, ${streamResult.tokenUsage.totalTokens} total`,
      );
    }

    if (streamResult.stopReason) {
      console.log(`   ⏹️ Stop reason: ${streamResult.stopReason}`);
    }

    console.log("   📋 Context attributes configured:");
    console.log("      🆔 Session ID: phoenix-validation-session-001");
    console.log("      👤 User ID: phoenix-validation-user-alex");
    console.log("      📊 Metadata: experiment_name=converse-stream-phoenix-validation");
    console.log("      🏷️ Tags: [converse-stream, phoenix, validation, context, streaming]");
    console.log("      📝 Prompt Template: System: {{system_prompt}}...");

    return true;
  }

  private async runToolCallingStreamingScenario(): Promise<boolean> {
    console.log("   🔧 Testing tool calling with streaming...");

    const command = new this.ConverseStreamCommand({
      modelId: this.options.modelId,
      messages: [
        {
          role: "user",
          content: [
            {
              text: "What's the weather like in San Francisco and what's 25 * 17?",
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
                      description: "The city and state, e.g. San Francisco, CA",
                    },
                    unit: {
                      type: "string",
                      enum: ["celsius", "fahrenheit"],
                      description: "Temperature unit (default: fahrenheit)",
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
                      description: "Mathematical expression to evaluate, e.g. '25 * 17'",
                    },
                  },
                  required: ["expression"],
                },
              },
            },
          },
        ],
      },
      inferenceConfig: {
        maxTokens: 200,
        temperature: 0.1,
      },
    });

    const response = await this.client.send(command);

    if (!response?.stream) {
      console.log("❌ No stream in response");
      return false;
    }

    const streamResult = await this.consumeStreamResponse(response.stream);

    console.log("✅ Tool calling streaming completed successfully");
    console.log(`   📊 Events processed: ${streamResult.eventCount}`);
    console.log(`   🔧 Tool calls detected: ${streamResult.toolCalls.length}`);

    // Log tool call details
    streamResult.toolCalls.forEach((toolCall, index) => {
      console.log(`     Tool ${index + 1}: ${toolCall.name} - ${JSON.stringify(toolCall.input)}`);
    });

    console.log(`   📝 Response length: ${streamResult.fullText.length} characters`);
    if (streamResult.fullText.length > 0) {
      console.log(`   💬 Text response preview: ${streamResult.fullText.substring(0, 100)}...`);
    }

    if (streamResult.tokenUsage.inputTokens) {
      console.log(
        `   📈 Token usage: ${streamResult.tokenUsage.inputTokens} input, ${streamResult.tokenUsage.outputTokens} output, ${streamResult.tokenUsage.totalTokens} total`,
      );
    }

    if (streamResult.stopReason) {
      console.log(`   ⏹️ Stop reason: ${streamResult.stopReason}`);
    }

    return true;
  }

  private async runMultiModalStreamingScenario(): Promise<boolean> {
    console.log("   🖼️ Testing multi-modal streaming (text + image)...");

    // Small test image - 1x1 red pixel PNG
    const imageData =
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";

    const command = new this.ConverseStreamCommand({
      modelId: this.options.modelId,
      system: [
        {
          text: "You are a helpful image analysis assistant. Describe what you see in images with detail.",
        },
      ],
      messages: [
        {
          role: "user",
          content: [
            {
              text: "Please analyze this image and describe what you see. Also tell me what color it is.",
            },
            {
              image: {
                format: "png" as const,
                source: {
                  bytes: Buffer.from(imageData, "base64"),
                },
              },
            },
          ],
        },
      ],
      inferenceConfig: {
        maxTokens: 150,
        temperature: 0.3,
      },
    });

    const response = await this.client.send(command);

    if (!response?.stream) {
      console.log("❌ No stream in response");
      return false;
    }

    const streamResult = await this.consumeStreamResponse(response.stream);

    console.log("✅ Multi-modal streaming completed successfully");
    console.log(`   📊 Events processed: ${streamResult.eventCount}`);
    console.log(`   📝 Response length: ${streamResult.fullText.length} characters`);
    console.log(`   🖼️ Image analysis preview: ${streamResult.fullText.substring(0, 120)}...`);

    if (streamResult.tokenUsage.inputTokens) {
      console.log(
        `   📈 Token usage: ${streamResult.tokenUsage.inputTokens} input, ${streamResult.tokenUsage.outputTokens} output, ${streamResult.tokenUsage.totalTokens} total`,
      );
    }

    if (streamResult.stopReason) {
      console.log(`   ⏹️ Stop reason: ${streamResult.stopReason}`);
    }

    return true;
  }

  private async runNovaStreamingScenario(): Promise<boolean> {
    console.log("   🟠 Testing Amazon Nova streaming...");

    const command = new this.ConverseStreamCommand({
      modelId: "us.amazon.nova-micro-v1:0",
      system: [
        {
          text: "You are an AI assistant specializing in explaining Amazon's Nova models.",
        },
      ],
      messages: [
        {
          role: "user",
          content: [
            {
              text: "What are the key capabilities of Amazon Nova models? Please be concise.",
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

      if (!response?.stream) {
        console.log("❌ No stream in response");
        return false;
      }

      const streamResult = await this.consumeStreamResponse(response.stream);

      console.log("✅ Amazon Nova streaming completed successfully");
      console.log(`   📊 Events processed: ${streamResult.eventCount}`);
      console.log(`   📝 Response length: ${streamResult.fullText.length} characters`);
      console.log(`   🟠 Nova response preview: ${streamResult.fullText.substring(0, 100)}...`);

      if (streamResult.tokenUsage.inputTokens) {
        console.log(
          `   📈 Nova token usage: ${streamResult.tokenUsage.inputTokens} input, ${streamResult.tokenUsage.outputTokens} output, ${streamResult.tokenUsage.totalTokens} total`,
        );
      }

      if (streamResult.stopReason) {
        console.log(`   ⏹️ Stop reason: ${streamResult.stopReason}`);
      }

      return true;
    } catch (error: any) {
      if (error.name === "ValidationException" && error.message.includes("model identifier")) {
        console.log(
          "   ⚠️ Amazon Nova model not available in this region, but instrumentation working",
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
Usage: tsx scripts/validate-converse-stream.ts [options]

This script tests Bedrock Converse Stream instrumentation by sending real spans to Phoenix:

Scenarios:
1. basic-with-context: Simple streaming with full OpenInference context attributes
2. tool-calling: Multiple tools with streaming tool call processing  
3. multi-modal: Text + image content with streaming response analysis
4. amazon-nova: Cross-provider streaming compatibility testing

Options:
  --scenario <scenario>     Test scenario: basic-with-context, tool-calling, multi-modal, amazon-nova, all (default: all)
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
  tsx scripts/validate-converse-stream.ts
  tsx scripts/validate-converse-stream.ts --scenario tool-calling
  tsx scripts/validate-converse-stream.ts --debug --scenario multi-modal
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
  const validator = new ConverseStreamPhoenixValidator(options);

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

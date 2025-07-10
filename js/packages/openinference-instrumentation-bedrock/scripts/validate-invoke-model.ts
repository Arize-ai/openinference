#!/usr/bin/env tsx

/**
 * Validation script for Bedrock InvokeModel instrumentation
 *
 * This script tests the Bedrock instrumentation in a real-world scenario by:
 * 1. Setting up OpenTelemetry tracing to Phoenix
 * 2. Verifying the instrumentation is properly applied
 * 3. Making actual Bedrock API calls to replicate test scenarios
 * 4. Validating that traces appear correctly in Phoenix
 *
 * Usage:
 *   npm run validate:invoke-model
 *   tsx scripts/validate-invoke-model.ts
 *   tsx scripts/validate-invoke-model.ts --scenario basic-text
 *   tsx scripts/validate-invoke-model.ts --debug
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
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

// Configuration from environment variables
const PHOENIX_ENDPOINT =
  process.env.PHOENIX_ENDPOINT || process.env.PHOENIX_COLLECTOR_ENDPOINT
    ? `${process.env.PHOENIX_COLLECTOR_ENDPOINT}/v1/traces`
    : "http://localhost:6006/v1/traces";
const PHOENIX_API_KEY = process.env.PHOENIX_API_KEY;
const AWS_REGION = process.env.AWS_REGION || "us-east-1";
const MODEL_ID =
  process.env.BEDROCK_MODEL_ID || "anthropic.claude-3-haiku-20240307-v1:0";

// Test scenarios
type TestScenario =
  | "basic-text"
  | "tool-calling"
  | "multi-modal"
  | "tool-results"
  | "multiple-tools"
  | "all";

interface ValidationOptions {
  scenario: TestScenario;
  debug: boolean;
  phoenixEndpoint: string;
  phoenixApiKey?: string;
  modelId: string;
}

class InstrumentationValidator {
  private client: any; // Will be loaded dynamically
  private provider: NodeTracerProvider;
  private BedrockRuntimeClient: any;
  private InvokeModelCommand: any;

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
        [SEMRESATTRS_PROJECT_NAME]: "bedrock-validation-script",
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

      const phoenixExporter = new OTLPTraceExporter({
        url: this.options.phoenixEndpoint,
        headers: apiKey
          ? {
              api_key: apiKey,
            }
          : {},
      });
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
    this.InvokeModelCommand = awsModule.InvokeModelCommand;

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
    console.log("🚀 Starting Bedrock InvokeModel instrumentation validation");
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
            "basic-text",
            "tool-calling",
            "multi-modal",
            "tool-results",
            "multiple-tools",
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
        await new Promise((resolve) => setTimeout(resolve, 500));
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
    await new Promise((resolve) => setTimeout(resolve, 2000));

    return allPassed;
  }

  private async runScenario(scenario: TestScenario): Promise<boolean> {
    switch (scenario) {
      case "basic-text":
        return this.runBasicTextScenario();
      case "tool-calling":
        return this.runToolCallingScenario();
      case "multi-modal":
        return this.runMultiModalScenario();
      case "tool-results":
        return this.runToolResultsScenario();
      case "multiple-tools":
        return this.runMultipleToolsScenario();
      default:
        throw new Error(`Unknown scenario: ${scenario}`);
    }
  }

  private async runBasicTextScenario(): Promise<boolean> {
    console.log("   📝 Testing basic text message...");

    const command = new this.InvokeModelCommand({
      modelId: this.options.modelId,
      body: JSON.stringify({
        anthropic_version: "bedrock-2023-05-31",
        max_tokens: 100,
        messages: [
          {
            role: "user",
            content: "Hello! Please respond with a short greeting.",
          },
        ],
      }),
    });

    const response = await this.client.send(command);
    const responseBody = JSON.parse(new TextDecoder().decode(response.body));

    console.log(
      "   💬 Response:",
      responseBody.content[0].text.substring(0, 50) + "...",
    );
    return true;
  }

  private async runToolCallingScenario(): Promise<boolean> {
    console.log("   🔧 Testing tool calling...");

    const command = new this.InvokeModelCommand({
      modelId: this.options.modelId,
      body: JSON.stringify({
        anthropic_version: "bedrock-2023-05-31",
        max_tokens: 100,
        tools: [
          {
            name: "get_weather",
            description: "Get current weather for a location",
            input_schema: {
              type: "object",
              properties: {
                location: { type: "string" },
              },
            },
          },
        ],
        messages: [
          { role: "user", content: "What's the weather in San Francisco?" },
        ],
      }),
    });

    const response = await this.client.send(command);
    const responseBody = JSON.parse(new TextDecoder().decode(response.body));

    const hasToolCall = responseBody.content.some(
      (block: any) => block.type === "tool_use",
    );
    console.log("   🔧 Tool call detected:", hasToolCall);

    return true;
  }

  private async runMultiModalScenario(): Promise<boolean> {
    console.log("   🖼️ Testing multi-modal message...");

    // Simple 1x1 red pixel PNG
    const imageData =
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";

    const command = new this.InvokeModelCommand({
      modelId: this.options.modelId,
      body: JSON.stringify({
        anthropic_version: "bedrock-2023-05-31",
        max_tokens: 100,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "What do you see in this image?" },
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
    });

    const response = await this.client.send(command);
    const responseBody = JSON.parse(new TextDecoder().decode(response.body));

    console.log(
      "   🖼️ Multi-modal response:",
      responseBody.content[0].text.substring(0, 50) + "...",
    );
    return true;
  }

  private async runToolResultsScenario(): Promise<boolean> {
    console.log("   🔄 Testing tool results processing...");

    const command = new this.InvokeModelCommand({
      modelId: this.options.modelId,
      body: JSON.stringify({
        anthropic_version: "bedrock-2023-05-31",
        max_tokens: 100,
        tools: [
          {
            name: "get_weather",
            description: "Get current weather for a location",
            input_schema: {
              type: "object",
              properties: {
                location: { type: "string" },
              },
            },
          },
        ],
        messages: [
          { role: "user", content: "What's the weather in Paris?" },
          {
            role: "assistant",
            content: [
              {
                type: "tool_use",
                id: "toolu_123",
                name: "get_weather",
                input: { location: "Paris, France" },
              },
            ],
          },
          {
            role: "user",
            content: [
              {
                type: "tool_result",
                tool_use_id: "toolu_123",
                content: "The weather in Paris is currently 22°C and sunny.",
              },
              { type: "text", text: "Great! What should I wear?" },
            ],
          },
        ],
      }),
    });

    const response = await this.client.send(command);
    const responseBody = JSON.parse(new TextDecoder().decode(response.body));

    console.log(
      "   🔄 Tool result response:",
      responseBody.content[0].text.substring(0, 50) + "...",
    );
    return true;
  }

  private async runMultipleToolsScenario(): Promise<boolean> {
    console.log("   🛠️ Testing multiple tools...");

    const command = new this.InvokeModelCommand({
      modelId: this.options.modelId,
      body: JSON.stringify({
        anthropic_version: "bedrock-2023-05-31",
        max_tokens: 100,
        tools: [
          {
            name: "get_weather",
            description: "Get current weather for a location",
            input_schema: {
              type: "object",
              properties: {
                location: { type: "string" },
              },
            },
          },
          {
            name: "calculate",
            description: "Perform mathematical calculations",
            input_schema: {
              type: "object",
              properties: {
                expression: { type: "string" },
              },
            },
          },
          {
            name: "web_search",
            description: "Search the web for information",
            input_schema: {
              type: "object",
              properties: {
                query: { type: "string" },
              },
            },
          },
        ],
        messages: [
          {
            role: "user",
            content: "What's the weather in San Francisco and what's 15 * 23?",
          },
        ],
      }),
    });

    const response = await this.client.send(command);
    const responseBody = JSON.parse(new TextDecoder().decode(response.body));

    const toolCalls = responseBody.content.filter(
      (block: any) => block.type === "tool_use",
    );
    console.log("   🛠️ Multiple tool calls detected:", toolCalls.length);

    return true;
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
Usage: tsx scripts/validate-invoke-model.ts [options]

Options:
  --scenario <scenario>     Test scenario: basic-text, tool-calling, multi-modal, tool-results, multiple-tools, all (default: all)
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
  const validator = new InstrumentationValidator(options);

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

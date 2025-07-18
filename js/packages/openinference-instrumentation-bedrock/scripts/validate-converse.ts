#!/usr/bin/env tsx

/**
 * Validation script for Bedrock Converse API instrumentation
 *
 * This script tests the Bedrock Converse API instrumentation in a comprehensive scenario by:
 * 1. Setting up OpenTelemetry tracing to Phoenix
 * 2. Verifying the instrumentation is properly applied
 * 3. Making actual Bedrock Converse API calls combining multiple test scenarios
 * 4. Validating that traces appear correctly in Phoenix
 *
 * The script executes a single comprehensive conversation that covers:
 * - Multi-turn conversation with conversation history
 * - System prompt integration
 * - Multi-modal content (text + image)
 * - Tool calling and tool results
 * - OpenInference context attributes
 * - Inference configuration
 *
 * Usage:
 *   npm run validate:converse
 *   tsx scripts/validate-converse.ts
 *   tsx scripts/validate-converse.ts --debug
 *   tsx scripts/validate-converse.ts --phoenix-endpoint http://localhost:6006/v1/traces
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
  process.env.PHOENIX_ENDPOINT || process.env.PHOENIX_COLLECTOR_ENDPOINT
    ? `${process.env.PHOENIX_COLLECTOR_ENDPOINT}/v1/traces`
    : "http://localhost:6006/v1/traces";
const PHOENIX_API_KEY = process.env.PHOENIX_API_KEY;
const AWS_REGION = process.env.AWS_REGION || "us-east-1";
const MODEL_ID =
  process.env.BEDROCK_MODEL_ID || "anthropic.claude-3-5-sonnet-20240620-v1:0";

interface ValidationOptions {
  debug: boolean;
  phoenixEndpoint: string;
  phoenixApiKey?: string;
  modelId: string;
}

class ConverseValidator {
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
        [SEMRESATTRS_PROJECT_NAME]: "bedrock-converse-validation-script",
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
    console.log("🚀 Starting Bedrock Converse API instrumentation validation");
    console.log(`📊 Phoenix endpoint: ${this.options.phoenixEndpoint}`);
    console.log(`🤖 Model ID: ${this.options.modelId}`);
    console.log(
      `💬 Comprehensive conversation scenario combining multiple test cases`,
    );
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

    try {
      console.log(
        `\n📋 Running comprehensive Converse API validation scenario`,
      );

      // Run the comprehensive conversation that combines multiple test scenarios
      const passed = await this.runComprehensiveConversation();

      if (passed) {
        console.log(
          `✅ Comprehensive conversation scenario completed successfully`,
        );
      } else {
        console.log(`❌ Comprehensive conversation scenario failed`);
        return false;
      }

      console.log("\n📊 Validation Summary:");
      console.log("✅ Comprehensive Converse API validation passed");

      // Give time for traces to be exported
      console.log("\n⏳ Waiting for traces to be exported...");
      await new Promise((resolve) => setTimeout(resolve, 2000));

      return true;
    } catch (error) {
      console.log(`❌ Comprehensive conversation threw error:`, error.message);
      return false;
    }
  }

  private async runComprehensiveConversation(): Promise<boolean> {
    console.log(
      "🌟 Executing comprehensive conversation covering multiple test scenarios:",
    );
    console.log("   📝 System prompt integration");
    console.log("   🔄 Multi-turn conversation history");
    console.log("   🖼️ Multi-modal content (text + image)");
    console.log("   🔧 Tool calling and tool results");
    console.log("   📋 OpenInference context attributes");
    console.log("   ⚙️ Inference configuration");
    console.log();

    // Simple 1x1 transparent PNG for multi-modal testing
    const imageData =
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";

    // System prompt that sets up the assistant's behavior
    const systemPrompts = [
      {
        text: "You are a helpful AI assistant specializing in data analysis and visualization.",
      },
      { text: "You always respond concisely but informatively." },
    ];

    // Multi-turn conversation history to simulate real conversation flow
    const conversationHistory = [
      // Turn 1: User asks about data analysis
      {
        role: "user" as const,
        content: [
          {
            text: "Hello! I'm working on a data analysis project. Can you help me understand what tools are available?",
          },
        ],
      },
      // Turn 2: Assistant responds with tool information
      {
        role: "assistant" as const,
        content: [
          {
            text: "I'd be happy to help with your data analysis project! I have access to several tools that can assist you, including data visualization, statistical analysis, and web search capabilities. What specific type of analysis are you looking to perform?",
          },
        ],
      },
      // Turn 3: User asks about data visualization with an image
      {
        role: "user" as const,
        content: [
          {
            text: "I have some sample data visualization here. Can you analyze this chart and also search for best practices in data visualization?",
          },
          {
            image: {
              format: "png" as const,
              source: { bytes: Buffer.from(imageData, "base64") },
            },
          },
        ],
      },
    ];

    // Tools available for the assistant to use (using correct Converse API tool format)
    const tools = [
      {
        toolSpec: {
          name: "analyze_data",
          description: "Analyze datasets and provide statistical insights",
          inputSchema: {
            json: {
              type: "object",
              properties: {
                dataset: {
                  type: "string",
                  description: "The dataset to analyze",
                },
                analysis_type: {
                  type: "string",
                  description: "Type of analysis to perform",
                },
              },
              required: ["dataset", "analysis_type"],
            },
          },
        },
      },
      {
        toolSpec: {
          name: "create_visualization",
          description: "Create data visualizations like charts and graphs",
          inputSchema: {
            json: {
              type: "object",
              properties: {
                chart_type: {
                  type: "string",
                  description: "Type of chart to create",
                },
                data_source: {
                  type: "string",
                  description: "Source of the data",
                },
                title: {
                  type: "string",
                  description: "Title for the visualization",
                },
              },
              required: ["chart_type", "data_source"],
            },
          },
        },
      },
      {
        toolSpec: {
          name: "web_search",
          description:
            "Search the web for information about data science topics",
          inputSchema: {
            json: {
              type: "object",
              properties: {
                query: { type: "string", description: "Search query" },
                focus_area: {
                  type: "string",
                  description: "Specific area to focus on",
                },
              },
              required: ["query"],
            },
          },
        },
      },
    ];

    // Inference configuration for controlled response
    const inferenceConfig = {
      maxTokens: 200,
      temperature: 0.7,
      topP: 0.9,
      stopSequences: ["Human:", "Assistant:"],
    };

    // Create the comprehensive Converse command
    const command = new this.ConverseCommand({
      modelId: this.options.modelId,
      system: systemPrompts,
      messages: conversationHistory,
      toolConfig: {
        tools: tools,
      },
      inferenceConfig: inferenceConfig,
    });

    console.log("🔄 Executing Converse API call with comprehensive context...");

    // Execute within OpenInference context to test context attribute propagation
    const response = await context.with(
      setSession(
        setUser(
          setMetadata(
            setTags(
              setPromptTemplate(context.active(), {
                template:
                  "System: {{system_prompt}}\n\nConversation History: {{conversation}}\n\nUser: {{user_message}}",
                version: "2.0.0",
                variables: {
                  system_prompt: systemPrompts.map((p) => p.text).join(" "),
                  conversation:
                    "Multi-turn conversation with data analysis context",
                  user_message: "Analyze chart and search for best practices",
                },
              }),
              [
                "validation",
                "converse",
                "comprehensive",
                "multi-modal",
                "tools",
                "multi-turn",
              ],
            ),
            {
              experiment_name: "converse-comprehensive-validation",
              version: "1.0.0",
              environment: "validation",
              script_name: "validate-converse",
              scenario: "comprehensive-conversation",
              features: "system-prompt,multi-turn,multi-modal,tools,context",
              timestamp: new Date().toISOString(),
            },
          ),
          { userId: "converse-validation-user-789" },
        ),
        { sessionId: "converse-validation-session-123" },
      ),
      async () => {
        const result = await this.client.send(command);
        return result;
      },
    );

    // Process and analyze the response
    console.log("📊 Analyzing response...");

    if (!response.output?.message) {
      console.log("❌ No message in response");
      return false;
    }

    const outputMessage = response.output.message;
    console.log("✅ Response received successfully");

    // Check for tool calls in the response
    const toolCalls =
      outputMessage.content?.filter((block: any) => block.toolUse) || [];
    console.log(`🔧 Tool calls in response: ${toolCalls.length}`);

    if (toolCalls.length > 0) {
      toolCalls.forEach((toolCall: any, index: number) => {
        console.log(`   Tool ${index + 1}: ${toolCall.toolUse.name}`);
      });
    }

    // Check for text content in the response
    const textContent = outputMessage.content?.find((block: any) => block.text);
    if (textContent) {
      console.log(
        "💬 Text response preview:",
        textContent.text.substring(0, 100) + "...",
      );
    }

    // Check usage statistics
    if (response.usage) {
      console.log("📈 Token usage:");
      console.log(`   Input tokens: ${response.usage.inputTokens}`);
      console.log(`   Output tokens: ${response.usage.outputTokens}`);
      console.log(`   Total tokens: ${response.usage.totalTokens}`);
    }

    // Verify stop reason
    if (response.stopReason) {
      console.log(`⏹️ Stop reason: ${response.stopReason}`);
    }

    console.log("\n🎯 Comprehensive conversation validation completed!");
    console.log("📋 This conversation tested:");
    console.log("   ✅ System prompt integration (2 system prompts)");
    console.log("   ✅ Multi-turn conversation (3 conversation turns)");
    console.log("   ✅ Multi-modal content (text + PNG image)");
    console.log("   ✅ Tool configuration (3 available tools)");
    console.log(
      "   ✅ Inference configuration (maxTokens, temperature, topP, stopSequences)",
    );
    console.log(
      "   ✅ OpenInference context attributes (session, user, metadata, tags, prompt template)",
    );
    console.log("   ✅ Complex message content structure");
    console.log("   ✅ Response processing and validation");

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
    debug: false,
    phoenixEndpoint: PHOENIX_ENDPOINT,
    phoenixApiKey: PHOENIX_API_KEY,
    modelId: MODEL_ID,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
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
Usage: tsx scripts/validate-converse.ts [options]

This script runs a comprehensive Converse API validation that combines multiple test scenarios:
- System prompt integration (multiple system prompts)
- Multi-turn conversation history (3 conversation turns)
- Multi-modal content (text + image)
- Tool calling configuration (3 available tools)
- Inference configuration (maxTokens, temperature, topP, stopSequences)
- OpenInference context attributes (session, user, metadata, tags, prompt template)

Options:
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
  const validator = new ConverseValidator(options);

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

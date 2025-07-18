#!/usr/bin/env tsx

/**
 * Cross-Platform Bedrock Converse API Instrumentation Comparison Script
 *
 * This script runs identical Bedrock Converse API calls using both JavaScript and Python 
 * instrumentations, captures spans in memory, and performs strict attribute-by-attribute 
 * assertions that the instrumentations produce identical results.
 *
 * The script executes a comprehensive Converse API scenario that covers:
 * - System prompt integration
 * - Multi-turn conversation history
 * - Multi-modal content (text + image)
 * - Tool configuration
 * - Inference configuration
 *
 * Usage:
 *   tsx scripts/compare-converse-instrumentations.ts
 *   tsx scripts/compare-converse-instrumentations.ts --debug
 *   tsx scripts/compare-converse-instrumentations.ts --strict-types
 *   tsx scripts/compare-converse-instrumentations.ts --full-span-output
 */

/* eslint-disable no-console, @typescript-eslint/no-explicit-any */

import { spawn } from "child_process";
import { writeFileSync, unlinkSync } from "fs";
import { tmpdir } from "os";
import { join, dirname } from "path";

import { BedrockInstrumentation, isPatched } from "../src/index";
import {
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import { InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { Resource } from "@opentelemetry/resources";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

// Configuration
const AWS_REGION = process.env.AWS_REGION || "us-east-1";
const MODEL_ID = process.env.BEDROCK_MODEL_ID || "anthropic.claude-3-5-sonnet-20240620-v1:0";

interface TestOptions {
  modelId: string;
  ignoreTiming: boolean;
  ignoreResource: boolean;
  strictTypes: boolean;
  debug: boolean;
  fullSpanOutput: boolean;
  writeSpanFiles: boolean;
}

interface NormalizedSpan {
  name: string;
  kind: string;
  attributes: Record<string, any>;
  status: string;
  events: any[];
  links: any[];
  resource_attributes?: Record<string, any>;
}

interface JSSpanData {
  name: string;
  kind: number;
  attributes: Record<string, any>;
  status: { code: number };
  events: any[];
  links: any[];
}

interface PythonSpanData {
  name: string;
  kind: string;
  attributes: Record<string, any>;
  status: { status_code: string };
  events: any[];
  links: any[];
  resource: { attributes: Record<string, any> };
}

class InstrumentationAsserter {
  private failures: string[] = [];

  constructor(private options: TestOptions) {}

  /**
   * Main assertion method that compares normalized spans
   */
  assert(jsSpan: NormalizedSpan, pythonSpan: NormalizedSpan): void {
    let passedAssertions = 0;
    this.failures = []; // Reset failures

    console.log("🏃 Asserting identical results...");

    // 1. Assert span name is identical
    if (this.assertEqual("span.name", jsSpan.name, pythonSpan.name)) {
      passedAssertions++;
    }

    // 2. Assert span kind is identical
    if (this.assertEqual("span.kind", jsSpan.kind, pythonSpan.kind)) {
      passedAssertions++;
    }

    // 3. Assert span status is identical
    if (this.assertEqual("span.status", jsSpan.status, pythonSpan.status)) {
      passedAssertions++;
    }

    // 4. Assert attribute count is identical
    const jsAttrCount = Object.keys(jsSpan.attributes).length;
    const pythonAttrCount = Object.keys(pythonSpan.attributes).length;
    if (this.assertEqual("attribute count", jsAttrCount, pythonAttrCount)) {
      passedAssertions++;
    }

    // 5. Assert each attribute individually
    const attributeAssertions = this.assertAttributesIdentical(
      jsSpan.attributes,
      pythonSpan.attributes
    );
    passedAssertions += attributeAssertions;

    // 6. Assert events are identical
    if (this.assertArraysIdentical("events", jsSpan.events, pythonSpan.events)) {
      passedAssertions++;
    }

    // 7. Assert links are identical
    if (this.assertArraysIdentical("links", jsSpan.links, pythonSpan.links)) {
      passedAssertions++;
    }

    // Report final results
    if (this.failures.length > 0) {
      console.log(`\n❌ ASSERTION FAILURES DETECTED: ${this.failures.length} total failures`);
      console.log(`📊 Assertions passed: ${passedAssertions}`);
      console.log(`\n💥 DETAILED FAILURES:`);
      this.failures.forEach((failure, index) => {
        console.log(`\n${index + 1}. ${failure}`);
      });
      this.logInstrumentationMismatch();
      process.exit(1);
    } else {
      console.log(`\n🎉 SUCCESS: All attributes identical between JavaScript and Python Converse API instrumentations!`);
      console.log(`📊 Total assertions passed: ${passedAssertions}`);
    }
  }

  /**
   * Assert two values are equal with detailed error reporting
   */
  private assertEqual<T>(attributeName: string, jsValue: T, pythonValue: T): boolean {
    if (!this.deepEqual(jsValue, pythonValue)) {
      this.recordFailure(attributeName, jsValue, pythonValue);
      return false;
    }
    console.log(`✅ ${attributeName}: ${JSON.stringify(jsValue)}`);
    return true;
  }

  /**
   * Assert all attributes are identical between spans
   */
  private assertAttributesIdentical(
    jsAttrs: Record<string, any>,
    pythonAttrs: Record<string, any>
  ): number {
    // Get all unique attribute keys from both spans
    const allKeys = new Set([...Object.keys(jsAttrs), ...Object.keys(pythonAttrs)]);
    let assertions = 0;

    for (const key of Array.from(allKeys).sort()) {
      // Assert key exists in both
      if (!(key in jsAttrs)) {
        this.failures.push(`Attribute "${key}" missing in JavaScript span`);
        continue;
      }
      if (!(key in pythonAttrs)) {
        this.failures.push(`Attribute "${key}" missing in Python span`);
        continue;
      }

      // Assert values are identical
      const jsValue = jsAttrs[key];
      const pythonValue = pythonAttrs[key];

      if (!this.deepEqual(jsValue, pythonValue)) {
        this.recordFailure(key, jsValue, pythonValue);
        continue;
      }

      console.log(`✅ ${key}: ${JSON.stringify(jsValue)}`);
      assertions++;
    }

    return assertions;
  }

  /**
   * Assert arrays are identical
   */
  private assertArraysIdentical(name: string, jsArray: any[], pythonArray: any[]): boolean {
    if (!this.deepEqual(jsArray, pythonArray)) {
      this.recordFailure(name, jsArray, pythonArray);
      return false;
    }
    console.log(`✅ ${name}: ${jsArray.length} items`);
    return true;
  }

  /**
   * Deep equality comparison with optional strict type checking
   */
  private deepEqual(a: any, b: any): boolean {
    if (a === b) return true;

    // Handle null/undefined
    if (a == null || b == null) return a === b;

    // Handle arrays
    if (Array.isArray(a) && Array.isArray(b)) {
      if (a.length !== b.length) return false;
      return a.every((item, index) => this.deepEqual(item, b[index]));
    }

    // Handle objects
    if (typeof a === "object" && typeof b === "object") {
      const keysA = Object.keys(a);
      const keysB = Object.keys(b);
      if (keysA.length !== keysB.length) return false;
      return keysA.every(key => key in b && this.deepEqual(a[key], b[key]));
    }

    // Handle primitive type coercion (unless strict types is enabled)
    if (!this.options.strictTypes) {
      // Allow string/number coercion for common cases
      if ((typeof a === "string" && typeof b === "number") ||
          (typeof a === "number" && typeof b === "string")) {
        return String(a) === String(b);
      }
    }

    return false;
  }

  /**
   * Record assertion failure for later reporting
   */
  private recordFailure(attributeName: string, jsValue: any, pythonValue: any): void {
    const failure = `❌ ${attributeName}\n  JavaScript: ${JSON.stringify(jsValue)}\n  Python:     ${JSON.stringify(pythonValue)}`;
    this.failures.push(failure);
  }

  /**
   * Log instrumentation mismatch message
   */
  private logInstrumentationMismatch(): void {
    console.error(`\n💥 CONVERSE API INSTRUMENTATION MISMATCH DETECTED!`);
    console.error(`🔧 The Converse API instrumentations produce different results and need alignment.`);
  }
}

class ConverseComparisonTester {
  private jsSpanExporter: InMemorySpanExporter;
  private jsProvider: NodeTracerProvider;
  private BedrockRuntimeClient: any;
  private ConverseCommand: any;

  constructor(private options: TestOptions) {
    this.jsSpanExporter = new InMemorySpanExporter();
    this.jsProvider = new NodeTracerProvider({
      resource: new Resource({
        [SEMRESATTRS_PROJECT_NAME]: "bedrock-converse-comparison-test-js",
      }),
    });
  }

  /**
   * Run the complete test suite
   */
  async runTest(): Promise<void> {
    console.log("🧪 Bedrock Converse API Instrumentation Comparison Test");
    console.log(`🤖 Model: ${this.options.modelId}`);
    console.log(`💬 Comprehensive conversation scenario combining multiple test cases`);
    console.log();

    try {
      // Run JavaScript instrumentation
      console.log("🔍 Running JavaScript Converse API instrumentation...");
      const jsSpan = await this.runJavaScriptTest();
      console.log("✅ JavaScript span captured successfully");

      // Run Python instrumentation
      console.log("\n🐍 Running Python Converse API instrumentation...");
      const pythonSpan = await this.runPythonTest();
      console.log("✅ Python span captured successfully");

      // Normalize spans
      const normalizedJsSpan = this.normalizeJavaScriptSpan(jsSpan);
      const normalizedPythonSpan = this.normalizePythonSpan(pythonSpan);

      if (this.options.debug) {
        console.log("\n🔍 DEBUG: Normalized JavaScript Span:");
        console.log(JSON.stringify(normalizedJsSpan, null, 2));
        console.log("\n🔍 DEBUG: Normalized Python Span:");
        console.log(JSON.stringify(normalizedPythonSpan, null, 2));
      }

      if (this.options.fullSpanOutput) {
        console.log("\n📋 FULL SPAN OUTPUT:");
        console.log("=" .repeat(80));
        console.log("🔍 RAW JAVASCRIPT SPAN (serializable parts):");
        console.log(JSON.stringify({
          name: jsSpan.name,
          kind: jsSpan.kind,
          attributes: jsSpan.attributes,
          status: jsSpan.status,
          events: jsSpan.events,
          links: jsSpan.links,
          startTime: jsSpan.startTime,
          endTime: jsSpan.endTime
        }, null, 2));
        console.log("\n🐍 RAW PYTHON SPAN:");
        console.log(JSON.stringify(pythonSpan, null, 2));
        console.log("=" .repeat(80));
      }

      // Write span files if requested
      if (this.options.writeSpanFiles) {
        this.writeSpanFiles(jsSpan, pythonSpan);
      }

      // Run assertions
      console.log();
      const asserter = new InstrumentationAsserter(this.options);
      asserter.assert(normalizedJsSpan, normalizedPythonSpan);

    } catch (error) {
      console.error(`\n❌ Test failed: ${error.message}`);
      process.exit(1);
    } finally {
      await this.cleanup();
    }
  }

  /**
   * Run JavaScript instrumentation test
   */
  private async runJavaScriptTest(): Promise<JSSpanData> {
    // Setup tracing first
    this.setupJavaScriptTracing();
    
    // Setup instrumentation
    await this.setupJavaScriptInstrumentation();

    // Create client after instrumentation setup
    const client = new this.BedrockRuntimeClient({ region: AWS_REGION });

    // Execute comprehensive Converse scenario
    const spans = await this.executeConverseScenario(client);

    if (spans.length === 0) {
      throw new Error("No spans captured from JavaScript instrumentation");
    }

    return spans[0];
  }

  /**
   * Setup JavaScript tracing
   */
  private setupJavaScriptTracing(): void {
    this.jsProvider.addSpanProcessor(new SimpleSpanProcessor(this.jsSpanExporter));
    this.jsProvider.register();
  }

  /**
   * Setup JavaScript instrumentation
   */
  private async setupJavaScriptInstrumentation(): Promise<void> {
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

      if (this.options.debug) {
        console.log("✅ Bedrock instrumentation registered and manually applied");
      }
    } else {
      if (this.options.debug) {
        console.log("✅ Bedrock instrumentation was already registered");
      }
    }

    // Reset span exporter to ensure clean state
    this.jsSpanExporter.reset();
  }

  /**
   * Execute comprehensive Converse scenario
   */
  private async executeConverseScenario(client: any): Promise<JSSpanData[]> {
    // Simple 1x1 transparent PNG for multi-modal testing
    const imageData = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";

    // System prompt that sets up the assistant's behavior
    const systemPrompts = [
      { text: "You are a helpful AI assistant specializing in data analysis." },
      { text: "Always respond concisely and informatively." }
    ];

    // Multi-turn conversation history
    const conversationHistory = [
      {
        role: "user" as const,
        content: [
          { text: "Hello! I need help analyzing data. What tools do you have?" }
        ]
      },
      {
        role: "assistant" as const,
        content: [
          { text: "I can help with data analysis, visualization, and web search. What type of analysis do you need?" }
        ]
      },
      {
        role: "user" as const,
        content: [
          { text: "I have a chart to analyze. Please analyze it and search for best practices." },
          { 
            image: { 
              format: "png" as const,
              source: { bytes: Buffer.from(imageData, "base64") }
            }
          }
        ]
      }
    ];

    // Tools available for the assistant
    const tools = [
      {
        toolSpec: {
          name: "analyze_data",
          description: "Analyze datasets and provide insights",
          inputSchema: {
            json: {
              type: "object",
              properties: {
                dataset: { type: "string", description: "The dataset to analyze" },
                analysis_type: { type: "string", description: "Type of analysis" }
              },
              required: ["dataset", "analysis_type"]
            }
          }
        }
      },
      {
        toolSpec: {
          name: "web_search",
          description: "Search for information",
          inputSchema: {
            json: {
              type: "object",
              properties: {
                query: { type: "string", description: "Search query" }
              },
              required: ["query"]
            }
          }
        }
      }
    ];

    // Inference configuration
    const inferenceConfig = {
      maxTokens: 150,
      temperature: 0.7,
      topP: 0.9,
      stopSequences: ["Human:", "Assistant:"]
    };

    const command = new this.ConverseCommand({
      modelId: this.options.modelId,
      system: systemPrompts,
      messages: conversationHistory,
      toolConfig: {
        tools: tools
      },
      inferenceConfig: inferenceConfig
    });

    await client.send(command);

    // Small delay to ensure spans are processed
    await new Promise(resolve => setTimeout(resolve, 100));

    return this.jsSpanExporter.getFinishedSpans();
  }

  /**
   * Run Python instrumentation test via subprocess
   */
  private async runPythonTest(): Promise<PythonSpanData> {
    const pythonScript = this.generatePythonTestScript();
    const tempFile = join(tmpdir(), `bedrock-converse-test-${Date.now()}.py`);

    try {
      writeFileSync(tempFile, pythonScript);

      const result = await this.executePythonScript(tempFile);
      const pythonSpan = JSON.parse(result);

      if (!pythonSpan) {
        throw new Error("No span data returned from Python script");
      }

      return pythonSpan;
    } finally {
      try {
        unlinkSync(tempFile);
      } catch (e) {
        // Ignore cleanup errors
      }
    }
  }

  /**
   * Generate Python test script
   */
  private generatePythonTestScript(): string {
    return `#!/usr/bin/env python3

import json
import sys
import boto3
import base64
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from openinference.instrumentation.bedrock import BedrockInstrumentor
from openinference.semconv.resource import ResourceAttributes as OIResourceAttributes

# Setup tracing
resource = Resource.create({
    OIResourceAttributes.PROJECT_NAME: "bedrock-converse-comparison-test-python",
})

tracer_provider = trace_sdk.TracerProvider(resource=resource)
span_exporter = InMemorySpanExporter()
span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
tracer_provider.add_span_processor(span_processor=span_processor)
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

# Setup instrumentation
BedrockInstrumentor().instrument()

# Create client
client = boto3.client("bedrock-runtime", region_name="${AWS_REGION}")

try:
    # Same comprehensive scenario as JS version
    image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    # System prompts
    system_prompts = [
        {"text": "You are a helpful AI assistant specializing in data analysis."},
        {"text": "Always respond concisely and informatively."}
    ]
    
    # Multi-turn conversation history
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Hello! I need help analyzing data. What tools do you have?"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"text": "I can help with data analysis, visualization, and web search. What type of analysis do you need?"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"text": "I have a chart to analyze. Please analyze it and search for best practices."},
                {
                    "image": {
                        "format": "png",
                        "source": {
                            "bytes": base64.b64decode(image_data)
                        }
                    }
                }
            ]
        }
    ]
    
    # Tools configuration
    tools = [
        {
            "toolSpec": {
                "name": "analyze_data",
                "description": "Analyze datasets and provide insights",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "dataset": {"type": "string", "description": "The dataset to analyze"},
                            "analysis_type": {"type": "string", "description": "Type of analysis"}
                        },
                        "required": ["dataset", "analysis_type"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "web_search",
                "description": "Search for information",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            }
        }
    ]
    
    # Inference configuration
    inference_config = {
        "maxTokens": 150,
        "temperature": 0.7,
        "topP": 0.9,
        "stopSequences": ["Human:", "Assistant:"]
    }
    
    # Execute Converse API call
    response = client.converse(
        modelId="${this.options.modelId}",
        system=system_prompts,
        messages=messages,
        toolConfig={"tools": tools},
        inferenceConfig=inference_config
    )
    
    # Get spans
    spans = span_exporter.get_finished_spans()
    
    if not spans:
        print("ERROR: No spans captured", file=sys.stderr)
        sys.exit(1)
    
    span = spans[0]
    
    # Convert span to serializable format
    span_data = {
        "name": span.name,
        "kind": str(span.kind),
        "attributes": dict(span.attributes or {}),
        "status": {"status_code": str(span.status.status_code)},
        "events": [
            {
                "name": event.name,
                "timestamp": event.timestamp,
                "attributes": dict(event.attributes or {})
            } for event in span.events
        ],
        "links": [
            {
                "context": {
                    "trace_id": link.context.trace_id,
                    "span_id": link.context.span_id
                },
                "attributes": dict(link.attributes or {})
            } for link in span.links
        ],
        "resource": {
            "attributes": dict(span.resource.attributes or {})
        }
    }
    
    print(json.dumps(span_data))

except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
`;
  }

  /**
   * Execute Python script and return stdout
   */
  private async executePythonScript(scriptPath: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const python = spawn("python", [scriptPath], {
        stdio: ["pipe", "pipe", "pipe"],
        env: { ...process.env }
      });

      let stdout = "";
      let stderr = "";

      python.stdout.on("data", (data) => {
        stdout += data.toString();
      });

      python.stderr.on("data", (data) => {
        stderr += data.toString();
      });

      python.on("close", (code) => {
        if (code !== 0) {
          reject(new Error(`Python script failed with code ${code}: ${stderr}`));
        } else {
          resolve(stdout.trim());
        }
      });

      python.on("error", (error) => {
        reject(new Error(`Failed to start Python process: ${error.message}`));
      });

      // Set timeout
      setTimeout(() => {
        python.kill();
        reject(new Error("Python script timed out"));
      }, 30000);
    });
  }

  /**
   * Normalize JavaScript span data
   */
  private normalizeJavaScriptSpan(span: JSSpanData): NormalizedSpan {
    return {
      name: span.name,
      kind: this.normalizeSpanKind(span.kind),
      attributes: span.attributes || {},
      status: this.normalizeStatus(span.status.code),
      events: span.events || [],
      links: span.links || [],
    };
  }

  /**
   * Normalize Python span data
   */
  private normalizePythonSpan(span: PythonSpanData): NormalizedSpan {
    const normalized: NormalizedSpan = {
      name: span.name,
      kind: this.normalizePythonSpanKind(span.kind),
      attributes: span.attributes || {},
      status: this.normalizePythonStatus(span.status.status_code),
      events: span.events || [],
      links: span.links || [],
    };

    if (!this.options.ignoreResource && span.resource) {
      normalized.resource_attributes = span.resource.attributes;
    }

    return normalized;
  }

  /**
   * Normalize span kind from number to string
   */
  private normalizeSpanKind(kind: number): string {
    const spanKinds = ["INTERNAL", "SERVER", "CLIENT", "PRODUCER", "CONSUMER"];
    return spanKinds[kind] || "INTERNAL";
  }

  /**
   * Normalize status code to string
   */
  private normalizeStatus(code: number): string {
    const statusCodes = ["UNSET", "OK", "ERROR"];
    return statusCodes[code] || "UNSET";
  }

  /**
   * Normalize Python span kind string to match JavaScript format
   */
  private normalizePythonSpanKind(kind: string): string {
    // Convert "SpanKind.INTERNAL" to "INTERNAL"
    if (kind.startsWith("SpanKind.")) {
      return kind.substring(9); // Remove "SpanKind." prefix
    }
    return kind;
  }

  /**
   * Normalize Python status code string to match JavaScript format
   */
  private normalizePythonStatus(status: string): string {
    // Convert "StatusCode.OK" to "OK"
    if (status.startsWith("StatusCode.")) {
      return status.substring(11); // Remove "StatusCode." prefix
    }
    return status;
  }

  /**
   * Write raw spans to separate files
   */
  private writeSpanFiles(jsSpan: JSSpanData, pythonSpan: PythonSpanData): void {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const currentDir = dirname(__filename);
    
    // JavaScript span file
    const jsSpanData = {
      name: jsSpan.name,
      kind: jsSpan.kind,
      attributes: jsSpan.attributes,
      status: jsSpan.status,
      events: jsSpan.events,
      links: jsSpan.links,
      startTime: jsSpan.startTime,
      endTime: jsSpan.endTime
    };
    
    const jsSpanFile = join(currentDir, `converse-js-span-${timestamp}.json`);
    writeFileSync(jsSpanFile, JSON.stringify(jsSpanData, null, 2));
    console.log(`📄 JavaScript span written to: ${jsSpanFile}`);
    
    // Python span file  
    const pythonSpanFile = join(currentDir, `converse-python-span-${timestamp}.json`);
    writeFileSync(pythonSpanFile, JSON.stringify(pythonSpan, null, 2));
    console.log(`📄 Python span written to: ${pythonSpanFile}`);
  }

  /**
   * Cleanup resources
   */
  private async cleanup(): Promise<void> {
    try {
      await this.jsProvider.shutdown();
    } catch (e) {
      // Ignore cleanup errors
    }
  }
}

/**
 * Parse command line arguments
 */
function parseArgs(): TestOptions {
  const args = process.argv.slice(2);
  const options: TestOptions = {
    modelId: MODEL_ID,
    ignoreTiming: true,
    ignoreResource: true,
    strictTypes: false,
    debug: false,
    fullSpanOutput: false,
    writeSpanFiles: false,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--model-id":
        options.modelId = args[++i];
        break;
      case "--strict-types":
        options.strictTypes = true;
        break;
      case "--debug":
        options.debug = true;
        break;
      case "--full-span-output":
        options.fullSpanOutput = true;
        break;
      case "--write-span-files":
        options.writeSpanFiles = true;
        break;
      case "--help":
        printHelp();
        process.exit(0);
        break;
      default:
        if (args[i].startsWith("--")) {
          console.error(`Unknown option: ${args[i]}`);
          process.exit(1);
        }
        break;
    }
  }

  return options;
}

/**
 * Print help information
 */
function printHelp(): void {
  console.log(`
Usage: tsx scripts/compare-converse-instrumentations.ts [options]

This script compares JavaScript and Python Bedrock Converse API instrumentations by:
- Making identical comprehensive Converse API calls with both SDKs
- Capturing spans from both instrumentations
- Performing strict attribute-by-attribute comparison
- Reporting any differences between the instrumentations

Comprehensive Test Scenario:
- System prompt integration (multiple system prompts)
- Multi-turn conversation history (3 conversation turns)
- Multi-modal content (text + image)
- Tool configuration (2 available tools)
- Inference configuration (maxTokens, temperature, topP, stopSequences)

Options:
  --model-id <id>           Bedrock model ID (default: ${MODEL_ID})
  --strict-types            Enable strict type checking (string "123" ≠ number 123)
  --debug                   Enable verbose logging
  --full-span-output        Output complete raw span data for detailed comparison
  --write-span-files        Write full raw spans to separate JSON files in scripts folder
  --help                    Show this help

Environment Variables:
  AWS_REGION                AWS region (default: us-east-1)
  BEDROCK_MODEL_ID          Bedrock model ID
  AWS_ACCESS_KEY_ID         AWS access key
  AWS_SECRET_ACCESS_KEY     AWS secret key
  AWS_PROFILE               AWS profile to use

Prerequisites:
  - Python with openinference-instrumentation-bedrock installed
  - AWS credentials configured
  - Access to Bedrock models

Example:
  tsx scripts/compare-converse-instrumentations.ts
  tsx scripts/compare-converse-instrumentations.ts --debug
  tsx scripts/compare-converse-instrumentations.ts --strict-types
  tsx scripts/compare-converse-instrumentations.ts --full-span-output
  tsx scripts/compare-converse-instrumentations.ts --write-span-files
`);
}

/**
 * Main execution
 */
async function main(): Promise<void> {
  const options = parseArgs();
  const tester = new ConverseComparisonTester(options);

  try {
    await tester.runTest();
  } catch (error) {
    console.error(`\n❌ Script failed: ${error.message}`);
    if (options.debug) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

if (require.main === module) {
  main().catch(console.error);
}
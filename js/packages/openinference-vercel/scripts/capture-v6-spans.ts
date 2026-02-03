/**
 * Script to capture AI SDK v6 telemetry spans for test fixtures
 *
 * This script exercises all relevant AI SDK v6 features with telemetry enabled
 * and captures the raw spans to JSON files for use as test fixtures.
 *
 * Run with: npx tsx scripts/capture-v6-spans.ts
 */

/* eslint-disable no-console */
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";

import { openai } from "@ai-sdk/openai";
import { embed, generateObject, generateText, streamText } from "ai";
import * as fs from "fs";
import * as path from "path";
import { fileURLToPath } from "url";
import { z } from "zod";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Setup OpenTelemetry with in-memory exporter to capture spans
const exporter = new InMemorySpanExporter();
const provider = new BasicTracerProvider();
provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
provider.register();

interface CapturedSpan {
  name: string;
  spanContext: {
    traceId: string;
    spanId: string;
  };
  parentSpanId?: string;
  attributes: Record<string, unknown>;
  status: {
    code: number;
    message?: string;
  };
  startTime: [number, number];
  endTime: [number, number];
  events: Array<{
    name: string;
    time: [number, number];
    attributes?: Record<string, unknown>;
  }>;
}

async function captureGenerateText(): Promise<void> {
  console.log("Capturing generateText spans...");

  const result = await generateText({
    model: openai("gpt-4o-mini"),
    prompt: "Write a haiku about programming.",
    experimental_telemetry: {
      isEnabled: true,
      functionId: "test-generate-text",
      metadata: {
        testCategory: "text-generation",
        customField: "custom-value",
      },
    },
  });

  console.log("generateText result:", result.text);
}

async function captureGenerateTextWithMessages(): Promise<void> {
  console.log("Capturing generateText with messages spans...");

  const result = await generateText({
    model: openai("gpt-4o-mini"),
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "What is 2 + 2?" },
      { role: "assistant", content: "2 + 2 equals 4." },
      { role: "user", content: "And what is 3 + 3?" },
    ],
    experimental_telemetry: {
      isEnabled: true,
      functionId: "test-generate-text-messages",
      metadata: {
        testCategory: "multi-turn-conversation",
      },
    },
  });

  console.log("generateText with messages result:", result.text);
}

async function captureGenerateObject(): Promise<void> {
  console.log("Capturing generateObject spans...");

  const recipeSchema = z.object({
    name: z.string(),
    ingredients: z.array(z.string()),
    steps: z.array(z.string()),
  });

  const result = await generateObject({
    model: openai("gpt-4o-mini"),
    prompt: "Generate a simple recipe for scrambled eggs.",
    schema: recipeSchema,
    experimental_telemetry: {
      isEnabled: true,
      functionId: "test-generate-object",
      metadata: {
        testCategory: "structured-output",
      },
    },
  });

  console.log("generateObject result:", JSON.stringify(result.object, null, 2));
}

async function captureStreamText(): Promise<void> {
  console.log("Capturing streamText spans...");

  const result = streamText({
    model: openai("gpt-4o-mini"),
    prompt: "Write a short poem about TypeScript.",
    experimental_telemetry: {
      isEnabled: true,
      functionId: "test-stream-text",
      metadata: {
        testCategory: "streaming",
      },
    },
  });

  let fullText = "";
  for await (const chunk of result.textStream) {
    fullText += chunk;
  }

  console.log("streamText result:", fullText);
}

async function captureEmbed(): Promise<void> {
  console.log("Capturing embed spans...");

  const result = await embed({
    model: openai.embedding("text-embedding-3-small"),
    value: "The quick brown fox jumps over the lazy dog.",
    experimental_telemetry: {
      isEnabled: true,
      functionId: "test-embed",
      metadata: {
        testCategory: "embeddings",
      },
    },
  });

  console.log(
    "embed result: embedding with",
    result.embedding.length,
    "dimensions",
  );
}

type CaptureFunction = () => Promise<void>;

async function runCapture(
  name: string,
  fn: CaptureFunction,
): Promise<{ name: string; success: boolean; error?: string }> {
  try {
    await fn();
    return { name, success: true };
  } catch (error) {
    console.error(`Error in ${name}:`, error);
    return {
      name,
      success: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

async function captureAllSpans(): Promise<void> {
  const results: Array<{ name: string; success: boolean; error?: string }> = [];

  // Run all capture functions, collecting results
  results.push(await runCapture("generateText", captureGenerateText));
  results.push(
    await runCapture(
      "generateTextWithMessages",
      captureGenerateTextWithMessages,
    ),
  );
  results.push(await runCapture("generateObject", captureGenerateObject));
  results.push(await runCapture("streamText", captureStreamText));
  results.push(await runCapture("embed", captureEmbed));

  // Give spans time to be exported
  await new Promise((resolve) => setTimeout(resolve, 1000));

  // Get all captured spans
  const spans = exporter.getFinishedSpans();

  console.log(`\nCaptured ${spans.length} spans total`);
  console.log(
    "\nCapture results:",
    results.map((r) => `${r.name}: ${r.success ? "✓" : "✗"}`).join(", "),
  );

  // Convert spans to JSON-serializable format
  const capturedSpans: CapturedSpan[] = spans.map((span) => ({
    name: span.name,
    spanContext: {
      traceId: span.spanContext().traceId,
      spanId: span.spanContext().spanId,
    },
    parentSpanId: span.parentSpanId,
    attributes: span.attributes as Record<string, unknown>,
    status: {
      code: span.status.code,
      message: span.status.message,
    },
    startTime: span.startTime,
    endTime: span.endTime,
    events: span.events.map((event) => ({
      name: event.name,
      time: event.time,
      attributes: event.attributes as Record<string, unknown>,
    })),
  }));

  // Group spans by operation type for easier fixture generation
  const spansByOperation: Record<string, CapturedSpan[]> = {};
  for (const span of capturedSpans) {
    const operationName =
      (span.attributes["operation.name"] as string) || span.name;
    const baseOperation = operationName.split(" ")[0]; // Remove functionId suffix
    if (!spansByOperation[baseOperation]) {
      spansByOperation[baseOperation] = [];
    }
    spansByOperation[baseOperation].push(span);
  }

  // Write fixtures directory
  const fixturesDir = path.join(
    __dirname,
    "..",
    "test",
    "__fixtures__",
    "v6-spans",
  );
  fs.mkdirSync(fixturesDir, { recursive: true });

  // Write all spans to a single file
  fs.writeFileSync(
    path.join(fixturesDir, "all-spans.json"),
    JSON.stringify(capturedSpans, null, 2),
  );

  // Write spans grouped by operation
  fs.writeFileSync(
    path.join(fixturesDir, "spans-by-operation.json"),
    JSON.stringify(spansByOperation, null, 2),
  );

  // Write individual operation files
  for (const [operation, opSpans] of Object.entries(spansByOperation)) {
    const filename = operation.replace(/\./g, "-") + ".json";
    fs.writeFileSync(
      path.join(fixturesDir, filename),
      JSON.stringify(opSpans, null, 2),
    );
  }

  console.log(`\nFixtures written to: ${fixturesDir}`);
  console.log("Operations captured:", Object.keys(spansByOperation).join(", "));

  // Print a summary of attributes found
  console.log("\n=== Attribute Summary ===");
  const allAttributes = new Set<string>();
  for (const span of capturedSpans) {
    Object.keys(span.attributes).forEach((key) => allAttributes.add(key));
  }

  const genAiAttrs = Array.from(allAttributes)
    .filter((a) => a.startsWith("gen_ai."))
    .sort();
  const aiAttrs = Array.from(allAttributes)
    .filter((a) => a.startsWith("ai."))
    .sort();
  const otherAttrs = Array.from(allAttributes)
    .filter((a) => !a.startsWith("gen_ai.") && !a.startsWith("ai."))
    .sort();

  console.log("\ngen_ai.* attributes:", genAiAttrs);
  console.log("\nai.* attributes:", aiAttrs);
  console.log("\nOther attributes:", otherAttrs);

  // Write capture results summary
  fs.writeFileSync(
    path.join(fixturesDir, "capture-results.json"),
    JSON.stringify(
      {
        captureDate: new Date().toISOString(),
        results,
        attributeSummary: {
          genAi: genAiAttrs,
          ai: aiAttrs,
          other: otherAttrs,
        },
      },
      null,
      2,
    ),
  );

  await provider.shutdown();
}

// Run the capture
captureAllSpans()
  .then(() => {
    console.log("\nDone!");
    process.exit(0);
  })
  .catch((error) => {
    console.error("Failed:", error);
    process.exit(1);
  });

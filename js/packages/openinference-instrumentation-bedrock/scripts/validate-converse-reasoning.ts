#!/usr/bin/env tsx

/* eslint-disable no-console, @typescript-eslint/no-explicit-any */

import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import {
  ConsoleSpanExporter,
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";

import { BedrockInstrumentation, isPatched } from "../src/index";

const AWS_REGION = process.env.AWS_REGION || "us-east-1";
const MODEL_ID = process.env.BEDROCK_MODEL_ID || "us.anthropic.claude-sonnet-4-6";
const PHOENIX_ENDPOINT = process.env.PHOENIX_ENDPOINT || "http://localhost:6006/v1/traces";

const THINKING_CONFIG = { thinking: { type: "enabled", budget_tokens: 5000 } };
const INFERENCE_CONFIG = { maxTokens: 16000 };

function setupTracing() {
  const provider = new NodeTracerProvider({
    spanProcessors: [
      new SimpleSpanProcessor(new ConsoleSpanExporter()),
      new SimpleSpanProcessor(new OTLPTraceExporter({ url: PHOENIX_ENDPOINT })),
    ],
  });
  provider.register();
  return provider;
}

async function setupInstrumentation() {
  const awsModule = await import("@aws-sdk/client-bedrock-runtime");

  if (!isPatched()) {
    const instrumentation = new BedrockInstrumentation();
    registerInstrumentations({ instrumentations: [instrumentation] });
    (instrumentation as any).patch(
      { BedrockRuntimeClient: awsModule.BedrockRuntimeClient },
      "3.0.0",
    );
  }

  return {
    client: new awsModule.BedrockRuntimeClient({ region: AWS_REGION }),
    ConverseCommand: awsModule.ConverseCommand,
    ConverseStreamCommand: awsModule.ConverseStreamCommand,
  };
}

async function converseNonStreaming(client: any, ConverseCommand: any) {
  console.log("\n--- Non-streaming Converse ---");
  const response = await client.send(
    new ConverseCommand({
      modelId: MODEL_ID,
      messages: [
        {
          role: "user",
          content: [{ text: "What is the 10th Fibonacci number? Think step by step." }],
        },
      ],
      inferenceConfig: INFERENCE_CONFIG,
      additionalModelRequestFields: THINKING_CONFIG,
    }),
  );

  for (const block of response.output?.message?.content ?? []) {
    if ("reasoningContent" in block) {
      const rc = block.reasoningContent as any;
      if (rc.reasoningText)
        console.log("[reasoning]", String(rc.reasoningText.text ?? "").slice(0, 120));
      else if (rc.redactedContent) console.log("[redacted_reasoning]");
    } else if ("text" in block) {
      console.log("[text]", block.text);
    }
  }
}

async function converseStreaming(client: any, ConverseStreamCommand: any) {
  console.log("\n--- Streaming Converse ---");
  const response = await client.send(
    new ConverseStreamCommand({
      modelId: MODEL_ID,
      messages: [
        {
          role: "user",
          content: [{ text: "What is the 10th Fibonacci number? Think step by step." }],
        },
      ],
      inferenceConfig: INFERENCE_CONFIG,
      additionalModelRequestFields: THINKING_CONFIG,
    }),
  );

  let reasoningText = "";
  let outputText = "";

  for await (const event of response.stream) {
    const delta = event.contentBlockDelta?.delta;
    if (!delta) continue;
    if ("reasoningContent" in delta && (delta.reasoningContent as any)?.text) {
      reasoningText += (delta.reasoningContent as any).text;
    } else if ("text" in delta && delta.text) {
      outputText += delta.text;
    }
  }

  if (reasoningText) console.log("[reasoning]", reasoningText.slice(0, 120));
  if (outputText) console.log("[text]", outputText);
}

async function converseMultiTurn(client: any, ConverseCommand: any) {
  console.log("\n--- Multi-turn Converse ---");

  const turn1 = await client.send(
    new ConverseCommand({
      modelId: MODEL_ID,
      messages: [
        {
          role: "user",
          content: [{ text: "What is the 10th Fibonacci number? Think step by step." }],
        },
      ],
      inferenceConfig: INFERENCE_CONFIG,
      additionalModelRequestFields: THINKING_CONFIG,
    }),
  );

  const turn1Message = turn1.output?.message;
  if (!turn1Message) throw new Error("No output in turn 1");

  for (const block of turn1Message.content ?? []) {
    if ("reasoningContent" in block) {
      const rc = block.reasoningContent as any;
      if (rc.reasoningText)
        console.log("Turn 1 [reasoning]", String(rc.reasoningText.text ?? "").slice(0, 80));
    } else if ("text" in block) {
      console.log("Turn 1 [text]", String(block.text).slice(0, 120));
    }
  }

  const turn2 = await client.send(
    new ConverseCommand({
      modelId: MODEL_ID,
      messages: [
        {
          role: "user",
          content: [{ text: "What is the 10th Fibonacci number? Think step by step." }],
        },
        { role: "assistant", content: turn1Message.content },
        { role: "user", content: [{ text: "Great! Now what is the 20th Fibonacci number?" }] },
      ],
      inferenceConfig: INFERENCE_CONFIG,
      additionalModelRequestFields: THINKING_CONFIG,
    }),
  );

  for (const block of turn2.output?.message?.content ?? []) {
    if ("reasoningContent" in block) {
      const rc = block.reasoningContent as any;
      if (rc.reasoningText)
        console.log("Turn 2 [reasoning]", String(rc.reasoningText.text ?? "").slice(0, 80));
    } else if ("text" in block) {
      console.log("Turn 2 [text]", String(block.text).slice(0, 120));
    }
  }
}

async function main() {
  const provider = setupTracing();
  const { client, ConverseCommand, ConverseStreamCommand } = await setupInstrumentation();

  try {
    await converseNonStreaming(client, ConverseCommand);
    await converseStreaming(client, ConverseStreamCommand);
    await converseMultiTurn(client, ConverseCommand);
  } finally {
    await provider.shutdown();
  }
}

if (require.main === module) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}

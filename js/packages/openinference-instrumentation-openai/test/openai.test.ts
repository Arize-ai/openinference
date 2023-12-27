import { OpenAIInstrumentation } from "../src";
import {
    InMemorySpanExporter,
    SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";
import { SemanticResourceAttributes } from "@opentelemetry/semantic-conventions";
const tracerProvider = new NodeTracerProvider();
tracerProvider.register();
const memoryExporter = new InMemorySpanExporter();
const tracer = tracerProvider.getTracer("default");
const resource = new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: "test-instrumentation-openai",
});

const instrumentation = new OpenAIInstrumentation();
instrumentation.disable();

import * as OpenAI from "openai";

describe("OpenAIInstrumentation", () => {
    let openai: OpenAI.OpenAI;

    const memoryExporter = new InMemorySpanExporter();
    const provider = new NodeTracerProvider();
    const tracer = provider.getTracer("default");

    instrumentation.setTracerProvider(tracerProvider);
    tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));

    beforeEach(() => {
        // @ts-expect-error the moduleExports property is private. This is needed to make the test work with auto-mocking
        instrumentation._modules[0].moduleExports = OpenAI;
        instrumentation.enable();
        openai = new OpenAI.OpenAI({
            apiKey: process.env.OPENAI_API_KEY,
        });
    });
    it("is patched", () => {
        expect((OpenAI as any).openInferencePatched).toBe(true);
    });
    it("creates a span for chat completions", async () => {
        const chatCompletion = await openai.chat.completions.create({
            messages: [{ role: "user", content: "Say this is a test" }],
            model: "gpt-3.5-turbo",
        });
        expect(memoryExporter.getFinishedSpans().length).toBe(1);
    });
});

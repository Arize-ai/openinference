import {
    InMemorySpanExporter,
    SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { OpenAIInstrumentation } from "../src/instrumentation";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";
import { SemanticResourceAttributes } from "@opentelemetry/semantic-conventions";
import { isWrapped } from "@opentelemetry/instrumentation";
const tracerProvider = new NodeTracerProvider();
tracerProvider.register();
const memoryExporter = new InMemorySpanExporter();
tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
const tracer = tracerProvider.getTracer("default");
const resource = new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: "test-instrumentation-openai",
});

const instrumentation = new OpenAIInstrumentation();
const OpenAI = jest.requireActual("openai");

describe("OpenAIInstrumentation", () => {
    let openai: typeof OpenAI;
    beforeAll(() => {
        openai = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY,
        });
    });
    it("is instrumented", () => {
        expect(isWrapped(OpenAI.Chat.Completions.prototype)).toBeTruthy();
    });
    it("creates a span for chat completions", async () => {
        const chatCompletion = await openai.chat.completions.create({
            messages: [{ role: "user", content: "Say this is a test" }],
            model: "gpt-3.5-turbo",
        });
        expect(memoryExporter.getFinishedSpans().length).toBe(1);
    });
});

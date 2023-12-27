import {
    InMemorySpanExporter,
    SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";
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

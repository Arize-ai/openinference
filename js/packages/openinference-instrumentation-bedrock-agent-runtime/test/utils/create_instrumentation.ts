import {BedrockAgentInstrumentation} from "../src";
import {InMemorySpanExporter, SimpleSpanProcessor,} from "@opentelemetry/sdk-trace-base";
import {NodeTracerProvider} from "@opentelemetry/sdk-trace-node";



export function create_instrumentation(){
    const tracerProvider = new NodeTracerProvider();
    tracerProvider.register();
    const instrumentation = new BedrockAgentInstrumentation();
    instrumentation.disable();
    instrumentation.setTracerProvider(tracerProvider);
    const memoryExporter = new InMemorySpanExporter();
    tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
}
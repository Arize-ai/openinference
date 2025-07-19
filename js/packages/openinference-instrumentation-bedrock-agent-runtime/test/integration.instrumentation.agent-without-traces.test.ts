import {BedrockAgentRuntimeClient, InvokeAgentCommand} from "@aws-sdk/client-bedrock-agent-runtime";
import {createPolly} from './utils/polly.config';
import {Polly} from "@pollyjs/core";
import {InMemorySpanExporter, SimpleSpanProcessor} from "@opentelemetry/sdk-trace-base";

import {NodeTracerProvider} from "@opentelemetry/sdk-trace-node";
import {BedrockAgentInstrumentation} from "../src";


describe("BedrockAgentInstrumentation Integration - agent attributes and API recording", () => {
    let instrumentation: BedrockAgentInstrumentation;
    let provider: NodeTracerProvider;
    let memoryExporter: InMemorySpanExporter;

    const region = "ap-south-1";

    const cassettePrefix = "bedrock-agent-without-traces";

    let polly: Polly;

    beforeAll(() => {
        // Setup instrumentation and tracer provider (following OpenAI pattern)
        instrumentation = new BedrockAgentInstrumentation();
        instrumentation.disable();
        memoryExporter = new InMemorySpanExporter();
        provider = new NodeTracerProvider();
        provider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
        provider.register();
        instrumentation.setTracerProvider(provider);
        // Manually set module exports for testing (following OpenAI pattern)
        (instrumentation as any)._modules[0].moduleExports = require("@aws-sdk/client-bedrock-agent-runtime");
        // Enable instrumentation ONCE
        instrumentation.enable();
    });

    beforeEach(() => {
        memoryExporter.reset();
        polly = createPolly(cassettePrefix);
    });

    afterEach(async () => {
        await polly.stop();
    });

    // Global cleanup
    afterAll(async () => {
        instrumentation.disable();
        await provider.shutdown();
    });

    it("should record agent attributes and API response in span", async () => {
        // instrumentation = create_instrumentation(memoryExporter);
        const client = new BedrockAgentRuntimeClient({region});
        const params = {
            inputText: "What is the current price of Microsoft?",
            agentId: "3EL4X42BSO",
            agentAliasId: "LSIZXQMZDN",
            sessionId: "default-session1_1234567890",
            enableTrace: false,
        };
        const command = new InvokeAgentCommand(params);
        const response = await client.send(command);
        for await (const event of response.completion as any) {
            const outputText = Buffer.from(event.chunk.bytes).toString("utf8");
            console.log("Chunk output text:", outputText);
        }
        expect(response).toBeDefined();
        expect(typeof response).toBe("object");
        const spans = memoryExporter.getFinishedSpans();
        expect(spans.length).toBe(1);
        const span = spans[0];
        expect(span.name).toBe("bedrock.invoke_agent");
    });
});

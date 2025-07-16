import {BedrockAgentInstrumentation, isPatched} from "../src";

import {ConsoleSpanExporter, NodeTracerProvider, SimpleSpanProcessor,} from "@opentelemetry/sdk-trace-node";
import {Resource} from "@opentelemetry/resources";
import {OTLPTraceExporter} from "@opentelemetry/exporter-trace-otlp-proto";
import {SEMRESATTRS_PROJECT_NAME} from "@arizeai/openinference-semantic-conventions";
import {registerInstrumentations} from "@opentelemetry/instrumentation";


class InstrumentationValidator {
    private provider: NodeTracerProvider;
    private BedrockAgentRuntimeClient: any;
    private InvokeAgentCommand: any;

    async runValidation() {
        console.log("Starting Bedrock Agent InvokeModel instrumentation validation");
        // Setup tracing and instrumentation
        this.setupTracing();
        console.log("Setup tracing complete");

        await this.setupInstrumentation();
        console.log("Setup Instrumentation instrumentation validation");

        // Verify instrumentation is applied
        if (!this.verifyInstrumentation()) {
            console.log("❌ Instrumentation verification failed - stopping validation");
            return false;
        }
        console.log("Running Basic Agent");

        await this.runBasicAgent();
    }

    private setupTracing() {
        this.provider = new NodeTracerProvider({
            resource: new Resource({
                [SEMRESATTRS_PROJECT_NAME]: "bedrock-agents",
            }),
        });
        console.log("Starting 111");
        const exporters = [new ConsoleSpanExporter()];
        const phoenixExporter = new OTLPTraceExporter({
            url: "http://localhost:6006/v1/traces",
        });
        exporters.push(phoenixExporter);
        exporters.forEach((exporter) => {
            this.provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
        });
        console.log("Starting 222");
        this.provider.register();
        console.log("Starting 333");
    }

    private async setupInstrumentation() {
        // Load AWS SDK first
        const awsModule = await import("@aws-sdk/client-bedrock-agent-runtime");
        this.BedrockAgentRuntimeClient = awsModule.BedrockAgentRuntimeClient;
        this.InvokeAgentCommand = awsModule.InvokeAgentCommand;

        // Check if already patched
        if (!isPatched()) {
            // Create instrumentation and register it
            const instrumentation = new BedrockAgentInstrumentation();
            registerInstrumentations({
                instrumentations: [instrumentation],
            });

            // Also manually patch the already-loaded module to ensure it works
            const moduleExports = {
                BedrockAgentRuntimeClient: awsModule.BedrockAgentRuntimeClient,
            };
            (instrumentation as any).patch(moduleExports, "3.0.0");

            console.log("✅ Bedrock agent instrumentation registered and manually applied");
        } else {
            console.log("✅ Bedrock agent instrumentation was already registered");
        }
    }

    private verifyInstrumentation() {
        // Check both the method signature and the global patch status
        const sendMethod = this.BedrockAgentRuntimeClient.prototype.send;
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

    private async runBasicAgent(): Promise<boolean> {
        const agentId = "<AgentID>";
        const agentAliasId = "<AgentAliasID>";
        const sessionId = `default-session1_${Math.floor(Date.now() / 1000)}`;
        const client = new this.BedrockAgentRuntimeClient({region: 'ap-south-1'});
        const params = {
            inputText: "What is the current price of Microsoft?",
            agentId,
            agentAliasId,
            sessionId,
            enableTrace: true,
        };

        try {
            console.log("Invoking agent with parameters:", params);
            const command = new this.InvokeAgentCommand(params);
            const response = await client.send(command);
            if (response.completion) {
                let foundEvent = false;
                for await (const event of response.completion as any) {
                    foundEvent = true;
                    if (event.chunk) {
                        const chunkData = event.chunk;
                        if (chunkData.bytes) {
                            const outputText = Buffer.from(chunkData.bytes).toString("utf8");
                        }
                    } else if (event.trace) {
                        console.log("Trace event:", event.trace.trace);
                    } else {
                        console.log("Other event:", event);
                    }
                }
                if (!foundEvent) {
                    console.log("No events found in completion stream.");
                }
            } else {
                console.log("No completion stream found.");
            }
        } catch (err) {
            console.error("Error invoking agent:", err);
            return 'Error invoking agent: ' + err.message;
        }
    }
}


// Main execution
async function main() {
    try {
        const validator = new InstrumentationValidator();
        await validator.runValidation();
    } catch (error) {
        process.exit(1);
    } finally {
    }
}


if (require.main === module) {
    main().catch(console.error);
}

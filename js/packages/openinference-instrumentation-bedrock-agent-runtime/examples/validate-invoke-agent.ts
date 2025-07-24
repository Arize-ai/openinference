import "./agent-instrumentation";

import {
  BedrockAgentRuntimeClient,
  InvokeAgentCommand,
} from "@aws-sdk/client-bedrock-agent-runtime";
import { diag } from "@opentelemetry/api";

async function run() {
  const agentId = "<AgentId>"; // Replace with your actual agent ID
  const agentAliasId = "<AgentAliasId>"; // Replace with your actual agent alias ID
  const sessionId = `default-session1_${Math.floor(Date.now() / 1000)}`;
  const region = "ap-south-1";

  const client = new BedrockAgentRuntimeClient({ region });

  const params = {
    inputText: "What is the current price of Microsoft?",
    agentId,
    agentAliasId,
    sessionId,
    enableTrace: true,
  };

  try {
    const command = new InvokeAgentCommand(params);
    const response = await client.send(command);
    if (response.completion) {
      let foundEvent = false;
      for await (const event of response.completion as unknown as AsyncIterable<unknown>) {
        foundEvent = true;
        if ((event as { chunk?: { bytes?: Uint8Array } }).chunk) {
          const chunkData = (event as { chunk: { bytes?: Uint8Array } }).chunk;
          if (chunkData.bytes) {
            const outputText = Buffer.from(chunkData.bytes).toString("utf8");
            diag.info("Chunk output text:", outputText);
          }
        }
      }
      if (!foundEvent) {
        diag.error("No events found in completion stream.");
      }
    } else {
      diag.error("No completion stream found.");
    }
  } catch (err) {
    diag.error("Error invoking agent:", err);
  }
}

async function main() {
  try {
    await run();
  } catch (error) {
    process.exit(1);
  }
}

if (require.main === module) {
  // eslint-disable-next-line no-console
  main().catch(console.error);
}

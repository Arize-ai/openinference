import "./agent-instrumentation";

import {
  BedrockAgentRuntimeClient,
  InvokeAgentCommand,
} from "@aws-sdk/client-bedrock-agent-runtime";
import * as console from "node:console";

async function run() {
  const agentId = "3EL4X42BSO";
  const agentAliasId = "LSIZXQMZDN";
  const sessionId = `default-session1_${Math.floor(Date.now() / 1000)}`;
  const region = "ap-south-1";

  const client = new BedrockAgentRuntimeClient({ region });

  const params = {
    inputText: "What is the current price of Microsoft?",
    agentId,
    agentAliasId,
    sessionId,
    enableTrace: false,
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
            // eslint-disable-next-line no-console
            console.debug("Chunk output text:", outputText);
          }
        } else {
          // eslint-disable-next-line no-console
          console.log("Other event:", event);
        }
      }
      if (!foundEvent) {
        // eslint-disable-next-line no-console
        console.error("No events found in completion stream.");
      }
    } else {
      // eslint-disable-next-line no-console
      console.error("No completion stream found.");
    }
  } catch (err) {
    // eslint-disable-next-line no-console
    console.error("Error invoking agent:", err);
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

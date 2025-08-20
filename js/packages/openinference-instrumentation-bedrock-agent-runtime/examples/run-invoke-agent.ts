import "./agent-instrumentation";

import {
  BedrockAgentRuntimeClient,
  InvokeAgentCommand,
} from "@aws-sdk/client-bedrock-agent-runtime";
import { diag } from "@opentelemetry/api";

async function run() {
  const agentId = process.env.BEDROCK_AGENT_ID;
  const agentAliasId = process.env.BEDROCK_AGENT_ALIAS_ID;
  const sessionId = `default-session1_${Math.floor(Date.now() / 1000)}`;
  const region = process.env.AWS_REGION;

  const client = new BedrockAgentRuntimeClient({ region });

  const params = {
    inputText:
      "How do i set up an ec2 instance? here is some code i have started with in the cli, can you interpret it for me? code: \n aws ec2 create --name myec2, i alsohave this javascript function running against the sdk code:\n const newEc2Id = new awsClient().createEc2('my-ec2');",
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

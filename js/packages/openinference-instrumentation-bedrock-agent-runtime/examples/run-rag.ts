import "./agent-instrumentation";

import {
  BedrockAgentRuntimeClient,
  RetrieveAndGenerateCommand,
  RetrieveAndGenerateCommandInput,
  RetrieveAndGenerateStreamCommand,
  RetrieveCommand,
  RetrieveCommandInput,
} from "@aws-sdk/client-bedrock-agent-runtime";
import { diag } from "@opentelemetry/api";
import * as process from "node:process";

const knowledgeBaseId = process.env.KNOWLEDGE_BASE_ID;
const modelArn = process.env.RAG_MODEL_ARN;
const s3Uri = process.env.S3_KNOWLEDGE_BASE_URI;
const s3InputText = process.env.S3_INPUT_TEXT;
const knowledgeBaseInputText = process.env.KNOWLEDGE_BASE_INPUT_TEXT;

async function runRagKnowledgeBase() {
  const region = process.env.AWS_REGION;
  const client = new BedrockAgentRuntimeClient({ region });
  const ragParams: RetrieveAndGenerateCommandInput = {
    input: {
      text: knowledgeBaseInputText,
    },
    retrieveAndGenerateConfiguration: {
      knowledgeBaseConfiguration: {
        knowledgeBaseId: knowledgeBaseId,
        modelArn: modelArn,
      },
      type: "KNOWLEDGE_BASE",
    },
  };
  try {
    const command = new RetrieveAndGenerateCommand(ragParams);
    const response = await client.send(command);
    diag.info("Chunk output text:", response);
  } catch (err) {
    diag.error("Error invoking agent:", err);
  }
}

async function runRagExternalSource() {
  const region = process.env.AWS_REGION;
  const client = new BedrockAgentRuntimeClient({ region });
  const ragParams: RetrieveAndGenerateCommandInput = {
    input: {
      text: s3InputText,
    },
    retrieveAndGenerateConfiguration: {
      type: "EXTERNAL_SOURCES",
      externalSourcesConfiguration: {
        sources: [
          {
            s3Location: {
              uri: s3Uri,
            },
            sourceType: "S3",
          },
        ],
        modelArn: modelArn,
      },
    },
  };

  try {
    const command = new RetrieveAndGenerateCommand(ragParams);
    const response = await client.send(command);
    diag.info("Rag External Sources Response:", response);
  } catch (err) {
    diag.error("Error invoking agent:", err);
  }
}

async function runRagStream() {
  const region = process.env.AWS_REGION;
  const client = new BedrockAgentRuntimeClient({ region });
  const ragParams: RetrieveAndGenerateCommandInput = {
    input: {
      text: knowledgeBaseInputText,
    },
    retrieveAndGenerateConfiguration: {
      knowledgeBaseConfiguration: {
        knowledgeBaseId: knowledgeBaseId,
        modelArn: modelArn,
      },
      type: "KNOWLEDGE_BASE",
    },
  };
  try {
    const command = new RetrieveAndGenerateStreamCommand(ragParams);
    const response = await client.send(command);
    for await (const event of response.stream as unknown as AsyncIterable<unknown>) {
      diag.info("Chunk output text:", event);
    }
  } catch (err) {
    diag.error("Error invoking agent:", err);
  }
}

async function runRetrieve() {
  const region = process.env.AWS_REGION;
  const client = new BedrockAgentRuntimeClient({ region });

  const params: RetrieveCommandInput = {
    retrievalQuery: {
      text: knowledgeBaseInputText,
    },
    knowledgeBaseId: knowledgeBaseId,
  };

  try {
    const command = new RetrieveCommand(params);
    const response = await client.send(command);
    diag.info("Chunk output text:", response);
  } catch (err) {
    diag.error("Error invoking agent:", err);
  }
}

async function main() {
  try {
    await runRagKnowledgeBase();
    await runRagExternalSource();
    await runRagStream();
    await runRetrieve();
  } catch {
    process.exit(1);
  }
}

if (require.main === module) {
  // eslint-disable-next-line no-console
  main().catch(console.error);
}

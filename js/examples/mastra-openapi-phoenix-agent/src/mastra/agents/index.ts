import { openai } from "@ai-sdk/openai";
import { Agent } from "@mastra/core/agent";
import { Memory } from "@mastra/memory";
import { MCPClient } from "@mastra/mcp";
import { storage } from "../storage";

const phoenixMcp = ({
  apiKey,
  apiUrl,
}: {
  apiKey?: string;
  apiUrl?: string;
}) => {
  let extraArgs: string[] = [];
  if (apiKey) {
    extraArgs.push(`--apiKey`, `${apiKey}`);
  }
  if (apiUrl) {
    extraArgs.push(`--baseUrl`, `${apiUrl}`);
  }
  return new MCPClient({
    servers: {
      phoenix: {
        command: "npx",
        args: ["-y", "@arizeai/phoenix-mcp@latest", ...extraArgs],
      },
      context7: {
        command: "npx",
        args: ["-y", "@upstash/context7-mcp@latest"],
      },
    },
    timeout: 30_000,
  });
};

export const phoenixAgent = async ({
  apiKey,
  apiUrl,
}: {
  apiKey?: string;
  apiUrl?: string;
}) => {
  const mcp = phoenixMcp({ apiKey, apiUrl });
  const mcpTools = await mcp.getTools();
  return new Agent({
    name: "Phoenix Agent",
    instructions: `
        You are an expert user of the Phoenix LLM observability platform by Arize AI. Analyze the user's query and provide practical recommendations or actions.

        The github repository for Phoenix is at https://github.com/Arize-ai/phoenix

        Guidelines:
        - Answer all questions about Phoenix citing the documentation
        - Access documentation by using the Context7 tool \`use context7 resolve-library-id /arize-ai/phoenix\` and then \`use context7 get-library-docs <id>\`
        - Documentation is written in markdown, so be sure to filter by the markdown language when searching github for documentation
        - Documentation is located in the top level 'docs' directory
        - Keep descriptions concise but informative
      `,
    model: openai("gpt-4o"),
    tools: { ...mcpTools },
    memory: new Memory({
      storage,
      options: {
        lastMessages: 10,
        semanticRecall: false,
        threads: {
          generateTitle: true,
        },
      },
    }),
  });
};

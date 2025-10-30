import "./instrumentation";

import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { ToolCallingAgent } from "beeai-framework/agents/toolCalling/agent";
import { TokenMemory } from "beeai-framework/memory/tokenMemory";
import { DuckDuckGoSearchTool } from "beeai-framework/tools/search/duckDuckGoSearch";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const llm = new OllamaChatModel("llama3.1");
const agent = new ToolCallingAgent({
  llm,
  memory: new TokenMemory(),
  tools: [
    new DuckDuckGoSearchTool(),
    new OpenMeteoTool(), // weather tool
  ],
});

async function main() {
  const response = await agent.run({ prompt: "How are you?" });
  // eslint-disable-next-line no-console
  console.log(`Agent 🤖 : `, response.result.text);
}

// eslint-disable-next-line no-console
main().catch(console.error);

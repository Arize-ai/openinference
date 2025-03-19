import "./instrumentation";
import { BeeAgent } from "beeai-framework/agents/bee/agent";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";
import { TokenMemory } from "beeai-framework/memory/tokenMemory";
import { DuckDuckGoSearchTool } from "beeai-framework/tools/search/duckDuckGoSearch";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const llm = new OllamaChatModel("llama3.1"); // default is llama3.1 (8B), it is recommended to use 70B model
const agent = new BeeAgent({
  llm, // for more explore 'beeai-framework/adapters'
  memory: new TokenMemory(), // for more explore 'beeai-framework/memory'
  tools: [new DuckDuckGoSearchTool(), new OpenMeteoTool()], // for more explore 'beeai-framework/tools'
});

async function main() {
  const response = await agent.run({
    prompt: "What's the current weather in Las Vegas?",
  });
  // eslint-disable-next-line no-console
  console.log(`Agent 🤖 : `, response.result.text);
}

// eslint-disable-next-line no-console
main().catch(console.error);

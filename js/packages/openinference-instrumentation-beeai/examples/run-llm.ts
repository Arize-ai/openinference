import "./instrumentation";
import { UserMessage } from "beeai-framework/backend/message";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const llm = new OllamaChatModel("llama3.1");
const prompt = "Hello, how are you?";

async function main() {
  const response = await llm.create({
    messages: [new UserMessage(prompt)],
  });
  // eslint-disable-next-line no-console
  console.log(`LLM 🤖 (txt) : `, response.getTextContent());
  // eslint-disable-next-line no-console
  console.log(`LLM 🤖 (raw) : `, JSON.stringify(response.messages));
}

// eslint-disable-next-line no-console
main().catch(console.error);

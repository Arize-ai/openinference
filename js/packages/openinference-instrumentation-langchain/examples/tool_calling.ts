import "./instrumentation";
import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";

const multiply = tool(
  ({ a, b }: { a: number; b: number }): number => {
    /**
     * Multiply a and b.
     */
    return a * b;
  },
  {
    name: "multiply",
    description: "Multiply two numbers",
  },
);

const main = async () => {
  const chatModel = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "gpt-4o-mini",
  });
  const modelWithTools = chatModel.bindTools([multiply]);
  const result = await modelWithTools.invoke([
    new HumanMessage("What is 2 * 3?"),
  ]);

  // eslint-disable-next-line no-console
  console.log(result.content);
  return result;
};

main();

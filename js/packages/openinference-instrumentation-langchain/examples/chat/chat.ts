import "../instrumentation";
import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";

const main = async () => {
  const chatModel = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const response = await chatModel.invoke("test");
  return response;
};

main();

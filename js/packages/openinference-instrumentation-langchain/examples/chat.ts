import "./instrumentation";
import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";

const main = async () => {
  const chatModel = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const response = await chatModel.invoke("Hello! How are you?");

  // eslint-disable-next-line no-console
  console.log(response.content);

  return response;
};

main();

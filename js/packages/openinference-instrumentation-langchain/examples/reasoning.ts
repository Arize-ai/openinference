import "./instrumentation";
import "dotenv/config";

import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const main = async () => {
  const chatModel = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: "o3-mini",
    reasoning: { effort: "medium" },
  });

  const request = new HumanMessage("Hello! How are you?");

  const response = await chatModel.invoke([request]);

  // get a new response, including a greeting in the message history
  const finalResponse = await chatModel.invoke([
    request,
    response,
    new HumanMessage("That is great to hear!"),
  ]);

  // eslint-disable-next-line no-console
  console.log(finalResponse.content);

  return finalResponse;
};

main();

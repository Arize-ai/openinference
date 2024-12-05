import "./instrumentation";
import "dotenv/config";
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";

const main = async () => {
  const chatModel = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    metadata: {
      session_id: "test-session-123",
    },
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

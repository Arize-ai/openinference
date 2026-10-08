import "./instrumentation";
import "dotenv/config";

import { HumanMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

// A 1x1 red PNG, small enough to stay under the default base64 image masking limit.
const RED_PIXEL_PNG =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";

/**
 * Sends a multimodal message (text + images) so the LLM span records each
 * content block under `llm.input_messages.0.message.contents.*` rather than
 * dropping the array content.
 */
const main = async () => {
  const chatModel = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    model: "gpt-4o-mini",
  });

  const request = new HumanMessage({
    content: [
      { type: "text", text: "Describe these two images in one short sentence each." },
      // OpenAI-style image block
      {
        type: "image_url",
        image_url: { url: `data:image/png;base64,${RED_PIXEL_PNG}` },
      },
      // LangChain standard (data) image block
      {
        type: "image",
        source_type: "base64",
        mime_type: "image/png",
        data: RED_PIXEL_PNG,
      },
    ],
  });

  const response = await chatModel.invoke([request]);

  // eslint-disable-next-line no-console
  console.log(response.content);

  return response;
};

main();

import "./instrumentation";
import { isPatched } from "../src";
import OpenAI from "openai";

// Check if OpenAI has been patched
if (!isPatched()) {
  throw new Error("OpenAI instrumentation failed");
}

// Initialize OpenAI
const openai = new OpenAI();

openai.chat.completions
  .create({
    model: "gpt-3.5-turbo",
    messages: [{ role: "system", content: "You are a helpful assistant." }],
    max_tokens: 150,
    temperature: 0.5,
  })
  .then((response) => {
    // eslint-disable-next-line no-console
    console.log(response.choices[0].message.content);
  });

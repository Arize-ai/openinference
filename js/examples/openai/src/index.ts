require("./tracer");

import { OpenAI } from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

(async function () {
  let chatCompletion = await openai.chat.completions.create({
    messages: [{ role: "user", content: "Say this is a test" }],
    model: "gpt-3.5-turbo",
  });
  chatCompletion.choices.forEach((choice) => {
    // eslint-disable-next-line no-console
    console.log(choice.message);
  });
  chatCompletion = await openai.chat.completions.create({
    messages: [{ role: "user", content: "Tell me a joke" }],
    model: "gpt-3.5-turbo",
  });
  chatCompletion.choices.forEach((choice) => {
    // eslint-disable-next-line no-console
    console.log(choice.message);
  });
})();

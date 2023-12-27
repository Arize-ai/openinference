require("./tracer");

import { OpenAI } from "openai";

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

(async function () {
    const chatCompletion = await openai.chat.completions.create({
        messages: [{ role: "user", content: "Say this is a test" }],
        model: "gpt-3.5-turbo",
    });
    chatCompletion.choices.forEach((choice) => {
        console.log(choice.message);
    });
})();

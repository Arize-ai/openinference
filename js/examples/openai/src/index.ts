import { OpenAI } from "openai";
import express from "express";
import bodyParser from "body-parser";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const app = express();
const port = process.env.PORT || 3000;

app.use(bodyParser.json());

app.post("/completion", async (req, res) => {
  const { query } = req.body;
  const chatCompletion = await openai.chat.completions.create({
    messages: [{ role: "user", content: query }],
    model: "gpt-3.5-turbo",
  });
  chatCompletion.choices.forEach((choice) => {
    res.send(choice.message.content);
  });
});

app.listen(port, () => {
  // eslint-disable-next-line no-console
  console.log(`Example app listening on port ${port}`);
});

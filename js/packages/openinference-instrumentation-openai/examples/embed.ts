import "./instrumentation";
import { isPatched } from "../src";
import OpenAI from "openai";

// Check if OpenAI has been patched
if (!isPatched()) {
  throw new Error("OpenAI instrumentation failed");
}

// Initialize OpenAI
const openai = new OpenAI();

openai.embeddings
  .create({
    model: "text-embedding-3-small",
    input: "Hello, world!",
  })
  .then((response) => {
    // eslint-disable-next-line no-console
    console.log(response.data[0].embedding);
  });

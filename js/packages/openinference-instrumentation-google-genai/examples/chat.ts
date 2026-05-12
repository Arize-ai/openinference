import "./instrumentation";

import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY!,
});

ai.models
  .generateContent({
    model: "gemini-2.5-flash",
    contents: "What is the capital of France?",
  })
  .then((response) => {
    // eslint-disable-next-line no-console
    console.log(response.text);
  });

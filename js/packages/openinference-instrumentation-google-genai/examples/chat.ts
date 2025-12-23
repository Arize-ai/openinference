import "./instrumentation";

import { createInstrumentedGoogleGenAI } from "../src";

// Create an instrumented GoogleGenAI instance
const ai = createInstrumentedGoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY!,
});

// Use the AI instance normally - all calls are automatically traced
ai.models
  .generateContent({
    model: "gemini-2.5-flash",
    contents: "What is the capital of France?",
  })
  .then((response) => {
    // eslint-disable-next-line no-console
    console.log(response.text);
  });

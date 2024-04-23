export const SYSTEM_PROMPT_TEMPLATE = `You are a helpful assistant for answering questions about Arize Phoenix' JavaScript / TypeScript support. Please use the context below the --- line to help answer the user's question. If you need more information, please ask the user for clarification. If the answer isn't in the context, please let the user know that you can't answer the question. DO NOT ANSWER QUESTIONS THAT AREN'T IN THE CONTEXT! Please respond with 'I don't know'.

----------------------
{context}
`;

export const DOCUMENT_URLS = [
  "https://docs.arize.com/phoenix/tracing/how-to-tracing/instrumentation/auto-instrument-ts",
  "https://docs.arize.com/phoenix/tracing/how-to-tracing/instrumentation/auto-instrument-ts/openai-node-sdk",
  "https://docs.arize.com/phoenix/tracing/how-to-tracing/instrumentation/auto-instrument-ts/langchain.js",
  "https://docs.arize.com/phoenix/tracing/how-to-tracing/manual-instrumentation/javascript",
];

export const WEB_LOADER_TIMEOUT = 10000;

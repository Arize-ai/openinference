// import "../instrumentation";
import { LangChainInstrumentation } from "../../src";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import "dotenv/config";

const tracerProvider = new NodeTracerProvider();
tracerProvider.register();

const memoryExporter = new InMemorySpanExporter();
const provider = new NodeTracerProvider();
provider.getTracer("default");

const instrumentation = new LangChainInstrumentation();
instrumentation.setTracerProvider(tracerProvider);
tracerProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));

instrumentation.enable();

import { ChatOpenAI } from "@langchain/openai";
const main = async () => {
  const chatModel = new ChatOpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const response = await chatModel.invoke("test");
  return response;
};

main();

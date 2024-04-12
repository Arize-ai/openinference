import { Run } from "@langchain/core/tracers/base";
const baseLangchainMessage = {
  lc_id: ["human"],
  lc_kwargs: {
    content: "hello, this is a test",
  },
};
export const getLangchainMessage = (
  config?: Partial<{
    lc_id: string[];
    lc_kwargs: Record<string, unknown>;
  }>,
) => {
  return Object.assign({ ...baseLangchainMessage }, config);
};

const baseLangchainRun: Run = {
  id: "run_id",
  start_time: 0,
  execution_order: 0,
  child_runs: [],
  child_execution_order: 0,
  events: [],
  name: "test run",
  run_type: "llm",
  inputs: {},
};

export const getLangchainRun = (config?: Partial<Run>) => {
  return Object.assign({ ...baseLangchainRun }, config);
};

export const completionsResponse = {
  id: "chatcmpl-8adq9JloOzNZ9TyuzrKyLpGXexh6p",
  object: "chat.completion",
  created: 1703743645,
  model: "gpt-3.5-turbo-0613",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: "This is a test.",
      },
      logprobs: null,
      finish_reason: "stop",
    },
  ],
  usage: {
    prompt_tokens: 12,
    completion_tokens: 5,
    total_tokens: 17,
  },
};

export const functionCallResponse = {
  id: "chatcmpl-9D6ZQKSVCtEeMT272J8h6xydy1jE2",
  object: "chat.completion",
  created: 1712910548,
  model: "gpt-3.5-turbo-0125",
  choices: [
    {
      index: 0,
      message: {
        role: "assistant",
        content: "",
        function_call: {
          name: "get_current_weather",
          arguments: '{"location":"Seattle, WA","unit":"fahrenheit"}',
        },
      },
      logprobs: null,
      finish_reason: "function_call",
    },
  ],
  usage: {
    prompt_tokens: 88,
    completion_tokens: 22,
    total_tokens: 110,
  },
};

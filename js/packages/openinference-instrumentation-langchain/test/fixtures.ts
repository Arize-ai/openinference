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

export const getLangchainRun = (config?: Partial<Run>): Run => {
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

/***
 *AIMessage 
 {
  "id": "chatcmpl-Bc4uFXiPV92QJpHTTCmA5ZTFDgMJr",
  "content": "",
  "additional_kwargs": {
    "tool_calls": [
      {
        "id": "call_AlQk4zMNaLGogyZRlFGNzioN",
        "type": "function",
        "function": "[Object]"
      }
    ]
  },
  "response_metadata": {
    "tokenUsage": {
      "promptTokens": 48,
      "completionTokens": 16,
      "totalTokens": 64
    },
    "finish_reason": "tool_calls",
    "model_name": "gpt-4o-mini-2024-07-18",
    "usage": {
      "prompt_tokens": 48,
      "completion_tokens": 16,
      "total_tokens": 64,
      "prompt_tokens_details": {
        "cached_tokens": 0,
        "audio_tokens": 0
      },
      "completion_tokens_details": {
        "reasoning_tokens": 0,
        "audio_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0
      }
    },
    "system_fingerprint": "fp_34a54ae93c"
  },
  "tool_calls": [
    {
      "name": "multiply",
      "args": {
        "input": "2 * 3"
      },
      "type": "tool_call",
      "id": "call_AlQk4zMNaLGogyZRlFGNzioN"
    }
  ],
  "invalid_tool_calls": [],
  "usage_metadata": {
    "output_tokens": 16,
    "input_tokens": 48,
    "total_tokens": 64,
    "input_token_details": {
      "audio": 0,
      "cache_read": 0
    },
    "output_token_details": {
      "audio": 0,
      "reasoning": 0
    }
  }
}
 
 * 
 */
export const toolCallResponse = {
  id: "chatcmpl-Bc4uFXiPV92QJpHTTCmA5ZTFDgMJr",
  content: "",
  additional_kwargs: {
    tool_calls: [
      {
        id: "call_AlQk4zMNaLGogyZRlFGNzioN",
        type: "function",
        function: "[Object]",
      },
    ],
  },
  response_metadata: {
    tokenUsage: {
      promptTokens: 48,
      completionTokens: 16,
      totalTokens: 64,
    },
    finish_reason: "tool_calls",
    model_name: "gpt-4o-mini-2024-07-18",
    usage: {
      prompt_tokens: 48,
      completion_tokens: 16,
      total_tokens: 64,
      prompt_tokens_details: {
        cached_tokens: 0,
        audio_tokens: 0,
      },
      completion_tokens_details: {
        reasoning_tokens: 0,
        audio_tokens: 0,
        accepted_prediction_tokens: 0,
        rejected_prediction_tokens: 0,
      },
    },
    system_fingerprint: "fp_34a54ae93c",
  },
  tool_calls: [
    {
      name: "multiply",
      args: {
        input: "2 * 3",
      },
      type: "tool_call",
      id: "call_AlQk4zMNaLGogyZRlFGNzioN",
    },
  ],
  invalid_tool_calls: [],
  usage_metadata: {
    output_tokens: 16,
    input_tokens: 48,
    total_tokens: 64,
    input_token_details: {
      audio: 0,
      cache_read: 0,
    },
    output_token_details: {
      audio: 0,
      reasoning: 0,
    },
  },
};

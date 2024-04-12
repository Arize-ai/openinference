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

import {
  DefaultTraceConfig,
  OPENINFERENCE_HIDE_INPUT_TEXT,
} from "@core/trace/trace_config/constants";
import { generateTraceConfig } from "@core/trace/trace_config/traceConfig";

describe("generateTraceConfig", () => {
  it("should return the default trace config when no options are provided", () => {
    expect(generateTraceConfig()).toEqual(DefaultTraceConfig);
  });

  it("should respect the options provided", () => {
    const options = {
      hideInputText: true,
      hideOutputText: false,
    };
    const expected = {
      ...DefaultTraceConfig,
      hideInputText: true,
      hideOutputText: false,
    };
    expect(generateTraceConfig(options)).toEqual(expected);
  });

  it("should fallback to environment variables if options are not provided", () => {
    process.env[OPENINFERENCE_HIDE_INPUT_TEXT] = "true";
    const expected = {
      ...DefaultTraceConfig,
      hideInputText: true,
      hideInputImages: true,
    };
    expect(generateTraceConfig({ hideInputImages: true })).toEqual(expected);
    delete process.env[OPENINFERENCE_HIDE_INPUT_TEXT];
  });
});

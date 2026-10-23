import { describe, expect, it } from "vitest";

import {
  DefaultTraceConfig,
  OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH,
  OPENINFERENCE_HIDE_INPUT_TEXT,
  OPENINFERENCE_HIDE_INPUTS,
} from "../../src/trace/trace-config/constants";
import { generateTraceConfig } from "../../src/trace/trace-config/traceConfig";

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

  it("should read environment variables when no options are provided", () => {
    process.env[OPENINFERENCE_HIDE_INPUTS] = "true";
    process.env[OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH] = "8000";
    try {
      expect(generateTraceConfig()).toEqual({
        ...DefaultTraceConfig,
        hideInputs: true,
        base64ImageMaxLength: 8000,
      });
    } finally {
      delete process.env[OPENINFERENCE_HIDE_INPUTS];
      delete process.env[OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH];
    }
  });
});

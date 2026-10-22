import { describe, it, expect } from "vitest";

import { extractModelName, getSystemFromModelId } from "../src/attributes/attribute-helpers";

describe("extractModelName", () => {
  // Standard model IDs (vendor.model-version)
  it("extracts Anthropic model name from standard ID", () => {
    expect(extractModelName("anthropic.claude-3-sonnet-20240229-v1:0")).toBe(
      "claude-3-sonnet-20240229",
    );
  });

  it("extracts Anthropic model name without version suffix", () => {
    expect(extractModelName("anthropic.claude-3-haiku-20240307-v1:0")).toBe(
      "claude-3-haiku-20240307",
    );
  });

  // Cross-region inference model IDs (region.vendor.model-version)
  it("extracts Anthropic model name from cross-region US ID", () => {
    expect(extractModelName("us.anthropic.claude-haiku-4-5-20251001-v1:0")).toBe(
      "claude-haiku-4-5-20251001",
    );
  });

  it("extracts Anthropic model name from cross-region EU ID", () => {
    expect(extractModelName("eu.anthropic.claude-sonnet-4-20250514-v1:0")).toBe(
      "claude-sonnet-4-20250514",
    );
  });

  it("extracts Anthropic model name from cross-region AP ID", () => {
    expect(extractModelName("ap.anthropic.claude-opus-4-5-20251101-v1:0")).toBe(
      "claude-opus-4-5-20251101",
    );
  });

  it("extracts Anthropic model name from cross-region global ID", () => {
    expect(extractModelName("global.anthropic.claude-sonnet-4-5-20250929-v1:0")).toBe(
      "claude-sonnet-4-5-20250929",
    );
  });

  it("extracts Anthropic model name from cross-region APAC ID", () => {
    expect(extractModelName("apac.anthropic.claude-3-5-sonnet-20240620-v1:0")).toBe(
      "claude-3-5-sonnet-20240620",
    );
  });

  it("extracts Anthropic model name from inference profile ARN", () => {
    expect(
      extractModelName(
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
      ),
    ).toBe("claude-3-5-sonnet-20240620");
  });

  // Non-Anthropic models
  it("extracts AI21 model name from standard ID", () => {
    expect(extractModelName("ai21.jamba-1-5-mini-v1:0")).toBe("jamba-1-5-mini-v1:0");
  });

  it("extracts Amazon model name from standard ID", () => {
    expect(extractModelName("amazon.titan-text-express-v1")).toBe("titan-text-express-v1");
  });

  it("extracts Meta model name from standard ID", () => {
    expect(extractModelName("meta.llama3-8b-instruct-v1:0")).toBe("llama3-8b-instruct-v1:0");
  });

  it("extracts Cohere model name from standard ID", () => {
    expect(extractModelName("cohere.command-text-v14")).toBe("command-text-v14");
  });

  it("extracts Amazon Nova model name from cross-region ID", () => {
    expect(extractModelName("us.amazon.nova-lite-v1:0")).toBe("nova-lite-v1:0");
  });

  // Edge cases
  it("returns input unchanged when no dot separator", () => {
    expect(extractModelName("invalid-model-id")).toBe("invalid-model-id");
  });

  it("returns input unchanged for empty string", () => {
    expect(extractModelName("")).toBe("");
  });
});

describe("getSystemFromModelId", () => {
  it("identifies Anthropic from standard ID", () => {
    expect(getSystemFromModelId("anthropic.claude-3-sonnet-20240229-v1:0")).toBe("anthropic");
  });

  it("identifies Anthropic from cross-region ID", () => {
    expect(getSystemFromModelId("us.anthropic.claude-haiku-4-5-20251001-v1:0")).toBe("anthropic");
  });

  it("identifies Meta from standard ID", () => {
    expect(getSystemFromModelId("meta.llama3-8b-instruct-v1:0")).toBe("meta");
  });
});

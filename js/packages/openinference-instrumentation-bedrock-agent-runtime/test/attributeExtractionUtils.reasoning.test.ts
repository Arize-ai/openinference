import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { getAttributesFromModelInvocationOutput } from "../src/attributes/attributeExtractionUtils";

const MSG = SemanticConventions.LLM_OUTPUT_MESSAGES;
const CONTENTS = SemanticConventions.MESSAGE_CONTENTS;
const TYPE = SemanticConventions.MESSAGE_CONTENT_TYPE;
const TEXT = SemanticConventions.MESSAGE_CONTENT_TEXT;
const DATA = SemanticConventions.MESSAGE_CONTENT_DATA;
const SIG = SemanticConventions.MESSAGE_CONTENT_SIGNATURE;
const ROLE = SemanticConventions.MESSAGE_ROLE;

describe("getAttributesFromModelInvocationOutput - reasoning content blocks", () => {
  it("maps a thinking block to type=reasoning with text and signature", () => {
    const output = {
      rawResponse: {
        content: JSON.stringify({
          content: [
            {
              type: "thinking",
              thinking: "Let me work through this step by step.",
              signature: "abc123signature",
            },
          ],
        }),
      },
    };

    const attrs = getAttributesFromModelInvocationOutput(output);

    expect(attrs[`${MSG}.0.${CONTENTS}.0.${TYPE}`]).toBe("reasoning");
    expect(attrs[`${MSG}.0.${CONTENTS}.0.${TEXT}`]).toBe("Let me work through this step by step.");
    expect(attrs[`${MSG}.0.${CONTENTS}.0.${SIG}`]).toBe("abc123signature");
  });

  it("maps a redacted_thinking block to type=reasoning with data", () => {
    const output = {
      rawResponse: {
        content: JSON.stringify({
          content: [{ type: "redacted_thinking", data: "encrypted-thinking-payload" }],
        }),
      },
    };

    const attrs = getAttributesFromModelInvocationOutput(output);

    expect(attrs[`${MSG}.0.${CONTENTS}.0.${TYPE}`]).toBe("reasoning");
    expect(attrs[`${MSG}.0.${CONTENTS}.0.${DATA}`]).toBe("encrypted-thinking-payload");
  });

  it("merges reasoning and text blocks into a single output message", () => {
    const output = {
      rawResponse: {
        content: JSON.stringify({
          role: "assistant",
          content: [
            { type: "thinking", thinking: "Reasoning...", signature: "sig-abc" },
            { type: "text", text: "Final answer." },
          ],
        }),
      },
    };

    const attrs = getAttributesFromModelInvocationOutput(output);

    expect(attrs[`${MSG}.0.${ROLE}`]).toBe("assistant");
    expect(attrs[`${MSG}.0.${CONTENTS}.0.${TYPE}`]).toBe("reasoning");
    expect(attrs[`${MSG}.0.${CONTENTS}.1.${TYPE}`]).toBe("text");
    expect(attrs[`${MSG}.0.${CONTENTS}.1.${TEXT}`]).toBe("Final answer.");
    expect(attrs[`${MSG}.1.${ROLE}`]).toBeUndefined();
    const idKeys = Object.keys(attrs).filter((k) =>
      k.endsWith(`.${SemanticConventions.MESSAGE_CONTENT_ID}`),
    );
    expect(idKeys).toEqual([]);
  });

  it("handles Converse-normalized rawResponse shape with reasoning block", () => {
    const output = {
      rawResponse: {
        content: JSON.stringify({
          output: {
            message: {
              role: "assistant",
              content: [
                {
                  text: null,
                  reasoningContent: {
                    reasoningText: { text: "Let's compute step by step.", signature: null },
                    redactedContent: null,
                  },
                  toolUse: null,
                },
                { text: "The 10th Fibonacci number is 55.", reasoningContent: null, toolUse: null },
              ],
            },
          },
          stopReason: "end_turn",
        }),
      },
    };

    const attrs = getAttributesFromModelInvocationOutput(output);

    expect(attrs[`${MSG}.0.${CONTENTS}.0.${TYPE}`]).toBe("reasoning");
    expect(attrs[`${MSG}.0.${CONTENTS}.0.${TEXT}`]).toBe("Let's compute step by step.");
    expect(attrs[`${MSG}.0.${CONTENTS}.1.${TYPE}`]).toBe("text");
    expect(attrs[`${MSG}.0.${CONTENTS}.1.${TEXT}`]).toBe("The 10th Fibonacci number is 55.");
    expect(attrs["output.value"]).toBe("The 10th Fibonacci number is 55.");
  });
});

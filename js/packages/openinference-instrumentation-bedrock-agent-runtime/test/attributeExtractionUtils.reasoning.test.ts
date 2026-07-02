import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { getAttributesFromModelInvocationOutput } from "../src/attributes/attributeExtractionUtils";

describe("getAttributesFromModelInvocationOutput - reasoning content blocks", () => {
  it("maps a thinking content block to message_content type=reasoning with text and signature", () => {
    const modelInvocationOutput = {
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

    const attributes = getAttributesFromModelInvocationOutput(modelInvocationOutput);

    // Index 0 is the pre-existing stringified raw-content message; the
    // per-content-block message produced from the parsed array is at index 1.
    expect(
      attributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TYPE}`
      ],
    ).toBe("reasoning");
    expect(
      attributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TEXT}`
      ],
    ).toBe("Let me work through this step by step.");
    expect(
      attributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_SIGNATURE}`
      ],
    ).toBe("abc123signature");
  });

  it("maps a redacted_thinking content block to message_content type=reasoning with data only", () => {
    const modelInvocationOutput = {
      rawResponse: {
        content: JSON.stringify({
          content: [
            {
              type: "redacted_thinking",
              data: "encrypted-thinking-payload",
            },
          ],
        }),
      },
    };

    const attributes = getAttributesFromModelInvocationOutput(modelInvocationOutput);

    expect(
      attributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TYPE}`
      ],
    ).toBe("reasoning");
    expect(
      attributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_DATA}`
      ],
    ).toBe("encrypted-thinking-payload");
    expect(
      attributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TEXT}`
      ],
    ).toBeUndefined();
    expect(
      attributes[
        `${SemanticConventions.LLM_OUTPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_SIGNATURE}`
      ],
    ).toBeUndefined();
  });

  it("never emits a message_content id attribute for reasoning blocks", () => {
    const modelInvocationOutput = {
      rawResponse: {
        content: JSON.stringify({
          content: [
            {
              type: "thinking",
              thinking: "Reasoning text",
              signature: "sig",
            },
            {
              type: "redacted_thinking",
              data: "redacted-data",
            },
          ],
        }),
      },
    };

    const attributes = getAttributesFromModelInvocationOutput(modelInvocationOutput);

    const idKeys = Object.keys(attributes).filter((key) =>
      key.endsWith(`.${SemanticConventions.MESSAGE_CONTENT_ID}`),
    );
    expect(idKeys).toEqual([]);
  });
});

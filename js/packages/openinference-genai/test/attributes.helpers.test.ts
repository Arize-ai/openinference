import {
  mapProviderAndSystem,
  mapModels,
  mapSpanKind,
  mapInvocationParameters,
  mapInputMessagesAndInputValue,
  mapOutputMessagesAndOutputValue,
  mapTokenCounts,
  convertGenAISpanAttributesToOpenInferenceSpanAttributes,
} from "../src/attributes.js";

/**
 * These tests validate each mapping helper individually with both
 * valid inputs (based on the OTel GenAI examples) and malformed inputs.
 * Reference: https://opentelemetry.io/docs/specs/semconv/gen-ai/non-normative/examples-llm-calls
 */

describe("attributes helpers", () => {
  describe("mapProviderAndSystem", () => {
    it("maps provider to llm.system and llm.provider", () => {
      const attrs = mapProviderAndSystem({
        "gen_ai.provider.name": "openai",
      });
      expect(attrs["llm.system"]).toBe("openai");
      expect(attrs["llm.provider"]).toBe("openai");
    });

    it("ignores non-string provider", () => {
      const attrs = mapProviderAndSystem({
        "gen_ai.provider.name": 123,
      });
      expect(attrs).toEqual({});
    });
  });

  describe("mapModels", () => {
    it("prefers response model, falls back to request model", () => {
      const withResponse = mapModels({
        "gen_ai.request.model": "gpt-4",
        "gen_ai.response.model": "gpt-4-0613",
      });
      expect(withResponse["llm.model_name"]).toBe("gpt-4-0613");

      const withRequestOnly = mapModels({
        "gen_ai.request.model": "gpt-4",
      });
      expect(withRequestOnly["llm.model_name"]).toBe("gpt-4");
    });

    it("ignores non-string model values", () => {
      const attrs = mapModels({
        "gen_ai.request.model": 42,
        // @ts-expect-error purposely malformed type
        "gen_ai.response.model": null,
      });
      expect(attrs).toEqual({});
    });
  });

  describe("mapSpanKind", () => {
    it("defaults to LLM when no agent markers are present", () => {
      const attrs = mapSpanKind({});
      expect(attrs["openinference.span.kind"]).toBe("LLM");
    });

    it("remains LLM for unrelated attributes (malformed)", () => {
      const attrs = mapSpanKind({ any: "thing" });
      expect(attrs["openinference.span.kind"]).toBe("LLM");
    });
  });

  describe("mapInvocationParameters", () => {
    it("maps known invocation params into llm.invocation_parameters JSON", () => {
      const attrs = mapInvocationParameters({
        "gen_ai.request.model": "gpt-4",
        "gen_ai.request.temperature": 0.5,
        "gen_ai.request.top_p": 0.9,
        "gen_ai.request.top_k": 40,
        "gen_ai.request.presence_penalty": 0.1,
        "gen_ai.request.frequency_penalty": 0.2,
        "gen_ai.request.max_tokens": 200,
        "gen_ai.request.seed": 123,
        "gen_ai.request.stop_sequences": ["stop"],
      });
      const json = attrs["llm.invocation_parameters"] as string;
      expect(typeof json).toBe("string");
      const parsed = JSON.parse(json);
      expect(parsed).toMatchObject({
        model: "gpt-4",
        temperature: 0.5,
        top_p: 0.9,
        top_k: 40,
        presence_penalty: 0.1,
        frequency_penalty: 0.2,
        max_completion_tokens: 200,
        seed: 123,
        stop_sequences: ["stop"],
      });
    });

    it("omits invalid types and supports custom prefix", () => {
      const attrs = mapInvocationParameters(
        {
          // invalid temp should be omitted
          "gen_ai.request.temperature": "hot",
          "gen_ai.request.max_tokens": 150,
        },
        "inputValue",
      );
      const json = attrs["inputValue"] as string;
      expect(typeof json).toBe("string");
      const parsed = JSON.parse(json);
      expect(parsed).toEqual({ max_completion_tokens: 150 });
    });
  });

  describe("mapInputMessagesAndInputValue", () => {
    it("maps structured input messages and sets input.value (with invocation params)", () => {
      const spanAttrs = {
        "gen_ai.input.messages": JSON.stringify([
          {
            role: "system",
            parts: [{ type: "text", content: "You are a helpful assistant." }],
          },
          {
            role: "user",
            parts: [{ type: "text", content: "Weather in Paris?" }],
          },
          {
            role: "assistant",
            parts: [
              {
                type: "tool_call",
                id: "call_VSPygqK",
                name: "get_weather",
                arguments: { location: "Paris" },
              },
            ],
          },
          {
            role: "tool",
            parts: [
              {
                type: "tool_call_response",
                id: "call_VSPygqK",
                response: "rainy, 57°F",
              },
            ],
          },
        ]),
        // include some invocation params to verify merging into input.value
        "gen_ai.request.model": "gpt-4",
        "gen_ai.request.temperature": 0.5,
      } as const;
      const attrs = mapInputMessagesAndInputValue(spanAttrs);

      // llm.input_messages.* extracted
      expect(attrs["llm.input_messages.0.message.role"]).toBe("system");
      expect(attrs["llm.input_messages.0.message.content"]).toBe(
        "You are a helpful assistant.",
      );
      expect(attrs["llm.input_messages.1.message.role"]).toBe("user");
      expect(attrs["llm.input_messages.1.message.content"]).toBe(
        "Weather in Paris?",
      );
      expect(attrs["llm.input_messages.2.message.role"]).toBe("assistant");
      expect(
        attrs[
          "llm.input_messages.2.message.tool_calls.0.tool_call.function.name"
        ],
      ).toBe("get_weather");
      expect(attrs["llm.input_messages.3.message.role"]).toBe("tool");
      expect(attrs["llm.input_messages.3.message.tool_call_id"]).toBe(
        "call_VSPygqK",
      );

      // input.value is JSON, includes invocation params and messages
      const inputValueJson = attrs["input.value"] as string;
      expect(typeof inputValueJson).toBe("string");
      const inputValue = JSON.parse(inputValueJson);
      expect(inputValue).toMatchObject({
        model: "gpt-4",
        temperature: 0.5,
      });
      expect(Array.isArray(inputValue.messages)).toBe(true);
      expect(attrs["input.mime_type"]).toBe("application/json");
    });

    it("falls back to deprecated prompt when input messages missing", () => {
      const attrs = mapInputMessagesAndInputValue({
        "gen_ai.prompt": JSON.stringify([
          { role: "user", parts: [{ type: "text", content: "Hi" }] },
        ]),
      });
      const inputValueJson = attrs["input.value"] as string;
      expect(JSON.parse(inputValueJson).messages[0]).toMatchObject({
        role: "user",
        content: "Hi",
      });
    });

    it("stringifies unparseable input messages (malformed)", () => {
      const attrs = mapInputMessagesAndInputValue({
        // not JSON
        "gen_ai.input.messages": "not-json",
      });
      expect(attrs["input.value"]).toBe(undefined);
      expect(attrs["input.mime_type"]).toBe(undefined);
      // no llm.input_messages.* keys created
      expect(
        Object.keys(attrs).some((k) => k.startsWith("llm.input_messages.")),
      ).toBe(false);
    });
  });

  describe("mapOutputMessagesAndOutputValue", () => {
    it("maps structured output messages and sets output.value", () => {
      const attrs = mapOutputMessagesAndOutputValue({
        "gen_ai.output.messages": JSON.stringify([
          {
            role: "assistant",
            parts: [{ type: "text", content: "The weather is rainy." }],
            finish_reason: "stop",
          },
        ]),
        "gen_ai.response.id": "chatcmpl-123",
        "gen_ai.response.model": "gpt-4-0613",
        "gen_ai.request.model": "gpt-4",
        "gen_ai.response.finish_reasons": ["stop"],
      });
      expect(attrs["llm.output_messages.0.message.role"]).toBe("assistant");
      expect(attrs["llm.output_messages.0.message.content"]).toBe(
        "The weather is rainy.",
      );
      const outJson = attrs["output.value"] as string;
      const parsed = JSON.parse(outJson);
      expect(parsed).toMatchObject({
        id: "chatcmpl-123",
        model: "gpt-4-0613",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "The weather is rainy." },
            finish_reason: "stop",
          },
        ],
      });
      expect(attrs["output.mime_type"]).toBe("application/json");
    });

    it("falls back to deprecated completion text when output messages missing", () => {
      const attrs = mapOutputMessagesAndOutputValue({
        "gen_ai.completion": JSON.stringify({ text: "Answer" }),
        "gen_ai.response.finish_reasons": ["stop"],
      });
      expect(attrs["llm.output_messages.0.message.role"]).toBe("assistant");
      expect(attrs["llm.output_messages.0.message.content"]).toBe("Answer");
    });

    it("stringifies unparseable output messages (malformed)", () => {
      const attrs = mapOutputMessagesAndOutputValue({
        // not JSON
        "gen_ai.output.messages": "[malformed]",
      });
      expect(attrs["output.value"]).toBe(undefined);
      expect(attrs["output.mime_type"]).toBe(undefined);
    });
  });

  describe("mapTokenCounts", () => {
    it("maps prompt, completion, and total token counts", () => {
      const attrs = mapTokenCounts({
        "gen_ai.usage.input_tokens": 97,
        "gen_ai.usage.output_tokens": 52,
      });
      expect(attrs["llm.token_count.prompt"]).toBe(97);
      expect(attrs["llm.token_count.completion"]).toBe(52);
      expect(attrs["llm.token_count.total"]).toBe(149);
    });

    it("ignores non-number token counts (malformed)", () => {
      const attrs = mapTokenCounts({
        "gen_ai.usage.input_tokens": "97",
        // @ts-expect-error purposely malformed type
        "gen_ai.usage.output_tokens": null,
      });
      expect(attrs).toEqual({});
    });
  });

  describe("convertGenAISpanAttributesToOpenInferenceSpanAttributes", () => {
    it("returns minimal defaults for empty attributes (span kind only)", () => {
      const attrs = convertGenAISpanAttributesToOpenInferenceSpanAttributes({});
      expect(attrs).toEqual({ "openinference.span.kind": "LLM" });
    });
  });
});

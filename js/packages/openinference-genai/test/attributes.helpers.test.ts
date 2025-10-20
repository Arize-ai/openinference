import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import {
  mapProviderAndSystem,
  mapModels,
  mapSpanKind,
  mapInvocationParameters,
  mapInputMessages,
  mapOutputMessages,
  mapTokenCounts,
  convertGenAISpanAttributesToOpenInferenceSpanAttributes,
  mapToolExecution,
  mapInputValue,
  mapOutputValue,
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
    it("defaults to LLM when no agent or tool execution markers are present", () => {
      const attrs = mapSpanKind({});
      expect(attrs["openinference.span.kind"]).toBe("LLM");
    });

    it("maps agent markers to AGENT", () => {
      const attrs = mapSpanKind({
        "gen_ai.agent.name": "my_agent",
      });
      expect(attrs["openinference.span.kind"]).toBe("AGENT");
    });

    it("maps tool execution markers to TOOL", () => {
      const attrs = mapSpanKind({
        "gen_ai.tool.name": "my_tool",
      });
      expect(attrs["openinference.span.kind"]).toBe("TOOL");
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
      const attrs = mapInvocationParameters({
        // invalid temp should be omitted
        "gen_ai.request.temperature": "hot",
        "gen_ai.request.max_tokens": 150,
      });
      const json = attrs[
        SemanticConventions.LLM_INVOCATION_PARAMETERS
      ] as string;
      expect(typeof json).toBe("string");
      const parsed = JSON.parse(json);
      expect(parsed).toEqual({ max_completion_tokens: 150 });
    });
  });

  describe("mapInputValue", () => {
    it("maps input value and mime type", () => {
      const attrs = mapInputValue({
        input: JSON.stringify([
          { role: "user", parts: [{ type: "text", content: "Hi" }] },
        ]),
      });
      expect(attrs["input.value"]).toBe(
        JSON.stringify([
          { role: "user", parts: [{ type: "text", content: "Hi" }] },
        ]),
      );
      expect(attrs["input.mime_type"]).toBe("application/json");
    });
    it("falls back to deprecated prompt when input is missing", () => {
      const attrs = mapInputValue({
        "gen_ai.prompt": JSON.stringify([
          { role: "user", parts: [{ type: "text", content: "Hi" }] },
        ]),
      });
      expect(attrs["input.value"]).toBe(
        JSON.stringify([
          { role: "user", parts: [{ type: "text", content: "Hi" }] },
        ]),
      );
      expect(attrs["input.mime_type"]).toBe("application/json");
    });
  });

  describe("mapOutputValue", () => {
    it("maps output value and mime type", () => {
      const attrs = mapOutputValue({
        output: JSON.stringify([
          { role: "assistant", parts: [{ type: "text", content: "Hi" }] },
        ]),
      });
      expect(attrs["output.value"]).toBe(
        JSON.stringify([
          { role: "assistant", parts: [{ type: "text", content: "Hi" }] },
        ]),
      );
      expect(attrs["output.mime_type"]).toBe("application/json");
    });
    it("falls back to deprecated completion when output is missing", () => {
      const attrs = mapOutputValue({
        "gen_ai.completion": JSON.stringify([
          { role: "assistant", parts: [{ type: "text", content: "Hi" }] },
        ]),
      });
      expect(attrs["output.value"]).toBe(
        JSON.stringify([
          { role: "assistant", parts: [{ type: "text", content: "Hi" }] },
        ]),
      );
      expect(attrs["output.mime_type"]).toBe("application/json");
    });
  });

  describe("mapInputMessagesAndInputValue", () => {
    it("maps structured input messages and forwards input.value", () => {
      const input = [
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
              type: "text",
              content: "Absolutely! Let me check the weather for you.",
            },
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
      ];

      const spanAttrs = {
        "gen_ai.input.messages": JSON.stringify(input),
        // include some invocation params to verify merging into input.value
        "gen_ai.request.model": "gpt-4",
        "gen_ai.request.temperature": 0.5,
        // pass input to ensure that it is simply forwarded to input.value
        input: JSON.stringify([
          {
            role: "user",
            parts: [{ type: "text", content: "Weather in Paris?" }],
          },
        ]),
      } as const;
      const attrs = mapInputMessages(spanAttrs);

      // llm.input_messages.* extracted
      expect(attrs["llm.input_messages.0.message.role"]).toBe("system");
      expect(
        attrs["llm.input_messages.0.message.contents.0.message_content.text"],
      ).toBe("You are a helpful assistant.");
      expect(attrs["llm.input_messages.1.message.role"]).toBe("user");
      expect(
        attrs["llm.input_messages.1.message.contents.0.message_content.text"],
      ).toBe("Weather in Paris?");
      expect(attrs["llm.input_messages.2.message.role"]).toBe("assistant");
      expect(
        attrs["llm.input_messages.2.message.contents.0.message_content.type"],
      ).toBe("text");
      expect(
        attrs["llm.input_messages.2.message.contents.0.message_content.text"],
      ).toBe("Absolutely! Let me check the weather for you.");
      expect(
        attrs[
          "llm.input_messages.2.message.tool_calls.0.tool_call.function.name"
        ],
      ).toBe("get_weather");
      expect(attrs["llm.input_messages.3.message.role"]).toBe("tool");
      expect(attrs["llm.input_messages.3.message.tool_call_id"]).toBe(
        "call_VSPygqK",
      );

      const inOutAttrs = {
        ...mapInputValue(spanAttrs),
        ...mapOutputValue(spanAttrs),
      };

      expect(inOutAttrs["input.value"]).toBe(spanAttrs["input"]);
      expect(inOutAttrs["input.mime_type"]).toBe("application/json");
    });

    it("falls back to deprecated prompt when input messages missing", () => {
      const spanAttrs = {
        "gen_ai.prompt": JSON.stringify([
          { role: "user", parts: [{ type: "text", content: "Hi" }] },
        ]),
      };
      const attrs = mapInputMessages(spanAttrs);
      expect(attrs["input.value"]).toBe(undefined);
      expect(attrs["input.mime_type"]).toBe(undefined);

      const inOutAttrs = {
        ...mapInputValue(spanAttrs),
        ...mapOutputValue(spanAttrs),
      };
      expect(inOutAttrs["input.value"]).toBe(spanAttrs["gen_ai.prompt"]);
      expect(inOutAttrs["input.mime_type"]).toBe("application/json");
    });

    it("stringifies unparseable input messages (malformed)", () => {
      const attrs = mapInputMessages({
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
      const spanAttrs = {
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
        output: JSON.stringify([
          {
            role: "assistant",
            parts: [{ type: "text", content: "The weather is rainy." }],
            finish_reason: "stop",
          },
        ]),
      };
      const attrs = mapOutputMessages(spanAttrs);
      expect(attrs["llm.output_messages.0.message.role"]).toBe("assistant");
      expect(
        attrs["llm.output_messages.0.message.contents.0.message_content.text"],
      ).toBe("The weather is rainy.");
      const inOutAttrs = {
        ...mapInputValue(spanAttrs),
        ...mapOutputValue(spanAttrs),
      };
      expect(inOutAttrs["output.value"]).toBe(spanAttrs["output"]);
      expect(inOutAttrs["output.mime_type"]).toBe("application/json");
    });

    it("falls back to deprecated completion text when output messages missing", () => {
      const spanAttrs = {
        "gen_ai.completion": JSON.stringify({ text: "Answer" }),
        "gen_ai.response.finish_reasons": ["stop"],
      };
      const attrs = mapOutputMessages(spanAttrs);
      expect(attrs["output.value"]).toBe(undefined);
      expect(attrs["output.mime_type"]).toBe(undefined);
      const inOutAttrs = {
        ...mapInputValue(spanAttrs),
        ...mapOutputValue(spanAttrs),
      };
      expect(inOutAttrs["output.value"]).toBe(spanAttrs["gen_ai.completion"]);
      expect(inOutAttrs["output.mime_type"]).toBe("application/json");
    });

    it("stringifies unparseable output messages (malformed)", () => {
      const attrs = mapOutputMessages({
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

  describe("mapToolExecution", () => {
    it("maps tool execution details", () => {
      const spanAttrs = {
        "gen_ai.tool.name": "get_weather",
        "gen_ai.tool.description":
          "Retrieves the current weather report for a specified city.",
        "gen_ai.tool.call.id": "1234",
        "gen_ai.tool.type": "function",
        input: '{"city": "New York"}',
        output:
          '{"status": "success", "report": "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."}',
      };
      const attrs = mapToolExecution(spanAttrs);
      expect(attrs["tool.name"]).toBe("get_weather");
      expect(attrs["tool.description"]).toBe(
        "Retrieves the current weather report for a specified city.",
      );
      expect(attrs["tool_call.id"]).toBe("1234");
      // input and output are mapped separately
      const inOutAttrs = {
        ...mapInputValue(spanAttrs),
        ...mapOutputValue(spanAttrs),
      };
      expect(inOutAttrs["input.value"]).toBe('{"city": "New York"}');
      expect(inOutAttrs["input.mime_type"]).toBe("application/json");
      expect(inOutAttrs["output.value"]).toBe(
        '{"status": "success", "report": "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."}',
      );
      expect(inOutAttrs["output.mime_type"]).toBe("application/json");
    });
  });

  describe("convertGenAISpanAttributesToOpenInferenceSpanAttributes", () => {
    it("returns minimal defaults for empty attributes (span kind only)", () => {
      const attrs = convertGenAISpanAttributesToOpenInferenceSpanAttributes({});
      expect(attrs).toEqual({ "openinference.span.kind": "LLM" });
    });
  });
});

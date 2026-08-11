package instrumentation

import (
	"os"
	"strconv"
)

// RedactedValue is the placeholder string written in place of any
// attribute value the TraceConfig has marked as hidden. Matches
// Python's openinference.instrumentation.REDACTED_VALUE so a trace
// crossing language boundaries reads the same.
const RedactedValue = "__REDACTED__"

// Environment variable names recognised by TraceConfigFromEnv. They
// match the canonical OPENINFERENCE_HIDE_* names used by the Python
// SDK, so a customer setting OPENINFERENCE_HIDE_INPUTS=true in their
// environment gets the same behavior across Python and Go.
const (
	EnvHideInputs                  = "OPENINFERENCE_HIDE_INPUTS"
	EnvHideOutputs                 = "OPENINFERENCE_HIDE_OUTPUTS"
	EnvHideInputMessages           = "OPENINFERENCE_HIDE_INPUT_MESSAGES"
	EnvHideOutputMessages          = "OPENINFERENCE_HIDE_OUTPUT_MESSAGES"
	EnvHideInputText               = "OPENINFERENCE_HIDE_INPUT_TEXT"
	EnvHideOutputText              = "OPENINFERENCE_HIDE_OUTPUT_TEXT"
	EnvHideLLMInvocationParameters = "OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS"
	EnvHideLLMTools                = "OPENINFERENCE_HIDE_LLM_TOOLS"
	EnvHidePrompts                 = "OPENINFERENCE_HIDE_PROMPTS"
)

// TraceConfig controls which OpenInference attributes the provider
// instrumentors emit. The zero value emits everything (full fidelity);
// individual flags can be set to suppress specific attribute families
// for PII / sensitive-data protection. Behavior matches Python's
// TraceConfig.mask exactly — see each flag's GoDoc for which keys are
// dropped versus redacted with RedactedValue.
//
// Construct via TraceConfigFromEnv() to pick up the canonical
// OPENINFERENCE_HIDE_* environment variables (matches Python / JS /
// Java behavior), or build directly for in-code overrides.
type TraceConfig struct {
	// HideInputs is the strongest input-side flag. It replaces
	// input.value with RedactedValue and DROPS the entire
	// llm.input_messages.* family (roles, content, names,
	// tool_call_ids, and nested tool_calls) AND drops llm.tools.*
	// (the advertised tool list is considered part of the input).
	// Use this when no input content should leave the process.
	HideInputs bool

	// HideOutputs is the strongest output-side flag. It replaces
	// output.value with RedactedValue and DROPS the entire
	// llm.output_messages.* family (roles, content, and nested
	// tool_calls). llm.finish_reason is metadata and remains visible.
	HideOutputs bool

	// HideInputMessages drops the entire llm.input_messages.*
	// family. input.value, llm.tools.*, and llm.invocation_parameters
	// remain visible — use HideInputs for stricter suppression.
	HideInputMessages bool

	// HideOutputMessages drops the entire llm.output_messages.*
	// family. output.value remains visible — use HideOutputs for
	// stricter suppression.
	HideOutputMessages bool

	// HideInputText keeps llm.input_messages.* structure (roles,
	// indices, tool-call shells) and replaces only the message
	// content field with RedactedValue. For fine-grained control
	// when the message *shape* is useful but the *text* is sensitive.
	HideInputText bool

	// HideOutputText keeps llm.output_messages.* structure and
	// replaces only the message content field with RedactedValue.
	HideOutputText bool

	// HideLLMInvocationParameters omits the llm.invocation_parameters
	// attribute (model temperature/top_p/etc.).
	HideLLMInvocationParameters bool

	// HideLLMTools drops llm.tools.* (the advertised tool list).
	// HideInputs implies this — set HideLLMTools when you want to
	// hide the tool list but keep input messages visible.
	HideLLMTools bool

	// HidePrompts omits the completions-API llm.prompts.* attributes.
	HidePrompts bool
}

// TraceConfigFromEnv populates a TraceConfig by reading the
// OPENINFERENCE_HIDE_* environment variables. Missing or unparsable
// vars default to false (no hiding). Boolean parsing accepts
// "1"/"true"/"t" (case-insensitive) as true; anything else is false.
//
// Both Middleware and NewTransport call this by default at
// instrumentor construction, so the canonical opt-in path for a
// customer is simply: `export OPENINFERENCE_HIDE_INPUTS=true`.
func TraceConfigFromEnv() TraceConfig {
	return TraceConfig{
		HideInputs:                  envBool(EnvHideInputs),
		HideOutputs:                 envBool(EnvHideOutputs),
		HideInputMessages:           envBool(EnvHideInputMessages),
		HideOutputMessages:          envBool(EnvHideOutputMessages),
		HideInputText:               envBool(EnvHideInputText),
		HideOutputText:              envBool(EnvHideOutputText),
		HideLLMInvocationParameters: envBool(EnvHideLLMInvocationParameters),
		HideLLMTools:                envBool(EnvHideLLMTools),
		HidePrompts:                 envBool(EnvHidePrompts),
	}
}

// MaskInputText returns either the supplied text or RedactedValue,
// depending on whether the config marks input text as hidden.
// Provider instrumentors use this to gate llm.input_messages.{i}.message.content.
func (c TraceConfig) MaskInputText(text string) string {
	if c.HideInputs || c.HideInputText {
		return RedactedValue
	}
	return text
}

// MaskOutputText returns either the supplied text or RedactedValue,
// depending on whether the config marks output text as hidden.
func (c TraceConfig) MaskOutputText(text string) string {
	if c.HideOutputs || c.HideOutputText {
		return RedactedValue
	}
	return text
}

// MaskInputValue returns either the supplied value or RedactedValue,
// depending on whether the config marks the top-level input as hidden.
func (c TraceConfig) MaskInputValue(value string) string {
	if c.HideInputs {
		return RedactedValue
	}
	return value
}

// MaskOutputValue returns either the supplied value or RedactedValue,
// depending on whether the config marks the top-level output as hidden.
func (c TraceConfig) MaskOutputValue(value string) string {
	if c.HideOutputs {
		return RedactedValue
	}
	return value
}

func envBool(key string) bool {
	v := os.Getenv(key)
	if v == "" {
		return false
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return false
	}
	return b
}

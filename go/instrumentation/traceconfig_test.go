package instrumentation_test

import (
	"testing"

	"github.com/Arize-ai/openinference/go/instrumentation"
)

func TestTraceConfigFromEnv_EmptyDefaultsToZeroValue(t *testing.T) {
	// Each Setenv("", "") relies on the helper resetting at test scope.
	for _, k := range []string{
		instrumentation.EnvHideInputs,
		instrumentation.EnvHideOutputs,
		instrumentation.EnvHideInputMessages,
		instrumentation.EnvHideOutputMessages,
		instrumentation.EnvHideInputText,
		instrumentation.EnvHideOutputText,
		instrumentation.EnvHideLLMInvocationParameters,
		instrumentation.EnvHideLLMTools,
		instrumentation.EnvHidePrompts,
	} {
		t.Setenv(k, "")
	}
	c := instrumentation.TraceConfigFromEnv()
	want := instrumentation.TraceConfig{}
	if c != want {
		t.Errorf("empty env: got %+v want zero value", c)
	}
}

func TestTraceConfigFromEnv_AllFlagsSet(t *testing.T) {
	cases := map[string]func(instrumentation.TraceConfig) bool{
		instrumentation.EnvHideInputs:                  func(c instrumentation.TraceConfig) bool { return c.HideInputs },
		instrumentation.EnvHideOutputs:                 func(c instrumentation.TraceConfig) bool { return c.HideOutputs },
		instrumentation.EnvHideInputMessages:           func(c instrumentation.TraceConfig) bool { return c.HideInputMessages },
		instrumentation.EnvHideOutputMessages:          func(c instrumentation.TraceConfig) bool { return c.HideOutputMessages },
		instrumentation.EnvHideInputText:               func(c instrumentation.TraceConfig) bool { return c.HideInputText },
		instrumentation.EnvHideOutputText:              func(c instrumentation.TraceConfig) bool { return c.HideOutputText },
		instrumentation.EnvHideLLMInvocationParameters: func(c instrumentation.TraceConfig) bool { return c.HideLLMInvocationParameters },
		instrumentation.EnvHideLLMTools:                func(c instrumentation.TraceConfig) bool { return c.HideLLMTools },
		instrumentation.EnvHidePrompts:                 func(c instrumentation.TraceConfig) bool { return c.HidePrompts },
	}
	for envVar, getter := range cases {
		t.Run(envVar, func(t *testing.T) {
			t.Setenv(envVar, "true")
			c := instrumentation.TraceConfigFromEnv()
			if !getter(c) {
				t.Errorf("env %s=true should produce config flag true, got %+v", envVar, c)
			}
		})
	}
}

func TestTraceConfigFromEnv_BooleanParsing(t *testing.T) {
	cases := []struct {
		val  string
		want bool
	}{
		{"true", true},
		{"True", true},
		{"TRUE", true},
		{"1", true},
		{"t", true},
		{"false", false},
		{"0", false},
		{"", false},
		{"yes", false}, // strconv.ParseBool only accepts canonical bools
		{"garbage", false},
	}
	for _, c := range cases {
		t.Run("val="+c.val, func(t *testing.T) {
			t.Setenv(instrumentation.EnvHideInputs, c.val)
			cfg := instrumentation.TraceConfigFromEnv()
			if cfg.HideInputs != c.want {
				t.Errorf("env value %q: got HideInputs=%v want %v", c.val, cfg.HideInputs, c.want)
			}
		})
	}
}

func TestTraceConfig_MaskInputText(t *testing.T) {
	cases := []struct {
		name string
		cfg  instrumentation.TraceConfig
		in   string
		want string
	}{
		{"no hiding", instrumentation.TraceConfig{}, "hello", "hello"},
		{"HideInputs", instrumentation.TraceConfig{HideInputs: true}, "hello", instrumentation.RedactedValue},
		{"HideInputText only", instrumentation.TraceConfig{HideInputText: true}, "hello", instrumentation.RedactedValue},
		{"both", instrumentation.TraceConfig{HideInputs: true, HideInputText: true}, "hello", instrumentation.RedactedValue},
		{"output flag doesn't affect input", instrumentation.TraceConfig{HideOutputs: true}, "hello", "hello"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.cfg.MaskInputText(c.in); got != c.want {
				t.Errorf("got %q want %q", got, c.want)
			}
		})
	}
}

func TestTraceConfig_MaskOutputText(t *testing.T) {
	cases := []struct {
		name string
		cfg  instrumentation.TraceConfig
		in   string
		want string
	}{
		{"no hiding", instrumentation.TraceConfig{}, "hello", "hello"},
		{"HideOutputs", instrumentation.TraceConfig{HideOutputs: true}, "hello", instrumentation.RedactedValue},
		{"HideOutputText only", instrumentation.TraceConfig{HideOutputText: true}, "hello", instrumentation.RedactedValue},
		{"input flag doesn't affect output", instrumentation.TraceConfig{HideInputs: true}, "hello", "hello"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.cfg.MaskOutputText(c.in); got != c.want {
				t.Errorf("got %q want %q", got, c.want)
			}
		})
	}
}

func TestTraceConfig_MaskInputOutputValue(t *testing.T) {
	full := instrumentation.TraceConfig{HideInputs: true, HideOutputs: true}
	none := instrumentation.TraceConfig{}

	if got := full.MaskInputValue("x"); got != instrumentation.RedactedValue {
		t.Errorf("HideInputs MaskInputValue: got %q", got)
	}
	if got := none.MaskInputValue("x"); got != "x" {
		t.Errorf("default MaskInputValue: got %q want %q", got, "x")
	}
	if got := full.MaskOutputValue("y"); got != instrumentation.RedactedValue {
		t.Errorf("HideOutputs MaskOutputValue: got %q", got)
	}
	if got := none.MaskOutputValue("y"); got != "y" {
		t.Errorf("default MaskOutputValue: got %q want %q", got, "y")
	}
}

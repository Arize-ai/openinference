package httputil_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/Arize-ai/openinference/go/openinference-instrumentation-openai-go/internal/httputil"
)

// parseSSE feeds stream to a parser in chunks of size n and returns the
// dispatched event data.
func parseSSE(stream string, n, maxBytes int) []string {
	var events []string
	p := &httputil.SSEParser{
		OnEvent:       func(data []byte) { events = append(events, string(data)) },
		MaxEventBytes: maxBytes,
	}
	for len(stream) > 0 {
		k := min(n, len(stream))
		_, _ = p.Write([]byte(stream[:k]))
		stream = stream[k:]
	}
	return events
}

func TestSSEParser_EventsAcrossEveryChunkBoundary(t *testing.T) {
	const stream = ": keep-alive comment\n" +
		"event: response.created\n" +
		"data: {\"a\":1}\n\n" +
		"event: response.output_text.delta\r\n" +
		"data: line one\r\n" +
		"data:line two\r\n" +
		"id: 7\r\n" +
		"\r\n" +
		"event: no-data\n\n" +
		"data: trailing event without a blank line"
	want := []string{`{"a":1}`, "line one\nline two"}
	for n := 1; n <= len(stream); n++ {
		if got := parseSSE(stream, n, 0); !reflect.DeepEqual(got, want) {
			t.Fatalf("chunk size %d: got %q want %q", n, got, want)
		}
	}
}

func TestSSEParser_OversizedEventIsDroppedAndParsingRecovers(t *testing.T) {
	stream := "data: " + strings.Repeat("x", 100) + "\n" +
		"data: more\n\n" +
		"data: small\n\n"
	for _, n := range []int{1, 7, len(stream)} {
		if got := parseSSE(stream, n, 64); !reflect.DeepEqual(got, []string{"small"}) {
			t.Errorf("chunk size %d: got %q want [small]", n, got)
		}
	}
}

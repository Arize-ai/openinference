package httputil_test

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/Arize-ai/openinference/go/instrumentation/internal/httputil"
)

func TestReadAndRestore_Roundtrip(t *testing.T) {
	body := io.NopCloser(strings.NewReader("hello world"))
	rc := body
	data, err := httputil.ReadAndRestore(&rc)
	if err != nil {
		t.Fatalf("ReadAndRestore: %v", err)
	}
	if string(data) != "hello world" {
		t.Errorf("data: got %q", data)
	}
	// rc must still be readable.
	rest, err := io.ReadAll(rc)
	if err != nil {
		t.Fatalf("re-read: %v", err)
	}
	if string(rest) != "hello world" {
		t.Errorf("re-read: got %q", rest)
	}
}

func TestReadAndRestore_NilBody(t *testing.T) {
	var rc io.ReadCloser
	if _, err := httputil.ReadAndRestore(&rc); err == nil {
		t.Fatal("expected error on nil body")
	}
	if _, err := httputil.ReadAndRestore(nil); err == nil {
		t.Fatal("expected error on nil pointer")
	}
}

type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, errors.New("boom") }
func (errReader) Close() error             { return nil }

func TestReadAndRestore_ReadError(t *testing.T) {
	var rc io.ReadCloser = errReader{}
	data, err := httputil.ReadAndRestore(&rc)
	if err == nil {
		t.Fatal("expected error from underlying reader")
	}
	if data != nil {
		t.Errorf("expected nil data on error, got %q", data)
	}
}

func TestIsStreaming(t *testing.T) {
	cases := []struct {
		ct   string
		want bool
	}{
		{"text/event-stream", true},
		{"text/event-stream; charset=utf-8", true},
		{"application/json", false},
		{"", false},
	}
	for _, c := range cases {
		resp := &http.Response{Header: http.Header{}}
		if c.ct != "" {
			resp.Header.Set("Content-Type", c.ct)
		}
		if got := httputil.IsStreaming(resp); got != c.want {
			t.Errorf("IsStreaming(%q): got %v want %v", c.ct, got, c.want)
		}
	}
	if httputil.IsStreaming(nil) {
		t.Error("IsStreaming(nil) should be false")
	}
}

// countingCloser records whether Close was called.
type countingCloser struct {
	io.Reader
	closed int
}

func (c *countingCloser) Close() error {
	c.closed++
	return nil
}

func TestSpanEndingBody_EndsOnClose(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	_, span := tp.Tracer("test").Start(context.Background(), "stream")

	cc := &countingCloser{Reader: strings.NewReader("data: 1\n\ndata: 2\n\n")}
	body := &httputil.SpanEndingBody{ReadCloser: cc, Span: span}

	// Pre-close: span must not yet be in Ended().
	if len(recorder.Ended()) != 0 {
		t.Fatalf("span ended prematurely: %d", len(recorder.Ended()))
	}

	// Read partway — still no End.
	buf := make([]byte, 7)
	if _, err := body.Read(buf); err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(recorder.Ended()) != 0 {
		t.Fatalf("span ended during streaming read: %d", len(recorder.Ended()))
	}

	if err := body.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if got := len(recorder.Ended()); got != 1 {
		t.Fatalf("expected 1 ended span after close, got %d", got)
	}
	if cc.closed != 1 {
		t.Errorf("underlying Close called %d times, want 1", cc.closed)
	}
}

func TestSpanEndingBody_EndsOnEOF(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	_, span := tp.Tracer("test").Start(context.Background(), "stream")

	cc := &countingCloser{Reader: strings.NewReader("data: 1\n\n")}
	body := &httputil.SpanEndingBody{ReadCloser: cc, Span: span}

	// Drain the body — should end on EOF.
	if _, err := io.ReadAll(body); err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if got := len(recorder.Ended()); got != 1 {
		t.Fatalf("expected 1 ended span after EOF, got %d", got)
	}
	// Subsequent Close must not double-end.
	_ = body.Close()
	if got := len(recorder.Ended()); got != 1 {
		t.Fatalf("Close after EOF re-ended span: %d", got)
	}
}

type erroringReader struct{ closed int }

func (e *erroringReader) Read([]byte) (int, error) { return 0, errors.New("network reset") }
func (e *erroringReader) Close() error {
	e.closed++
	return nil
}

func TestSpanEndingBody_NonEOFReadErrorDoesNotEnd(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	_, span := tp.Tracer("test").Start(context.Background(), "stream")

	body := &httputil.SpanEndingBody{ReadCloser: &erroringReader{}, Span: span}

	buf := make([]byte, 8)
	_, err := body.Read(buf)
	if err == nil {
		t.Fatal("expected read error")
	}
	// Span must NOT be ended by a non-EOF read error — caller still has
	// to call Close. End is reserved for terminal events.
	if got := len(recorder.Ended()); got != 0 {
		t.Fatalf("span ended on non-EOF read error: %d", got)
	}
	_ = body.Close()
	if got := len(recorder.Ended()); got != 1 {
		t.Fatalf("expected 1 ended span after Close, got %d", got)
	}
}

// Package httputil holds tiny helpers shared between the provider-specific
// instrumentor middlewares (anthropic, openai, …). They all need to read
// and restore an http.Request/http.Response body, detect SSE streaming
// responses, and tie a span's End to the lifetime of a streaming body —
// keeping these in one place avoids the "fix the bug twice" trap when a
// future refinement lands.
package httputil

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"

	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// ReadAndRestore drains rc and replaces it with a new ReadCloser over the
// drained bytes so the SDK's decoder can still read the body. Returns the
// drained bytes; if rc or *rc is nil — a valid HTTP state for bodyless
// requests — returns (nil, nil) so the caller forwards the request
// untouched without recording a spurious instrumentation error.
//
// Callers MUST treat a non-nil error as fatal for the in-flight request:
// the body may be partially consumed and forwarding it downstream would
// produce a malformed request.
func ReadAndRestore(rc *io.ReadCloser) ([]byte, error) {
	if rc == nil || *rc == nil {
		return nil, nil
	}
	data, err := io.ReadAll(*rc)
	if err != nil {
		return nil, err
	}
	_ = (*rc).Close()
	*rc = io.NopCloser(bytes.NewReader(data))
	return data, nil
}

// IsStreaming reports whether resp is an SSE streaming response. We use
// this to switch to deferred span-ending so the caller's stream consumer
// keeps receiving deltas in real time.
func IsStreaming(resp *http.Response) bool {
	if resp == nil {
		return false
	}
	return strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream")
}

// SpanEndingBody wraps an SSE streaming response body so that the
// associated span lives for the duration the caller is actually reading
// the stream — instead of ending when the HTTP handshake completes
// (which would give every streaming call a misleading near-zero latency).
//
// End() is called exactly once, on whichever of these happens first:
//
//   - Read returns io.EOF (caller drained the stream cleanly).
//   - Close is called (caller is done with the body, drained or not).
//
// Non-EOF Read errors are recorded on the span via RecordError but do
// not end the span — Close still has to fire for that.
type SpanEndingBody struct {
	io.ReadCloser
	Span   trace.Span
	closed atomic.Bool
}

// Read forwards to the underlying ReadCloser and ends the span on EOF.
// Non-EOF read errors are recorded as exception events AND mark the
// span's status as Error — otherwise a broken SSE stream would look
// healthy to any status-based filter on the backend.
func (b *SpanEndingBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if err != nil && !errors.Is(err, io.EOF) {
		b.Span.RecordError(err)
		b.Span.SetStatus(codes.Error, err.Error())
	}
	if errors.Is(err, io.EOF) {
		b.endOnce()
	}
	return n, err
}

// Close forwards to the underlying ReadCloser and ends the span. Close
// errors set the span status to Error so the failure is visible to
// status-based filters and not just to event-aware viewers.
func (b *SpanEndingBody) Close() error {
	closeErr := b.ReadCloser.Close()
	if closeErr != nil {
		b.Span.RecordError(closeErr)
		b.Span.SetStatus(codes.Error, closeErr.Error())
	}
	b.endOnce()
	return closeErr
}

func (b *SpanEndingBody) endOnce() {
	if !b.closed.Swap(true) {
		b.Span.End()
	}
}

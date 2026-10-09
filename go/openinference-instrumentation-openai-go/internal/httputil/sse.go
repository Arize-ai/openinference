package httputil

import "bytes"

// DefaultMaxSSEEventBytes caps how much of a single SSE event
// SSEParser buffers. A Responses API response.completed event carries
// the whole response, which can include base64 image output, so the cap
// is generous; an event over it is dropped rather than buffered.
const DefaultMaxSSEEventBytes = 64 << 20

// SSEParser incrementally parses a text/event-stream body as the caller
// reads it. Feed it every chunk with Write, in order; OnEvent receives
// the data of each complete event (multiple data lines joined with
// "\n", as the SSE spec requires). Other fields (event, id, retry) and
// comments are ignored. Lines may end in "\n" or "\r\n".
//
// It never fails: malformed or oversized input is skipped, because the
// parser only observes a stream that the caller is consuming.
type SSEParser struct {
	// OnEvent receives each event's data. The slice is reused after the
	// call returns, so copy it to keep it.
	OnEvent func(data []byte)
	// MaxEventBytes caps the bytes buffered for one event. Zero means
	// DefaultMaxSSEEventBytes.
	MaxEventBytes int

	line      []byte // current line, unless the event overflowed
	lineLen   int    // bytes seen on the current line, stored or not
	lineFirst byte
	data      []byte
	hasData   bool
	overflow  bool // current event exceeded the cap; skip to its end
}

// Write consumes p. It always returns len(p), nil.
func (s *SSEParser) Write(p []byte) (int, error) {
	n := len(p)
	for len(p) > 0 {
		i := bytes.IndexByte(p, '\n')
		if i < 0 {
			s.appendLine(p)
			break
		}
		s.appendLine(p[:i])
		s.endLine()
		p = p[i+1:]
	}
	return n, nil
}

func (s *SSEParser) limit() int {
	if s.MaxEventBytes > 0 {
		return s.MaxEventBytes
	}
	return DefaultMaxSSEEventBytes
}

func (s *SSEParser) appendLine(p []byte) {
	if len(p) == 0 {
		return
	}
	if s.lineLen == 0 {
		s.lineFirst = p[0]
	}
	s.lineLen += len(p)
	if s.overflow {
		return
	}
	if len(s.line)+len(s.data)+len(p) > s.limit() {
		s.overflow = true
		s.line = s.line[:0]
		s.data = s.data[:0]
		return
	}
	s.line = append(s.line, p...)
}

func (s *SSEParser) endLine() {
	blank := s.lineLen == 0 || (s.lineLen == 1 && s.lineFirst == '\r')
	switch {
	case blank:
		s.dispatch()
	case !s.overflow:
		s.processField(bytes.TrimSuffix(s.line, []byte("\r")))
	}
	s.line = s.line[:0]
	s.lineLen = 0
}

func (s *SSEParser) processField(line []byte) {
	value, ok := bytes.CutPrefix(line, []byte("data:"))
	if !ok {
		return
	}
	value = bytes.TrimPrefix(value, []byte(" "))
	if s.hasData {
		s.data = append(s.data, '\n')
	}
	s.data = append(s.data, value...)
	s.hasData = true
}

func (s *SSEParser) dispatch() {
	if s.hasData && !s.overflow && s.OnEvent != nil {
		s.OnEvent(s.data)
	}
	s.data = s.data[:0]
	s.hasData = false
	s.overflow = false
}

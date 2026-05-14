module github.com/Arize-ai/openinference/go/openinference-instrumentation-openai-go

go 1.25.0

require (
	github.com/Arize-ai/openinference/go/openinference-instrumentation v0.1.0
	github.com/Arize-ai/openinference/go/openinference-semantic-conventions v0.1.0
	github.com/openai/openai-go v1.12.0
	go.opentelemetry.io/otel v1.43.0
	go.opentelemetry.io/otel/sdk v1.43.0
	go.opentelemetry.io/otel/trace v1.43.0
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/tidwall/gjson v1.14.4 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	go.opentelemetry.io/auto/sdk v1.2.1 // indirect
	go.opentelemetry.io/otel/metric v1.43.0 // indirect
	golang.org/x/sys v0.42.0 // indirect
)

// In-repo development: see go/go.work for the umbrella workspace and
// note that replaces in non-main go.mod files are ignored by customers
// consuming this module via `go get`, so relative paths are safe.
replace github.com/Arize-ai/openinference/go/openinference-semantic-conventions => ../openinference-semantic-conventions

replace github.com/Arize-ai/openinference/go/openinference-instrumentation => ../openinference-instrumentation

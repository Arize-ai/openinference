module github.com/Arize-ai/openinference/go/openinference-instrumentation

go 1.25.0

require (
	github.com/Arize-ai/openinference/go/openinference-semantic-conventions v0.1.0
	go.opentelemetry.io/otel v1.43.0
	go.opentelemetry.io/otel/sdk v1.43.0
	go.opentelemetry.io/otel/trace v1.43.0
)

require (
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	go.opentelemetry.io/auto/sdk v1.2.1 // indirect
	go.opentelemetry.io/otel/metric v1.43.0 // indirect
	golang.org/x/sys v0.42.0 // indirect
)

// In-repo development uses go.work; this replace makes raw `go build`
// from inside the module dir also work without the workspace. Removed
// or no-opped after the inaugural v0.1.0 tag exists in the registry.
// Customers consuming the published module via `go get` ignore this
// directive (Go only applies replaces in the *main* module's go.mod,
// not in transitive deps).
replace github.com/Arize-ai/openinference/go/openinference-semantic-conventions => ../openinference-semantic-conventions

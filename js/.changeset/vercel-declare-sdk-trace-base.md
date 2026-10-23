---
"@arizeai/openinference-vercel": patch
---

Declare `@opentelemetry/sdk-trace-base` as a dependency. The span processors import it at runtime, but it was only a devDependency, so apps had to install it themselves. OpenTelemetry JS SDK 3.0 removed `sdk-trace-base`, and the remaining 2.x releases peer `@opentelemetry/api` below 1.10, so with the latest packages the import failed (`ERR_MODULE_NOT_FOUND`) or the install failed (ERESOLVE). The package now installs its own 2.x copy.

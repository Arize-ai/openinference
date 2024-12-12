---
"@arizeai/openinference-instrumentation-langchain": major
"@arizeai/openinference-instrumentation-openai": major
"@arizeai/openinference-semantic-conventions": major
"@arizeai/openinference-vercel": major
"@arizeai/openinference-core": major
---

ESM support

Packages are now shipped as "Dual Package" meaning that ESM and CJS module resolution
should be supported for each package.

Support is described as "experimental" because opentelemetry describes support for autoinstrumenting
ESM projects as "ongoing". See https://github.com/open-telemetry/opentelemetry-js/blob/61d5a0e291db26c2af638274947081b29db3f0ca/doc/esm-support.md

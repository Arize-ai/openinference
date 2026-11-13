---
"@arizeai/openinference-instrumentation-openai": minor
---

Record the endpoint each request is sent to as `server.address` and `server.port`, taken from the client's base URL. Hosts with no known provider, such as gateways and self-hosted proxies, are recorded too, so a span shows where an OpenAI-compatible call went even when `llm.provider` can't be inferred. Only the host and port are recorded, not the path or query.

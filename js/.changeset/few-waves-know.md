---
"@arizeai/openinference-instrumentation-mcp": minor
---

This reworks the context propagation instrumentation for MCP to instrument Transport instead of Client/Server. The latter was picked at first due to concerns of the implications of propagating back from a server to a client, but with more thought, it was too much concern. Notably, if treating as two nodes talking to each other, it seems just fine to have A -> B -> A since the second A just happens to be the same server, but is not the same RPC method. This is the point notably brought up in #1524 (review) where MCP explicitly supports a server calling back to the client. This PR adds this usage to the test cases to demonstrate.

# @arizeai/openinference-instrumentation-mcp

## 0.2.4

### Patch Changes

- Updated dependencies [9d3bdb4]
  - @arizeai/openinference-core@1.0.6

## 0.2.3

### Patch Changes

- Updated dependencies [59be946]
  - @arizeai/openinference-semantic-conventions@2.1.1
  - @arizeai/openinference-core@1.0.5

## 0.2.2

### Patch Changes

- Updated dependencies [34a4159]
  - @arizeai/openinference-semantic-conventions@2.1.0
  - @arizeai/openinference-core@1.0.4

## 0.2.1

### Patch Changes

- Updated dependencies [c2ee804]
- Updated dependencies [5f904bf]
- Updated dependencies [5f90a80]
  - @arizeai/openinference-semantic-conventions@2.0.0
  - @arizeai/openinference-core@1.0.3

## 0.2.0

### Minor Changes

- 7187802: This reworks the context propagation instrumentation for MCP to instrument Transport instead of Client/Server. The latter was picked at first due to concerns of the implications of propagating back from a server to a client, but with more thought, it was too much concern. Notably, if treating as two nodes talking to each other, it seems just fine to have A -> B -> A since the second A just happens to be the same server, but is not the same RPC method. This is the point notably brought up in #1524 (review) where MCP explicitly supports a server calling back to the client. This PR adds this usage to the test cases to demonstrate.

## 0.1.0

### Minor Changes

- 63fd72d: Context propagation for MCP

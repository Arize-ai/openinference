# LLM Provider & System Attributes Coverage Report

This document shows which **OpenInference instrumentation packages** set the following tracing attributes used by **Arize/Phoenix**:

- `llm.provider` – the LLM provider (OpenAI, Anthropic, Google, etc.)
- `llm.system` – the AI system or product (OpenAI, Anthropic, VertexAI, etc.)

---

## Summary

- **Python**
  - Total packages: **28**
  - Packages with `llm.provider`: **14 / 28 (50%)**
  - Packages with `llm.system`: **6 / 28 (21%)**

- **JavaScript**
  - Total packages: **11**
  - Packages with `llm.provider`: **6 / 11 (~55%)**
  - Packages with `llm.system`: **4 / 11 (~36%)**

- **Java**
  - Total packages: **2**
  - Packages with `llm.provider`: **2 / 2 (100%)**
  - Packages with `llm.system`: **2 / 2 (100%)**

**Overall observations**
- `llm.provider` is fairly well supported across languages.
- `llm.system` support is much more limited, especially in Python and JavaScript.
- System attribution depends heavily on whether the upstream library exposes clear system-level metadata.

---

## Python – LLM Provider & System Support

This section covers **Python OpenInference instrumentation packages**.

### Packages Supporting **Both** `llm.provider` and `llm.system`

| Package |
|------|
| openinference-instrumentation-openlit |
| openinference-instrumentation-openai |
| openinference-instrumentation-openai-agents |
| openinference-instrumentation-llama-index |
| openinference-instrumentation-langchain |
| openinference-instrumentation-anthropic |

---

### Packages Supporting **Only** `llm.provider`

| Package |
|------|
| openinference-instrumentation-litellm |
| openinference-instrumentation-google-genai |
| openinference-instrumentation-google-adk |
| openinference-instrumentation-dspy |
| openinference-instrumentation-beeai |
| openinference-instrumentation-bedrock |
| openinference-instrumentation-agno |

---

### Packages Supporting **Neither** Attribute

| Package |
|------|
| openinference-instrumentation-guardrails |
| openinference-instrumentation-autogen |
| openinference-instrumentation-autogen-agentchat |
| openinference-instrumentation-crewai |
| openinference-instrumentation-haystack |
| openinference-instrumentation-instructor |
| openinference-instrumentation-portkey |
| openinference-instrumentation-groq |
| openinference-instrumentation-mistralai |
| openinference-instrumentation-mcp |
| openinference-instrumentation-promptflow |
| openinference-instrumentation-pydantic-ai |
| openinference-instrumentation-smolagents |
| openinference-instrumentation-vertexai |
| openinference-instrumentation-openllmetry |

---

## Notes (Python)

- **LangChain**
  Emits both attributes using metadata (`ls_provider`), but provider mapping is incomplete and marked with TODOs.

- **LiteLLM**
  Provider is derived from the model string. No system information is currently available.

- **Google GenAI / Google ADK**
  Provider is always set to `google`. System information is not exposed by the libraries.

- **DSPy**
  Provider is inferred from the provider class name. No system information is available.

- **BeeAI**
  Provider is read from `provider_id`. System is not set, and test coverage is missing.

- **Bedrock**
  Provider is statically set to `aws`. No system attribute is emitted.

- **Agno**
  Provider is read from the model configuration. No system information is available.

- **Guardrails**
  No provider or system attributes are emitted. Library may not expose enough runtime data yet.

- **Autogen**
  Add LLM_PROVIDER only when model is explicitly available. Do not add LLM_SYSTEM (no reliable data).

- **Autogen AgentChat**
  Treat AgentChat LLM spans as OpenAI-only. Keep LLM_PROVIDER/LLM_SYSTEM explicit, not inferred.

- **CrewAI**
  Agent framework does not currently emit provider or system attributes.

- **Portkey**
  Add LLM_PROVIDER only when model is explicitly available. Do not add LLM_SYSTEM (no reliable data).

- **Haystack / Instructor**
  Model information may exist, but provider and system are not currently extracted.

---

## JavaScript – LLM Provider & System Support

This section covers **JavaScript OpenInference instrumentation packages**.

---

### Packages Supporting **Both** `llm.provider` and `llm.system`

| Package |
|------|
| openinference-instrumentation-openai |
| openinference-instrumentation-bedrock |
| openinference-instrumentation-bedrock-agent-runtime |
| openinference-instrumentation-anthropic |

---

### Packages Supporting **Only** `llm.provider`

| Package |
|------|
| openinference-instrumentation-beeai |
| openinference-genai |

---

### Packages Supporting **Neither** Attribute

| Package |
|------|
| openinference-instrumentation-langchain |
| openinference-instrumentation-langchain-v0 |
| openinference-instrumentation-mcp |
| openinference-mastra |
| openinference-vercel |

---

### Notes (JavaScript)

- **OpenAI**
  Emits both provider and system attributes.

- **Bedrock**
  Uses Model ID to set `llm.system` (`bedrock`) and explicitly sets `llm.provider` (AWS).

- **Bedrock Agent Runtime**
  Explicitly sets both `llm.system` (`bedrock`) and `llm.provider` (AWS).

- **Anthropic**
  Explicitly sets both `llm.system` (`anthropic`) and `llm.provider` (ANTHROPIC).

- **GenAI**
  Explicitly set `llm.provider` (ATTR_GEN_AI_PROVIDER_NAME).

- **BeeAI**
  Emits provider only. System attribute is not currently set.

---

## Java – LLM Provider & System Support

This section covers **Java OpenInference instrumentation packages**.

---

### Packages Supporting **Both** `llm.provider` and `llm.system`

| Package |
|------|
| openinference-instrumentation-springAI |
| openinference-instrumentation-langchain4j |

---

### Notes (Java)

- **Spring AI**
  Explicitly sets both `llm.system` (`spring-ai`) and `llm.provider` (OPENAI).

- **LangChain4j**
  Explicitly sets both `llm.system` (`langchain4j`) and `llm.provider` (OPENAI).

---

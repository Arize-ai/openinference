# Arize / Phoenix – LLM Provider & System Support (Python)

This document shows which **Python OpenInference instrumentation packages** set the following tracing attributes used by **Arize/Phoenix**:

- `llm.provider` – the LLM provider (OpenAI, Anthropic, Google, etc.)
- `llm.system` – the AI system or product (OpenAI, Anthropic, VertexAI, etc.)

Only behavior verified in the latest analysis is included.

---

## Attribute Support

### ✅ Packages Supporting **Both** `llm.provider` and `llm.system`

| Package |
|------|
| openinference-instrumentation-openlit |
| openinference-instrumentation-openai |
| openinference-instrumentation-openai-agents |
| openinference-instrumentation-llama-index |
| openinference-instrumentation-langchain |
| openinference-instrumentation-anthropic |

---

### ⚠️ Packages Supporting **Only** `llm.provider`

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

### ❌ Packages Supporting **Neither** Attribute

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

## Notes & Improvement Areas

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

- **Autogen / Autogen AgentChat / CrewAI**  
  Agent frameworks do not currently emit provider or system attributes.

- **Haystack / Instructor / Portkey**  
  Model information may exist, but provider and system are not currently extracted.

---

## Summary

- `llm.provider` is supported by many libraries.
- `llm.system` support is limited and only available where the library exposes clear system data.
- Expanding system support depends on upstream library metadata availability.

---

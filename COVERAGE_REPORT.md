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
  Inside `_PromptCallableWrapper` (wrapping `PromptCallableBase.__call__`), I check the `PromptCallableBase` instance for a `model` or `model_name` and use `infer_llm_provider_from_model` to populate the `LLM_MODEL_NAME` and `LLM_PROVIDER` on the span.

- **Autogen**
  Whenever an agent or client instance (e.g., `BaseChatAgent` or `BaseOpenAIChatCompletionClient`) exposes a `model`, I capture it and infer the provider to populate `LLM_MODEL_NAME` and `LLM_PROVIDER`.

- **Autogen AgentChat**
  In the `_BaseOpenAIChatCompletionClientCreateWrapper` and `_BaseOpenAIChatCompletionClientCreateStreamWrapper`, I explicitly set `LLM_PROVIDER` to `openai` and `LLM_SYSTEM` to `openai`, since these wrappers always run OpenAI requests.

- **CrewAI**
  In the `_ExecuteCoreWrapper` wrapping `Task._execute_core`, I check the agent or task instance for a `model` attribute and use `infer_llm_provider_from_model` to populate `LLM_MODEL_NAME` and `LLM_PROVIDER`.

- **Haystack**
  Wrappers like `_ComponentRunWrapper` and `_AsyncComponentRunWrapper` inspect the component instance. When a `model` is exposed, I record its name and infer the provider to annotate the span with `LLM_MODEL_NAME` and `LLM_PROVIDER`.

- **Instructor**
  The `_HandleResponseWrapper` wrapping `handle_response_model` inspects the instance for `model` or `model_name` and sets `LLM_MODEL_NAME` and `LLM_PROVIDER` on the span using `infer_llm_provider_from_model`.

- **PortKey**
  Wrappers check the client or tool instance for `model` or `model_name` and use `infer_llm_provider_from_model` to set `LLM_MODEL_NAME` and `LLM_PROVIDER` in spans. System attributes are skipped since they are not reliably exposed.

- **Groq**
  In the `_CompletionsWrapper` and `_AsyncCompletionsWrapper`, I added `LLM_PROVIDER` and `LLM_MODEL_NAME` using the request parameters in `get_extra_attributes_from_request`.

- **MistralAI**
  In the `_SyncChatWrapper`, `_AsyncChatWrapper`, and `_AsyncStreamChatWrapper`, I added `LLM_PROVIDER`, `LLM_SYSTEM` and `LLM_MODEL_NAME` in the span attributes extracted from the Mistral Chat and Agents requests.

- **MCP**
  MCP does not have explicit LLM calls. It enables tracing of agent-like interactions in MCP streams.

- **Prompt Flow**
  In the `ChatFlow` class, prompt call is traced via `@trace`, capturing the model configuration and chat context. The `LLM_PROVIDER`, `LLM_SYSTEM` and `LLM_MODEL_NAME` attributes are inferred from the `AzureOpenAIModelConfiguration`.

- **Pydantic AI**
  In the `_extract_common_attributes` method, I added `LLM_PROVIDER` inference based on `GEN_AI_SYSTEM` using `_SYSTEM_TO_PROVIDER` mapping.

- **Smolagents**
  In the `_ModelWrapper`, I added `LLM_SYSTEM` using `infer_llm_system_from_model` method and `LLM_PROVIDER` using `_SYSTEM_TO_PROVIDER` mapping.

- **VertexAI**
  In the `_Wrapper`, I added update where `LLM_SYSTEM` is always set to `vertexai` and `LLM_PROVIDER` is always set to `google`.

- **OpenLLMetry**
  In the `OpenInferenceSpanProcessor`, I added checks to only allow known OpenInference values for `LLM_SYSTEM` and `LLM_PROVIDER` attributes.


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

from typing import Optional

from autogen import ConversableAgent  # type: ignore

from openinference.semconv.trace import OpenInferenceLLMProviderValues


def infer_llm_provider_from_model(
    model_name: Optional[str],
) -> Optional[OpenInferenceLLMProviderValues]:
    if not model_name:
        return None

    model = model_name.lower()

    if model.startswith(("gpt-", "gpt.", "o3", "o4")):
        return OpenInferenceLLMProviderValues.OPENAI.value

    if model.startswith(("claude-", "anthropic.claude")):
        return OpenInferenceLLMProviderValues.ANTHROPIC.value

    if model.startswith(("mistral", "mixtral")):
        return OpenInferenceLLMProviderValues.MISTRALAI.value

    if model.startswith(("command", "cohere.command")):
        return OpenInferenceLLMProviderValues.COHERE.value

    if model.startswith("gemini"):
        return OpenInferenceLLMProviderValues.GOOGLE.value

    if model.startswith("grok"):
        return OpenInferenceLLMProviderValues.XAI.value

    if model.startswith("deepseek"):
        return OpenInferenceLLMProviderValues.DEEPSEEK.value

    return None


def extract_model_from_autogen_agent(agent: ConversableAgent) -> Optional[str]:
    llm_config = getattr(agent, "llm_config", None)
    if not isinstance(llm_config, dict):
        return None

    model = llm_config.get("model")
    if isinstance(model, str):
        return model

    config_list = llm_config.get("config_list")
    if isinstance(config_list, list) and config_list:
        candidate = config_list[0]
        if isinstance(candidate, dict):
            model = candidate.get("model")
            if isinstance(model, str):
                return model

    return None

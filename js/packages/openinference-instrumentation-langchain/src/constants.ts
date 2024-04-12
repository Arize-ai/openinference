import {
  LLMAttributePostfixes,
  SemanticAttributePrefixes,
  ToolAttributePostfixes,
} from "@arizeai/openinference-semantic-conventions";

const PROMPT_TEMPLATE_PREFIX =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.prompt_template}` as const;

export const PROMPT_TEMPLATE_VARIABLES =
  `${PROMPT_TEMPLATE_PREFIX}.variables` as const;

export const PROMPT_TEMPLATE_TEMPLATE =
  `${PROMPT_TEMPLATE_PREFIX}.template` as const;

export const LLM_FUNCTION_CALL =
  `${SemanticAttributePrefixes.llm}.function_call` as const;

export const TOOL_NAME =
  `${SemanticAttributePrefixes.tool}.${ToolAttributePostfixes.name}` as const;

export const TOOL_DESCRIPTION =
  `${SemanticAttributePrefixes.tool}.${ToolAttributePostfixes.description}` as const;

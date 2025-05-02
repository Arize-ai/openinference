import json
from typing import Dict, Any, Optional, List, Tuple

from enum import Enum
    
# Import the existing OpenInference attribute classes
from openinference.semconv.trace import SpanAttributes, MessageAttributes, ToolCallAttributes
from openinference.semconv.trace import ToolAttributes, OpenInferenceMimeTypeValues

# GenAI semantic convention constants
class GenAIAttributes:
    AGENT_DESCRIPTION = "gen_ai.agent.description"
    AGENT_ID = "gen_ai.agent.id"
    AGENT_NAME = "gen_ai.agent.name"
    OPERATION_NAME = "gen_ai.operation.name"
    OUTPUT_TYPE = "gen_ai.output.type"
    REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
    REQUEST_ENCODING_FORMATS = "gen_ai.request.encoding_formats"
    REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_MODEL = "gen_ai.request.model"
    REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    REQUEST_SEED = "gen_ai.request.seed"
    REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_K = "gen_ai.request.top_k"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_MODEL = "gen_ai.response.model"
    SYSTEM = "gen_ai.system"
    TOKEN_TYPE = "gen_ai.token.type"
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    TOOL_DESCRIPTION = "gen_ai.tool.description"
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_TYPE = "gen_ai.tool.type"
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"


# Special handling for input and output values
def get_input_output_values(eventsStr: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Handle input and output values for GenAI semantic conventions.
    
    This function checks for input and output values in the GenAI attributes
    and maps them to the appropriate OpenInference attributes.

    Input values are stored in the content of the first event.
    Output values are stored in the arguments of the final tool call event.
    """

    # Parse the events
    events = json.loads(eventsStr)

    # Check that we have the input event
    if len(events) == 0:
        return None, None
    
    # Get the first event which will contain the input value
    first_event = events[0]
    input_value = first_event.get("content", None)
    
    if len(events) == 1:
        return input_value, None

    # Get the last event which will contain the output value
    last_event = events[-1]
    # Check if the last event has a message
    if "message" not in last_event:
        return input_value, None
    # Check if the last message has tool calls
    if "tool_calls" not in last_event["message"] or len(last_event["message"]["tool_calls"]) == 0:
        return input_value, None
    # Get the last tool call
    last_tool_call = last_event["message"]["tool_calls"][-1]
    # Check that the tool call has function arguments
    if "function" not in last_tool_call or "arguments" not in last_tool_call["function"]:
        return input_value, None
    # Get the function arguments
    function_args = last_tool_call["function"]["arguments"]
    return input_value, function_args

def map_gen_ai_to_openinference(gen_ai_attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Maps GenAI semantic convention attributes to OpenInference conventions.
    
    Args:
        gen_ai_attrs: Dictionary with keys following the gen_ai.* convention
        
    Returns:
        Dictionary with keys following the OpenInference convention
    """
    
    GenAI = GenAIAttributes
    open_attrs: Dict[str, Any] = {}
    
    # Direct mappings
    direct_mappings: Dict[str, Optional[str]] = {
        GenAI.AGENT_DESCRIPTION: SpanAttributes.TOOL_DESCRIPTION,
        GenAI.AGENT_ID: "tool.id",  # Not in SpanAttributes, keeping string literal
        GenAI.AGENT_NAME: SpanAttributes.TOOL_NAME,
        GenAI.OPERATION_NAME: None,  # No direct mapping
        GenAI.RESPONSE_ID: SpanAttributes.SESSION_ID,  # Approximate mapping
        GenAI.SYSTEM: SpanAttributes.LLM_SYSTEM,
        GenAI.TOOL_NAME: SpanAttributes.TOOL_NAME,
        GenAI.TOOL_DESCRIPTION: SpanAttributes.TOOL_DESCRIPTION,
        GenAI.TOOL_CALL_ID: ToolCallAttributes.TOOL_CALL_ID,
        GenAI.USAGE_INPUT_TOKENS: SpanAttributes.LLM_TOKEN_COUNT_PROMPT,
        GenAI.USAGE_OUTPUT_TOKENS: SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
    }
    
    # Handle mappings where we don't have a direct correspondence
    incomplete_mappings: List[str] = [
        GenAI.OUTPUT_TYPE,  # Could be related to output.mime_type but needs specific logic
        GenAI.REQUEST_CHOICE_COUNT,  # No direct mapping
        GenAI.REQUEST_ENCODING_FORMATS,  # No direct mapping
        GenAI.REQUEST_MAX_TOKENS,  # Goes into llm.invocation_parameters
        GenAI.REQUEST_PRESENCE_PENALTY,  # Goes into llm.invocation_parameters
        GenAI.REQUEST_SEED,  # Goes into llm.invocation_parameters
        GenAI.REQUEST_STOP_SEQUENCES,  # Goes into llm.invocation_parameters
        GenAI.REQUEST_FREQUENCY_PENALTY,  # Goes into llm.invocation_parameters
        GenAI.TOOL_TYPE,  # No clear mapping
        GenAI.RESPONSE_FINISH_REASONS,  # No direct mapping
        GenAI.TOKEN_TYPE  # No direct mapping
    ]
    
    # Process direct mappings
    for gen_ai_key, open_key in direct_mappings.items():
        if gen_ai_key in gen_ai_attrs and open_key is not None:
            open_attrs[open_key] = gen_ai_attrs[gen_ai_key]

    # Special handling for input and output values
    input_value, output_value = get_input_output_values(gen_ai_attrs.get("events", ""))
    if input_value is not None:
        open_attrs[SpanAttributes.INPUT_VALUE] = input_value
    if output_value is not None:
        open_attrs[SpanAttributes.OUTPUT_VALUE] = output_value
    
    # Special handling for model names
    if GenAI.REQUEST_MODEL in gen_ai_attrs:
        open_attrs[SpanAttributes.LLM_MODEL_NAME] = gen_ai_attrs[GenAI.REQUEST_MODEL]
    
    if GenAI.RESPONSE_MODEL in gen_ai_attrs:
        # In case both request and response model are provided, response model takes precedence
        open_attrs[SpanAttributes.LLM_MODEL_NAME] = gen_ai_attrs[GenAI.RESPONSE_MODEL]
    
    # Handle token counts
    if GenAI.USAGE_INPUT_TOKENS in gen_ai_attrs and GenAI.USAGE_OUTPUT_TOKENS in gen_ai_attrs:
        input_tokens = gen_ai_attrs[GenAI.USAGE_INPUT_TOKENS]
        output_tokens = gen_ai_attrs[GenAI.USAGE_OUTPUT_TOKENS]
        open_attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = input_tokens + output_tokens
    
    # Handle invocation parameters (needs to be converted to a JSON string)
    invocation_params: Dict[str, Any] = {}
    param_keys: List[Tuple[str, str]] = [
        (GenAI.REQUEST_TEMPERATURE, "temperature"),
        (GenAI.REQUEST_TOP_P, "top_p"),
        (GenAI.REQUEST_TOP_K, "top_k"),
        (GenAI.REQUEST_MAX_TOKENS, "max_tokens"),
        (GenAI.REQUEST_FREQUENCY_PENALTY, "frequency_penalty"),
        (GenAI.REQUEST_PRESENCE_PENALTY, "presence_penalty"),
        (GenAI.REQUEST_STOP_SEQUENCES, "stop"),
        (GenAI.REQUEST_SEED, "seed")
    ]
    
    for gen_ai_key, param_name in param_keys:
        if gen_ai_key in gen_ai_attrs:
            invocation_params[param_name] = gen_ai_attrs[gen_ai_key]
    
    # Add model name to invocation params if available
    if GenAI.REQUEST_MODEL in gen_ai_attrs:
        invocation_params["model_name"] = gen_ai_attrs[GenAI.REQUEST_MODEL]
    
    # Convert to JSON string if we have any invocation parameters
    if invocation_params:
        open_attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS] = json.dumps(invocation_params)
    
    # Map output type to mime type (approximate mapping)
    if GenAI.OUTPUT_TYPE in gen_ai_attrs:
        output_type = gen_ai_attrs[GenAI.OUTPUT_TYPE]
        mime_type_map = {
            "text": OpenInferenceMimeTypeValues.TEXT.value,
            "json": OpenInferenceMimeTypeValues.JSON.value,
            "image": "image/png",  # Default image type, not in OpenInferenceMimeTypeValues
            "speech": "audio/mpeg"  # Default audio type, not in OpenInferenceMimeTypeValues
        }
        if output_type in mime_type_map:
            open_attrs[SpanAttributes.OUTPUT_MIME_TYPE] = mime_type_map[output_type]
    
    return open_attrs


"""
Strands to OpenInference Converter for Arize AI

This module provides a span processor that converts Strands telemetry data
to OpenInference format for compatibility with Arize AI.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

class StrandsToOpenInferenceProcessor(SpanProcessor):
    """
    SpanProcessor that converts Strands telemetry attributes to OpenInference format
    for compatibility with Arize AI.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the processor.
        
        Args:
            debug: Whether to log debug information
        """
        super().__init__()
        self.debug = debug
        self.processed_spans = set()
        self.current_cycle_id = None
        self.span_hierarchy = {}

    def on_start(self, span, parent_context=None):
        """Called when a span is started. Track span hierarchy."""
        span_id = span.get_span_context().span_id
        parent_id = None
        
        if parent_context and hasattr(parent_context, 'span_id'):
            parent_id = parent_context.span_id
        elif span.parent and hasattr(span.parent, 'span_id'):
            parent_id = span.parent.span_id
            
        self.span_hierarchy[span_id] = {
            'name': span.name,
            'span_id': span_id,
            'parent_id': parent_id,
            'start_time': datetime.now().isoformat()
        }

    def on_end(self, span: Span):
        """
        Called when a span ends. Transform the span attributes from Strands format
        to OpenInference format.
        """
        if not hasattr(span, '_attributes') or not span._attributes:
            return

        original_attrs = dict(span._attributes)
        span_id = span.get_span_context().span_id
        
        if span_id in self.span_hierarchy:
            self.span_hierarchy[span_id]['attributes'] = original_attrs
        
        try:
            if "event_loop.cycle_id" in original_attrs:
                self.current_cycle_id = original_attrs.get("event_loop.cycle_id")
            
            transformed_attrs = self._transform_attributes(original_attrs, span)
            span._attributes.clear()
            span._attributes.update(transformed_attrs)
            self.processed_spans.add(span_id)
            
            if self.debug:
                logger.info(f"Transformed span '{span.name}': {len(original_attrs)} -> {len(transformed_attrs)} attributes")
                
        except Exception as e:
            logger.error(f"Failed to transform span '{span.name}': {e}", exc_info=True)
            span._attributes.clear()
            span._attributes.update(original_attrs)

    def _transform_attributes(self, attrs: Dict[str, Any], span: Span) -> Dict[str, Any]:
        """
        Transform Strands attributes to OpenInference format.
        """
        result = {}
        span_kind = self._determine_span_kind(span, attrs)
        result["openinference.span.kind"] = span_kind
        self._set_graph_node_attributes(span, attrs, result)
        prompt = attrs.get("gen_ai.prompt")
        completion = attrs.get("gen_ai.completion")
        model_id = attrs.get("gen_ai.request.model")
        agent_name = attrs.get("agent.name") or attrs.get("gen_ai.agent.name")

        if model_id:
            result["llm.model_name"] = model_id
            result["gen_ai.request.model"] = model_id
            
        if agent_name:
            result["llm.system"] = "strands-agents"
            result["llm.provider"] = "strands-agents"
        
        if "tag.tags" in attrs:
            tags = attrs.get("tag.tags")
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, str):
                        result[f"tag.{tag}"] = str(tag)
            elif isinstance(tags, str):
                result[f"tag.{tags}"] = str(tags)
        
        # Handle different span types
        if span_kind == "LLM":
            self._handle_chain_and_llm_span(attrs, result, prompt, completion)
        elif span_kind == "TOOL":
            self._handle_tool_span(attrs, result)
        elif span_kind == "AGENT":
            self._handle_agent_span(attrs, result, prompt)
        elif span_kind == "CHAIN":
            self._handle_chain_and_llm_span(attrs, result, prompt, completion)
        
        # Handle token usage
        self._map_token_usage(attrs, result)
        
        important_attrs = [
            "session.id", "user.id", "llm.prompt_template.template",
            "llm.prompt_template.version", "llm.prompt_template.variables",
            "gen_ai.event.start_time", "gen_ai.event.end_time"
        ]
        
        for key in important_attrs:
            if key in attrs:
                result[key] = attrs[key]
        
        self._add_metadata(attrs, result)    
        return result
    
    def _determine_span_kind(self, span: Span, attrs: Dict[str, Any]) -> str:
        """Determine the OpenInference span kind."""
        span_name = span.name
        
        if "Model invoke" in span_name:
            return "LLM"
        elif span_name.startswith("Tool:"):
            return "TOOL"
        elif attrs.get("agent.name") or attrs.get("gen_ai.agent.name"):
            return "AGENT"
        elif "Cycle" in span_name:
            return "CHAIN"
        return "CHAIN"
    
    def _set_graph_node_attributes(self, span: Span, attrs: Dict[str, Any], result: Dict[str, Any]):
        """
        Set graph node attributes for Arize visualization.
        Hierarchy: Agent -> Cycles -> (LLMs and/or Tools)
        """
        span_name = span.name
        span_kind = result["openinference.span.kind"]        
        span_id = span.get_span_context().span_id
        
        # Get parent information from span hierarchy
        span_info = self.span_hierarchy.get(span_id, {})
        parent_id = span_info.get('parent_id')
        parent_info = self.span_hierarchy.get(parent_id, {}) if parent_id else {}
        parent_name = parent_info.get('name', '')
        
        if span_kind == "AGENT":
            result["graph.node.id"] = "strands_agent"
        
        if span_kind == "CHAIN" and "Cycle " in span_name:
            cycle_id = span_name.replace("Cycle ", "").strip()
            result["graph.node.id"] = f"cycle_{cycle_id}"
            result["graph.node.parent_id"] = "strands_agent"
        
        if span_kind == "LLM":
            result["graph.node.id"] = f"llm_{span_id}"
            if parent_name.startswith("Cycle"):
                cycle_id = parent_name.replace("Cycle ", "").strip()
                result["graph.node.parent_id"] = f"cycle_{cycle_id}"
            else:
                result["graph.node.parent_id"] = "strands_agent"
        
        if span_kind == "TOOL":
            tool_name = attrs.get("tool.name", span_name.replace("Tool: ", "").strip())
            tool_id = attrs.get("tool.id", span_id)
            result["graph.node.id"] = f"tool_{tool_name}_{tool_id}"
            if parent_name.startswith("Cycle "):
                cycle_id = parent_name.replace("Cycle ", "").strip()
                result["graph.node.parent_id"] = f"cycle_{cycle_id}"
            else:
                result["graph.node.parent_id"] = "strands_agent"

        if self.debug:
            logger.info(f"span_name: {span_name}")
            logger.info(f"span_kind: {span_kind}")
            logger.info(f"span_id: {span_id}")
            logger.info(f"span_info: {span_info}")
            logger.info(f"parent_id: {parent_id}")
            logger.info(f"parent_info: {parent_info}")
            logger.info(f"parent_name: {parent_name}")
            logger.info("==========================")
            logger.info(f"Span: {span_name} || (ID: {span_id})")
            logger.info(f"  Parent: {parent_name} || (ID: {parent_id})")
            logger.info(f"  Graph Node: {result.get('graph.node.id')} -> Parent: {result.get('graph.node.parent_id')}")

    def _handle_chain_and_llm_span(self, attrs: Dict[str, Any], result: Dict[str, Any], prompt: Any, completion: Any):
        """Handle LLM-specific attributes."""
        if prompt:
            self._map_messages(prompt, result, is_input=True)
        
        if completion:
            self._map_messages(completion, result, is_input=False)
        
        self._add_input_output_values(attrs, result)
        self._map_invocation_parameters(attrs, result)
    
    def _handle_tool_span(self, attrs: Dict[str, Any], result: Dict[str, Any]):
        """Handle tool-specific attributes."""
        if tool_name := attrs.get("tool.name"):
            result["tool.name"] = tool_name
        
        if tool_id := attrs.get("tool.id"):
            result["tool.id"] = tool_id
        
        if tool_status := attrs.get("tool.status"):
            result["tool.status"] = str(tool_status)
        
        if tool_description := attrs.get("tool.description"):
            result["tool.description"] = tool_description
        
        if tool_params := attrs.get("tool.parameters"):
            result["tool.parameters"] = self._serialize_value(tool_params)
            tool_call = {
                "tool_call.id": attrs.get("tool.id", ""),
                "tool_call.function.name": attrs.get("tool.name", ""),
                "tool_call.function.arguments": self._serialize_value(tool_params)
            }
            
            input_message = {
                "message.role": "assistant",
                "message.tool_calls": [tool_call]
            }
            result["llm.input_messages"] = json.dumps([input_message], separators=(",", ":"))
            result["llm.input_messages.0.message.role"] = "assistant"
            result["tool_call.id"] = attrs.get("tool.id", "")
            result["tool_call.function.name"] = attrs.get("tool.name", "")
            result["tool_call.function.arguments"] = self._serialize_value(tool_params)
        
            for key, value in tool_call.items():
                result[f"llm.input_messages.0.message.tool_calls.0.{key}"] = value
        
        # Map tool result
        if tool_result := attrs.get("tool.result"):
            result["tool.result"] = self._serialize_value(tool_result)
            tool_result_content = tool_result
            if isinstance(tool_result, dict):
                tool_result_content = tool_result.get("content", tool_result)
                if "error" in tool_result:
                    result["tool.error"] = self._serialize_value(tool_result.get("error"))

            output_message = {
                "message.role": "tool",
                "message.content": self._serialize_value(tool_result_content),
                "message.tool_call_id": attrs.get("tool.id", "")
            }

            if tool_name := attrs.get("tool.name"):
                output_message["message.name"] = tool_name
            result["llm.output_messages"] = json.dumps([output_message], separators=(",", ":"))
            result["llm.output_messages.0.message.role"] = "tool"
            result["llm.output_messages.0.message.content"] = self._serialize_value(tool_result_content)
            result["llm.output_messages.0.message.tool_call_id"] = attrs.get("tool.id", "")
            
            if tool_name:
                result["llm.output_messages.0.message.name"] = tool_name

        if start_time := attrs.get("gen_ai.event.start_time"):
            result["tool.start_time"] = start_time
        
        if end_time := attrs.get("gen_ai.event.end_time"):
            result["tool.end_time"] = end_time

        tool_metadata = {}
        for key, value in attrs.items():
            if key.startswith("tool.") and key not in ["tool.name", "tool.id", "tool.parameters", "tool.result", "tool.status"]:
                tool_metadata[key] = self._serialize_value(value)
        
        if tool_metadata:
            result["tool.metadata"] = json.dumps(tool_metadata, separators=(",", ":"))
    
    def _handle_agent_span(self, attrs: Dict[str, Any], result: Dict[str, Any], prompt: Any):
        """Handle agent-specific attributes."""
        result["llm.system"] = "strands-agents"
        result["llm.provider"] = "strands-agents"
        
        if tools := (attrs.get("agent.tools") or attrs.get("gen_ai.agent.tools")):
            self._map_tools(tools, result)
        
        if prompt:
            input_message = {
                "message.role": "user",
                "message.content": str(prompt)
            }
            result["llm.input_messages"] = json.dumps([input_message], separators=(",", ":"))
            result["input.value"] = str(prompt)
            result["llm.input_messages.0.message.role"] = "user"
            result["llm.input_messages.0.message.content"] = str(prompt)
        self._add_input_output_values(attrs, result)  
    
    def _map_messages(self, messages_data: Any, result: Dict[str, Any], is_input: bool):
        """Map Strands messages to OpenInference message format."""
        key_prefix = "llm.input_messages" if is_input else "llm.output_messages"
        
        if isinstance(messages_data, str):
            try:
                messages_data = json.loads(messages_data)
            except json.JSONDecodeError:
                messages_data = [{"role": "user" if is_input else "assistant", "content": messages_data}]
        
        messages_list = self._normalize_messages(messages_data)
        result[key_prefix] = json.dumps(messages_list, separators=(",", ":"))

        for idx, msg in enumerate(messages_list):
            if not isinstance(msg, dict):
                continue
                
            for sub_key, sub_val in msg.items():
                clean_key = sub_key.replace("message.", "") if sub_key.startswith("message.") else sub_key
                dotted_key = f"{key_prefix}.{idx}.message.{clean_key}"
                
                if clean_key == "tool_calls" and isinstance(sub_val, list):
                    # Handle tool calls with proper structure
                    for tool_idx, tool_call in enumerate(sub_val):
                        if isinstance(tool_call, dict):
                            for tool_key, tool_val in tool_call.items():
                                tool_dotted_key = f"{key_prefix}.{idx}.message.tool_calls.{tool_idx}.{tool_key}"
                                result[tool_dotted_key] = self._serialize_value(tool_val)
                else:
                    result[dotted_key] = self._serialize_value(sub_val)
    
    def _normalize_messages(self, data: Any) -> List[Dict[str, Any]]:
        """Normalize messages data to a consistent list format."""
        if isinstance(data, list):
            return [self._normalize_message(m) for m in data]
        
        if isinstance(data, dict) and all(isinstance(k, str) and k.isdigit() for k in data.keys()):
            ordered = sorted(data.items(), key=lambda kv: int(kv[0]))
            return [self._normalize_message(v) for _, v in ordered]
        
        if isinstance(data, dict):
            return [self._normalize_message(data)]
        
        return [{"message.role": "user", "message.content": str(data)}]
    
    def _normalize_message(self, msg: Any) -> Dict[str, Any]:
        """Normalize a single message to OpenInference format."""
        if not isinstance(msg, dict):
            return {"message.role": "user", "message.content": str(msg)}
        
        result = {}
        for key in ["role", "content", "name", "tool_call_id", "finish_reason"]:
            if key in msg:
                result[f"message.{key}"] = msg[key]
        
        if "toolUse" in msg and isinstance(msg["toolUse"], list):
            tool_calls = []
            for tool_use in msg["toolUse"]:
                tool_calls.append({
                    "tool_call.id": tool_use.get("toolUseId", ""),
                    "tool_call.function.name": tool_use.get("name", ""),
                    "tool_call.function.arguments": json.dumps(tool_use.get("input", {}))
                })
            result["message.tool_calls"] = tool_calls
        
        return result
    
    def _map_tools(self, tools_data: Any, result: Dict[str, Any]):
        """Map tools from Strands to OpenInference format."""
        if isinstance(tools_data, str):
            try:
                tools_data = json.loads(tools_data)
            except json.JSONDecodeError:
                return
        
        if not isinstance(tools_data, list):
            return
        
        openinf_tools = []
        for tool in tools_data:
            if isinstance(tool, dict):
                openinf_tool = {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                }
                
                if "parameters" in tool:
                    openinf_tool["parameters"] = tool["parameters"]
                elif "input_schema" in tool:
                    openinf_tool["parameters"] = tool["input_schema"]
                
                openinf_tools.append(openinf_tool)
        
        if openinf_tools:
            for idx, tool in enumerate(openinf_tools):
                for key, value in tool.items():
                    dotted_key = f"llm.tools.{idx}.{key}"
                    result[dotted_key] = self._serialize_value(value)
    
    def _map_token_usage(self, attrs: Dict[str, Any], result: Dict[str, Any]):
        """Map token usage metrics."""
        token_mappings = [
            ("gen_ai.usage.prompt_tokens", "llm.token_count.prompt"),
            ("gen_ai.usage.completion_tokens", "llm.token_count.completion"),
            ("gen_ai.usage.total_tokens", "llm.token_count.total"),
        ]
        
        for strands_key, openinf_key in token_mappings:
            if value := attrs.get(strands_key):
                result[openinf_key] = value
    
    def _map_invocation_parameters(self, attrs: Dict[str, Any], result: Dict[str, Any]):
        """Map invocation parameters."""
        params = {}
        
        param_mappings = {
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
        }
        
        for key, param_key in param_mappings.items():
            if key in attrs:
                params[param_key] = attrs[key]
        
        if params:
            result["llm.invocation_parameters"] = json.dumps(params, separators=(",", ":"))
    
    def _add_input_output_values(self, attrs: Dict[str, Any], result: Dict[str, Any]):
        """Add input.value and output.value for Arize compatibility."""
        span_kind = result.get("openinference.span.kind")
        model_name = result.get("llm.model_name") or attrs.get("gen_ai.request.model") or "unknown"
        invocation_params = {}
        if "llm.invocation_parameters" in result:
            try:
                invocation_params = json.loads(result["llm.invocation_parameters"])
            except:
                pass
        
        if span_kind == "LLM":
            if "llm.input_messages" in result:
                try:
                    input_messages = json.loads(result["llm.input_messages"])
                    if input_messages:
                        input_structure = {
                            "messages": input_messages,
                            "model": model_name
                        }
                        if max_tokens := invocation_params.get("max_tokens"):
                            input_structure["max_tokens"] = max_tokens
                        
                        result["input.value"] = json.dumps(input_structure, separators=(",", ":"))
                        result["input.mime_type"] = "application/json"
                except:
                    if prompt_content := result.get("llm.input_messages.0.message.content"):
                        result["input.value"] = prompt_content
                        result["input.mime_type"] = "application/json"

            if "llm.output_messages" in result:
                try:
                    output_messages = json.loads(result["llm.output_messages"])
                    if output_messages and len(output_messages) > 0:
                        first_msg = output_messages[0]
                        content = first_msg.get("message.content", "")
                        role = first_msg.get("message.role", "assistant")
                        finish_reason = first_msg.get("message.finish_reason", "stop")
                        output_structure = {
                            "id": attrs.get("gen_ai.response.id"),
                            "choices": [{
                                "finish_reason": finish_reason,
                                "index": 0,
                                "logprobs": None,
                                "message": {
                                    "content": content,
                                    "role": role,
                                    "refusal": None,
                                    "annotations": []
                                }
                            }],
                            "model": model_name,
                            "usage": {
                                "completion_tokens": result.get("llm.token_count.completion"),
                                "prompt_tokens": result.get("llm.token_count.prompt"),
                                "total_tokens": result.get("llm.token_count.total")
                            }
                        }
                        
                        result["output.value"] = json.dumps(output_structure, separators=(",", ":"))
                        result["output.mime_type"] = "application/json"
                except:
                    if completion_content := result.get("llm.output_messages.0.message.content"):
                        result["output.value"] = completion_content
                        result["output.mime_type"] = "application/json"
                    
        elif span_kind == "AGENT":
            if prompt := attrs.get("gen_ai.prompt"):
                result["input.value"] = str(prompt)
                result["input.mime_type"] = "text/plain"

            if completion := attrs.get("gen_ai.completion"):
                result["output.value"] = str(completion)
                result["output.mime_type"] = "text/plain"
                
        elif span_kind == "TOOL":
            if tool_params := attrs.get("tool.parameters"):
                if isinstance(tool_params, str):
                    result["input.value"] = tool_params
                else:
                    result["input.value"] = json.dumps(tool_params, separators=(",", ":"))
                result["input.mime_type"] = "application/json"
            
            if tool_result := attrs.get("tool.result"):
                if isinstance(tool_result, str):
                    result["output.value"] = tool_result
                else:
                    result["output.value"] = json.dumps(tool_result, separators=(",", ":"))
                result["output.mime_type"] = "application/json"
                
        elif span_kind == "CHAIN":
            if prompt := attrs.get("gen_ai.prompt"):
                if isinstance(prompt, str):
                    result["input.value"] = prompt
                else:
                    result["input.value"] = json.dumps(prompt, separators=(",", ":"))
                result["input.mime_type"] = "text/plain" if isinstance(prompt, str) else "application/json"
            
            if completion := attrs.get("gen_ai.completion"):
                if isinstance(completion, str):
                    result["output.value"] = completion  
                else:
                    result["output.value"] = json.dumps(completion, separators=(",", ":"))
                result["output.mime_type"] = "text/plain" if isinstance(completion, str) else "application/json"
    
    def _add_metadata(self, attrs: Dict[str, Any], result: Dict[str, Any]):
        """Add remaining attributes to metadata."""
        metadata = {}
        skip_keys = {"gen_ai.prompt", "gen_ai.completion", "agent.tools", "gen_ai.agent.tools"}
        
        for key, value in attrs.items():
            if key not in skip_keys and key not in result:
                metadata[key] = self._serialize_value(value)
        
        if metadata:
            result["metadata"] = json.dumps(metadata, separators=(",", ":"))
    
    def _serialize_value(self, value: Any) -> Any:
        """Ensure a value is serializable."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        
        try:
            return json.dumps(value, separators=(",", ":"))
        except (TypeError, OverflowError):
            return str(value)

    def shutdown(self):
        """Called when the processor is shutdown."""
        pass

    def force_flush(self, timeout_millis=None):
        """Called to force flush."""
        return True
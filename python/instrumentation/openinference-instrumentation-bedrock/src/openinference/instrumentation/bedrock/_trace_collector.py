"""Trace manager module for managing trace spans and nodes in Bedrock instrumentation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from openinference.instrumentation.bedrock._attribute_extractor import AttributeExtractor

logger = logging.getLogger(__name__)


def generate_unique_trace_id(event_type: str, trace_id: str) -> str:
    """
    Generate a unique trace ID by combining event type and a portion of the original trace ID.

    Args:
        event_type: The type of event (e.g., 'preProcessingTrace')
        trace_id: The original trace ID

    Returns:
        A unique trace ID string
    """
    if "guardrail" in trace_id:
        # if guardrail, use the first 7 parts of the trace id in order to differentiate
        # between pre and post guardrail; it will look something like this:
        # 4ce64021-13b2-23c5-9d70-aaefe8881138-guardrail-pre-0
        return f"{event_type}_{'-'.join(trace_id.split('-')[:7])}"
    return f"{event_type}_{'-'.join(trace_id.split('-')[:5])}"


class TraceSpan:
    """
    Represents a span within a trace that contains chunks of data.

    A TraceSpan is a logical unit of work within a trace that can contain
    multiple chunks of data and can be linked to parent and child nodes.
    """

    def __init__(self, span_type: str):
        """
        Initialize a new TraceSpan.

        Args:
            span_type: The type of span (e.g., 'modelInvocationInput')
        """
        self.chunks: List[Dict[str, Any]] = []
        self.span_type: str = span_type
        self.children_nodes: List["TraceNode"] = []
        self.parent_node: Optional["TraceNode"] = None

    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Add a chunk of data to this span.

        Args:
            chunk: A dictionary containing trace data
        """
        self.chunks.append(chunk)

    def add_child_node(self, trace_node: "TraceNode") -> None:
        """
        Add a child node to this span.

        Args:
            trace_node: The TraceNode to add as a child
        """
        self.children_nodes.append(trace_node)

    def set_parent_node(self, parent_node: "TraceNode") -> None:
        """
        Set the parent node for this span.

        Args:
            parent_node: The TraceNode to set as parent
        """
        self.parent_node = parent_node


class TraceNode:
    """
    Represents a node in the trace tree that can contain multiple spans.

    A TraceNode is a container for spans and chunks that represents a
    logical component in the trace hierarchy, such as an agent or a model.
    """

    def __init__(self, trace_id: str, event_type: Optional[str] = None):
        """
        Initialize a new TraceNode.

        Args:
            trace_id: The unique identifier for this trace node
            event_type: The type of event this node represents (e.g., 'agent-collaborator')
        """
        self.node_trace_id: str = trace_id
        self.node_type: Optional[str] = event_type
        self.spans: List[Union[TraceSpan, "TraceNode"]] = []
        self.parent_trace_node: Optional["TraceNode"] = None
        self.current_span: Optional[TraceSpan] = None
        self.chunks: List[Dict[str, Any]] = []

    def add_span(self, span: Union[TraceSpan, "TraceNode"]) -> None:
        """
        Add a span to this node and set it as the current span.

        Args:
            span: The span to add (either a TraceSpan or another TraceNode)
        """
        self.spans.append(span)
        if isinstance(span, TraceSpan):
            self.current_span = span

    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Add a chunk of data directly to this node.

        Args:
            chunk: A dictionary containing trace data
        """
        self.chunks.append(chunk)


class TraceStack:
    """
    Manages a stack of TraceNodes for tracking the current execution context.

    The TraceStack provides operations to push, pop, and fetch nodes,
    maintaining the hierarchy of trace nodes during processing.
    """

    def __init__(self) -> None:
        """Initialize an empty TraceStack."""
        self.stack: List[TraceNode] = []
        self.current_node: Optional[TraceNode] = None

    def push(self, node: TraceNode) -> None:
        """
        Push a node onto the stack.

        Args:
            node: The TraceNode to push onto the stack
        """
        self.stack.append(node)

    def pop(self) -> Optional[TraceNode]:
        """
        Pop the top node from the stack.

        Returns:
            The popped TraceNode or None if the stack is empty
        """
        if self.stack:
            logger.debug(f"Popping node from stack. Stack size before: {len(self.stack)}")
            node = self.stack.pop()
            logger.debug(f"Stack size after pop: {len(self.stack)}")
            return node
        else:
            logger.debug("Attempted to pop from empty stack")
            return None

    def fetch(self, trace_id: str) -> Optional[TraceNode]:
        """
        Find a node in the stack by its trace ID.

        Args:
            trace_id: The trace ID to search for

        Returns:
            The matching TraceNode or None if not found
        """
        for node in self.stack or []:
            if node.node_trace_id == trace_id:
                return node
        return None

    @property
    def head(self) -> Optional[TraceNode]:
        """
        Get the node at the top of the stack without removing it.

        Returns:
            The top TraceNode or None if the stack is empty
        """
        return self.stack[-1] if self.stack else None


class TraceCollector:
    """
    Collects and organizes trace data into a hierarchical structure.

    The TraceCollector processes incoming trace data, creates appropriate
    TraceNodes and TraceSpans, and maintains the relationships between them.
    """

    def __init__(self) -> None:
        """Initialize a new TraceCollector with a default root node."""
        default_trace_id = "default-parent-node"
        default_agent_name = "bedrock_agent.invoke_agent"
        self.initial_node = TraceNode(default_trace_id, default_agent_name)
        self.trace_stack: TraceStack = TraceStack()
        self.trace_stack.push(self.initial_node)
        self.trace_nodes: dict[str, TraceNode] = {default_trace_id: self.initial_node}
        self.trace_ids: List[str] = [self.initial_node.node_trace_id]

    def _handle_chunk_for_current_node(
        self, node: TraceNode, chunk_type: str, trace_data: Dict[str, Any]
    ) -> None:
        """
        Handle a chunk of data for the current node based on the chunk type.

        Args:
            node: The TraceNode to handle the chunk for
            chunk_type: The type of chunk being processed
            trace_data: The trace data to add to the node or span
        """
        if chunk_type not in ["invocationInput", "modelInvocationInput"]:
            # Add chunk to trace node as well, useful for propogating metadata to the parent node
            node.add_chunk(trace_data)
            if node.current_span:
                node.current_span.add_chunk(trace_data)
            else:
                self._start_new_span(node, chunk_type, trace_data)
        else:
            self._start_new_span(node, chunk_type, trace_data)

    def _handle_new_trace_node(
        self,
        parent_node: TraceNode,
        node_trace_id: str,
        event_type: str,
        chunk_type: str,
        trace_data: Dict[str, Any],
    ) -> None:
        """
        Create and handle a new trace node.

        This method creates a new TraceNode based on the provided parameters,
        adds it to the appropriate parent node, and updates the trace stack.

        Args:
            parent_node: The parent TraceNode
            node_trace_id: The unique ID for the new trace node
            event_type: The type of event for the new node
            chunk_type: The type of chunk being processed
            trace_data: The trace data to add to the new node
        """
        chunk_data = trace_data.get(event_type, {}).get(chunk_type, {})
        if "agentCollaboratorInvocationInput" in chunk_data:
            trace_node = TraceNode(node_trace_id, "agent-collaborator")
            trace_node.chunks.append(trace_data)
        elif event_type == "failureTrace":
            trace_node = TraceNode(node_trace_id, event_type)
            trace_node.chunks.append(trace_data)
        elif event_type == "guardrailTrace":
            trace_node = TraceNode(node_trace_id, event_type)
            trace_node.add_chunk(trace_data)
            trace_span = TraceSpan(chunk_type)
            trace_span.add_chunk(trace_data)
            trace_node.add_span(trace_span)
        else:
            trace_node = TraceNode(node_trace_id, event_type)
            trace_node.add_chunk(trace_data)
            trace_span = TraceSpan(chunk_type)
            trace_span.add_chunk(trace_data)
            trace_span.parent_node = parent_node  # This is child for the Agent Span
            trace_node.add_span(trace_span)

        trace_node.parent_trace_node = parent_node
        parent_node.add_span(trace_node)
        self.trace_stack.push(trace_node)
        self.trace_nodes[node_trace_id] = trace_node
        self.trace_ids.append(node_trace_id)

    def _handle_existing_trace_node(
        self,
        parent_node: TraceNode,
        node_trace_id: str,
        chunk_type: str,
        trace_data: Dict[str, Any],
    ) -> None:
        """
        Handle an existing trace node.

        This method processes trace data for a node that already exists in the trace stack.

        Args:
            parent_node: The parent TraceNode
            node_trace_id: The ID of the existing trace node
            chunk_type: The type of chunk being processed
            trace_data: The trace data to add to the node
        """
        if (
            parent_node
            and parent_node.parent_trace_node
            and node_trace_id == parent_node.parent_trace_node.node_trace_id
        ):
            self.trace_stack.pop()

        trace_node = self.trace_stack.fetch(node_trace_id)
        if trace_node:
            self._handle_chunk_for_current_node(trace_node, chunk_type, trace_data)
        else:
            logger.warning(f"Could not find trace node with ID {node_trace_id}")

    def _handle_chunk_for_parent_node(
        self, parent_node: TraceNode, chunk_type: str, trace_data: Dict[str, Any]
    ) -> None:
        """
        Handle a chunk of data for the parent node.

        Args:
            parent_node: The parent TraceNode to handle the chunk for
            chunk_type: The type of chunk being processed
            trace_data: The trace data to add to the parent node
        """
        if parent_node.current_span:
            parent_node.current_span.add_chunk(trace_data)
        else:
            self._start_new_span(parent_node, chunk_type, trace_data)

    @staticmethod
    def _start_new_span(node: TraceNode, chunk_type: str, trace_data: Dict[str, Any]) -> None:
        """
        Start a new span for a node with the given chunk type and data.

        Args:
            node: The TraceNode to create a new span for
            chunk_type: The type of chunk for the new span
            trace_data: The trace data to add to the new span
        """
        trace_span = TraceSpan(chunk_type)
        trace_span.add_chunk(trace_data)
        trace_span.parent_node = node
        node.add_span(trace_span)

    def collect(self, obj: Dict[str, Any]) -> TraceNode:
        """
        Process and collect trace data from the provided object.

        This method is the main entry point for processing trace data. It extracts
        relevant information from the input object, determines the appropriate
        trace node to handle the data, and delegates to specialized methods for
        different processing scenarios.

        Args:
            obj: A dictionary containing trace data to process

        Returns:
            The current TraceNode after processing
        """

        # Extract trace data from the input object
        trace_data: Dict[str, Any] = self._extract_trace_data(obj)

        # Get the current parent node from the stack
        parent_trace_node = self.trace_stack.head
        if not parent_trace_node:
            logger.warning("No parent trace node found in stack")
            return self.initial_node

        # Extract event type, chunk type and generate trace ID
        event_type = AttributeExtractor.get_event_type(trace_data)
        node_trace_id = generate_unique_trace_id(
            event_type, AttributeExtractor.extract_trace_id(trace_data)
        )
        chunk_type = AttributeExtractor.get_chunk_type(trace_data.get(event_type, {}))
        if event_type == "guardrailTrace":
            chunk_type = "guardrail"

        # Initialize variables
        agent_node_trace_id = ""

        # Generate agent node trace ID for agent collaborator invocations
        if trace_data.get(event_type, {}).get(chunk_type, {}).get(
            "agentCollaboratorInvocationInput"
        ) or trace_data.get(event_type, {}).get(chunk_type, {}).get(
            "agentCollaboratorInvocationOutput"
        ):
            agent_node_trace_id = f"{node_trace_id}-agent"

        # Process the trace data based on its characteristics
        return self._process_trace_data(
            trace_data,
            parent_trace_node,
            node_trace_id,
            agent_node_trace_id,
            event_type,
            chunk_type,
        )

    @classmethod
    def _extract_trace_data(cls, obj: dict[str, Any]) -> dict[str, Any]:
        """
        Extract trace data from the bedrock trace object.

        Args:
            obj: The input object containing trace data

        Returns:
            Extracted trace data as a dictionary
        """
        if "trace" in obj:
            return obj["trace"].get("trace") or {}
        else:
            return obj

    def _process_trace_data(
        self,
        trace_data: Dict[str, Any],
        parent_trace_node: TraceNode,
        node_trace_id: str,
        agent_node_trace_id: str,
        event_type: str,
        chunk_type: str,
    ) -> TraceNode:
        """
        Process the trace data based on its characteristics.

        Args:
            trace_data: The trace data to process
            parent_trace_node: The current parent trace node
            node_trace_id: The generated trace ID for this data
            agent_node_trace_id: The agent node trace ID if applicable
            event_type: The type of event in the trace data
            chunk_type: The type of chunk in the trace data

        Returns:
            The current TraceNode after processing
        """
        if node_trace_id not in self.trace_ids:
            # Handle new trace node
            self._handle_new_trace_node_scenario(
                parent_trace_node, node_trace_id, event_type, chunk_type, trace_data
            )
        elif parent_trace_node and parent_trace_node.node_trace_id in [
            node_trace_id,
            agent_node_trace_id,
        ]:
            # Handle case where parent node matches the trace ID
            self._handle_parent_node_match_scenario(
                parent_trace_node, agent_node_trace_id, event_type, chunk_type, trace_data
            )
        elif (
            parent_trace_node.parent_trace_node
            and parent_trace_node.parent_trace_node.node_trace_id == node_trace_id
        ):
            # Handle case where parent's parent node matches the trace ID
            # This will be invoked while changing the one trace type to
            # another(preProcessing to orchestration)
            self.trace_stack.pop()
            parent_trace_node = self.trace_stack.head or self.initial_node
            if parent_trace_node:
                self._handle_chunk_for_current_node(parent_trace_node, chunk_type, trace_data)
        elif node_trace_id in self.trace_ids:
            # Handle case where trace ID exists but parent node doesn't match
            if trace_node := self.trace_nodes.get(node_trace_id):
                parent_trace_node = trace_node
            self._handle_existing_trace_id_scenario(
                parent_trace_node, agent_node_trace_id, event_type, chunk_type, trace_data
            )
        else:
            # Log unexpected case
            logger.warning(f"Unhandled trace data scenario for trace ID: {node_trace_id}")

        return self.trace_stack.head or self.initial_node

    def _handle_new_trace_node_scenario(
        self,
        parent_trace_node: TraceNode,
        node_trace_id: str,
        event_type: str,
        chunk_type: str,
        trace_data: Dict[str, Any],
    ) -> None:
        """
        Handle scenario where a new trace node needs to be created.

        Args:
            parent_trace_node: The current parent trace node
            node_trace_id: The trace ID for the new node
            event_type: The type of event
            chunk_type: The type of chunk
            trace_data: The trace data to process
        """
        excluded_node_types = ["bedrock_agent.invoke_agent", "agent-collaborator"]
        if (
            parent_trace_node.node_type not in excluded_node_types
            and parent_trace_node.node_type != event_type
        ):
            self.trace_stack.pop()
            parent_trace_node = self.trace_stack.head or self.initial_node

        self._handle_new_trace_node(
            parent_trace_node, node_trace_id, event_type, chunk_type, trace_data
        )

    def _handle_parent_node_match_scenario(
        self,
        parent_trace_node: TraceNode,
        agent_node_trace_id: str,
        event_type: str,
        chunk_type: str,
        trace_data: Dict[str, Any],
    ) -> None:
        """
        Handle scenario where the parent node matches the trace ID.

        Args:
            parent_trace_node: The current parent trace node
            agent_node_trace_id: The agent node trace ID
            event_type: The type of event
            chunk_type: The type of chunk
            trace_data: The trace data to process
        """
        if (
            trace_data.get(event_type, {})
            .get(chunk_type, {})
            .get("agentCollaboratorInvocationInput")
        ):
            # Create new Node for Agent Collaboration
            self._handle_new_trace_node(
                parent_trace_node, agent_node_trace_id, event_type, chunk_type, trace_data
            )
        else:
            # Node already existed, add the chunk to existing node
            self._handle_chunk_for_current_node(parent_trace_node, chunk_type, trace_data)

    def _handle_existing_trace_id_scenario(
        self,
        parent_trace_node: TraceNode,
        agent_node_trace_id: str,
        event_type: str,
        chunk_type: str,
        trace_data: Dict[str, Any],
    ) -> None:
        """
        Handle scenario where the trace ID exists but parent node doesn't match.

        Args:
            parent_trace_node: The current parent trace node
            agent_node_trace_id: The agent node trace ID
            event_type: The type of event
            chunk_type: The type of chunk
            trace_data: The trace data to process
        """
        # We have received Trace with existing TraceID, but we are unable to locate the parent node
        if (
            trace_data.get(event_type, {})
            .get(chunk_type, {})
            .get("agentCollaboratorInvocationOutput")
        ):
            try:
                # This requires to pop the two parent nodes for the setting the Proper agent node
                # while reverse traversing from the stack
                # There are cases where we have created two parent node for the model Invocation
                # in Multi Agent mode. So we need to traverse back to proper agent node from
                # the orchestration trace of Agent.
                while (
                    self.trace_stack.head
                    and self.trace_stack.head.node_trace_id != agent_node_trace_id
                ):
                    self.trace_stack.pop()

                if self.trace_stack.head:
                    self.trace_stack.head.chunks.append(trace_data)
                else:
                    logger.warning(f"Could not find agent node with ID {agent_node_trace_id}")
                    self.initial_node.chunks.append(trace_data)
            except Exception as e:
                logger.error(f"Error processing agent collaborator output: {e}")
                self.initial_node.chunks.append(trace_data)
        else:
            self._handle_chunk_for_current_node(parent_trace_node, chunk_type, trace_data)

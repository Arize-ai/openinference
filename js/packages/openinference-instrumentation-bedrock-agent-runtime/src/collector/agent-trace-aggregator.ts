import { AgentTraceNode } from "./agent-trace-node";
import { TraceNodeStack } from "./trace-node-stack";
import { AgentChunkSpan } from "./agent-chunk-span";
import { AttributeExtractor } from "../attributes/attribute-extractor";
import { generateUniqueTraceId } from "../attributes/attribute-utils";
import { getObjectDataFromUnknown } from "../utils/json-utils";
import { UnknownRecord } from "../index";

/**
 * Aggregates agent trace data into a hierarchical structure of nodes and spans.
 */
export class AgentTraceAggregator {
  /** Root node that serves as the parent for all trace nodes. */
  readonly rootNode: AgentTraceNode;

  private readonly traceStack = new TraceNodeStack();
  private readonly nodesById: Record<string, AgentTraceNode> = {};
  private readonly seenIds: Set<string> = new Set();

  /**
   * Creates a new instance of AgentTraceAggregator with a default root node.
   */
  constructor() {
    this.rootNode = new AgentTraceNode(
      "default-parent-node",
      "bedrock_agent.invoke_agent",
    );
    this.traceStack.push(this.rootNode);
    this.nodesById[this.rootNode.nodeTraceId] = this.rootNode;
    this.seenIds.add(this.rootNode.nodeTraceId);
  }

  /**
   * Processes incoming trace data and returns the current trace node.
   * @param raw - The raw trace data to be processed
   * @returns The current active trace node after processing
   */
  collect(raw: UnknownRecord) {
    const traceData = this.unwrapTrace(raw);
    if (!traceData) {
      return;
    }
    const eventType = AttributeExtractor.getEventType(traceData);
    const traceId = AttributeExtractor.extractTraceId(traceData);
    if (!traceId) {
      return;
    }
    const nodeTraceId = generateUniqueTraceId(eventType, traceId);
    const chunkType = AttributeExtractor.getChunkType(
      getObjectDataFromUnknown({ data: traceData, key: eventType }) ?? {},
    );
    if (!chunkType) {
      return;
    }
    const agentChildId =
      this.detectAgentCollaboration(traceData, eventType, chunkType) &&
      `${nodeTraceId}-agent`;
    this.routeTraceData(
      traceData,
      this.traceStack.head!,
      nodeTraceId,
      agentChildId,
      eventType,
      chunkType,
    );
  }

  /**
   * Extracts trace data from nested Bedrock trace structure.
   * @param obj - The object potentially containing wrapped trace data
   * @returns The unwrapped trace data
   */
  private unwrapTrace(
    obj: Record<string, unknown>,
  ): Record<string, unknown> | null {
    return getObjectDataFromUnknown({ data: obj, key: "trace" });
  }

  /**
   * Checks if trace data contains agent collaboration markers.
   * @param traceData - The trace data to check
   * @param eventType - The type of event in the trace data
   * @param chunkType - The type of chunk in the event
   * @returns True if agent collaboration is detected, false otherwise
   */
  private detectAgentCollaboration(
    traceData: UnknownRecord,
    eventType: string,
    chunkType: string,
  ): boolean {
    const eventObj = getObjectDataFromUnknown({
      data: traceData,
      key: eventType,
    });
    if (!eventObj) {
      return false;
    }
    const chunkObj = getObjectDataFromUnknown({
      data: eventObj,
      key: chunkType,
    });
    if (!chunkObj) {
      return false;
    }
    return !!(
      chunkObj &&
      (chunkObj["agentCollaboratorInvocationInput"] ||
        chunkObj["agentCollaboratorInvocationOutput"])
    );
  }

  /**
   * Routes trace data to appropriate processing method based on trace state.
   * @param traceData - The trace data to process
   * @param parent - The parent node to which the trace data might be attached
   * @param nodeTraceId - The unique ID for the trace node
   * @param agentChildId - The ID for a potential agent collaboration child node, or false if not applicable
   * @param eventType - The type of event in the trace data
   * @param chunkType - The type of chunk in the event
   * @returns The current trace node after processing
   */
  private routeTraceData(
    traceData: UnknownRecord,
    parent: AgentTraceNode,
    nodeTraceId: string,
    agentChildId: string | false,
    eventType: string,
    chunkType: string,
  ): AgentTraceNode {
    // New node
    if (!this.seenIds.has(nodeTraceId)) {
      this.addNewTraceNodeIfAbsent(
        parent,
        nodeTraceId,
        eventType,
        chunkType,
        traceData,
      );
      return this.traceStack.head!;
    }

    // Current parent is correct target
    else if ([nodeTraceId, agentChildId].includes(parent.nodeTraceId)) {
      this.processChunkOrCreateAgentCollaborationNode(
        parent,
        agentChildId,
        eventType,
        chunkType,
        traceData,
      );
      return this.traceStack.head!;
    }

    // Need to pop up to parent node
    else if (parent.parentTraceNode?.nodeTraceId === nodeTraceId) {
      this.traceStack.pop();
      this.appendChunkToCurrentNodeOrStartNewSpan(
        this.traceStack.head!,
        chunkType,
        traceData,
      );
      return this.traceStack.head!;
    }

    // Node exists elsewhere in tree
    else if (this.seenIds.has(nodeTraceId)) {
      this.processTraceByExistingId(
        parent,
        agentChildId,
        eventType,
        chunkType,
        traceData,
      );
      return this.traceStack.head!;
    }
    return parent;
  }

  /**
   * Creates a new trace node when encountering an unseen trace ID.
   * @param parent - The potential parent node for the new trace node
   * @param nodeTraceId - The unique ID for the new trace node
   * @param eventType - The type of event for the new node
   * @param chunkType - The type of chunk in the event
   * @param traceData - The trace data to associate with the new node
   */
  private addNewTraceNodeIfAbsent(
    parent: AgentTraceNode,
    nodeTraceId: string,
    eventType: string,
    chunkType: string,
    traceData: UnknownRecord,
  ): void {
    const excluded = ["bedrock.invoke_agent", "agent-collaborator"];
    let anchor = parent;
    if (
      !excluded.includes(parent.nodeType ?? "") &&
      parent.nodeType !== eventType
    ) {
      this.traceStack.pop();
      anchor = this.traceStack.head ?? this.rootNode;
    }
    this.createAndAttachChildNode(
      anchor,
      nodeTraceId,
      eventType,
      chunkType,
      traceData,
    );
  }

  /**
   * Creates and attaches a new child node to the parent.
   * @param parent - The parent node to attach the new node to
   * @param nodeTraceId - The unique ID for the new node
   * @param eventType - The type of event for the new node
   * @param chunkType - The type of chunk in the event
   * @param traceData - The trace data to associate with the new node
   */
  private createAndAttachChildNode(
    parent: AgentTraceNode,
    nodeTraceId: string,
    eventType: string,
    chunkType: string,
    traceData: UnknownRecord,
  ): void {
    const eventObj =
      getObjectDataFromUnknown({ data: traceData, key: eventType }) ?? {};
    const chunkObj =
      getObjectDataFromUnknown({ data: eventObj, key: chunkType }) ?? {};
    let newNode: AgentTraceNode;
    if (chunkObj.agentCollaboratorInvocationInput) {
      newNode = new AgentTraceNode(nodeTraceId, "agent-collaborator");
      newNode.addChunk(traceData);
    } else if (eventType === "failureTrace") {
      newNode = new AgentTraceNode(nodeTraceId, eventType);
      newNode.addChunk(traceData);
    } else {
      newNode = new AgentTraceNode(nodeTraceId, eventType);
      const span = new AgentChunkSpan(chunkType);
      span.addChunk(traceData);
      span.parentNode = parent;
      newNode.addSpan(span);
    }

    newNode.parentTraceNode = parent;
    parent.addSpan(newNode);
    this.traceStack.push(newNode);

    this.nodesById[nodeTraceId] = newNode;
    this.seenIds.add(nodeTraceId);
  }

  /**
   * Processes chunks for existing nodes or creates agent collaboration nodes.
   * @param parent - The parent node to process
   * @param agentChildId - The ID for a potential agent collaboration child node, or false if not applicable
   * @param eventType - The type of event in the trace data
   * @param chunkType - The type of chunk in the event
   * @param traceData - The trace data to process
   */
  private processChunkOrCreateAgentCollaborationNode(
    parent: AgentTraceNode,
    agentChildId: string | false,
    eventType: string,
    chunkType: string,
    traceData: UnknownRecord,
  ): void {
    const eventObj = traceData[eventType] ?? {};
    const chunkObj =
      getObjectDataFromUnknown({ data: eventObj, key: chunkType }) ?? {};
    if (agentChildId && chunkObj.agentCollaboratorInvocationInput) {
      this.createAndAttachChildNode(
        parent,
        agentChildId,
        eventType,
        chunkType,
        traceData,
      );
    } else {
      this.appendChunkToCurrentNodeOrStartNewSpan(parent, chunkType, traceData);
    }
  }

  /**
   * Handles trace data for nodes that exist elsewhere in the trace tree.
   * @param _parent - The parent node (unused but maintained for API consistency)
   * @param agentChildId - The ID for a potential agent collaboration child node, or false if not applicable
   * @param eventType - The type of event in the trace data
   * @param chunkType - The type of chunk in the event
   * @param traceData - The trace data to process
   */
  private processTraceByExistingId(
    _parent: AgentTraceNode,
    agentChildId: string | false,
    eventType: string,
    chunkType: string,
    traceData: UnknownRecord,
  ): void {
    const eventObj =
      getObjectDataFromUnknown({ data: traceData, key: eventType }) ?? {};
    const chunkObj =
      getObjectDataFromUnknown({ data: eventObj, key: chunkType }) ?? {};
    if (agentChildId && chunkObj.agentCollaboratorInvocationOutput) {
      while (
        this.traceStack.head &&
        this.traceStack.head.nodeTraceId !== agentChildId
      ) {
        this.traceStack.pop();
      }
      (this.traceStack.head ?? this.rootNode).addChunk(traceData);
    } else {
      if (this.traceStack.head) {
        this.appendChunkToCurrentNodeOrStartNewSpan(
          this.traceStack.head,
          chunkType,
          traceData,
        );
      } else {
        this.appendChunkToCurrentNodeOrStartNewSpan(
          this.rootNode,
          chunkType,
          traceData,
        );
      }
    }
  }

  /**
   * Appends chunk to current span or creates a new span if required.
   * @param node - The node to which the chunk should be associated
   * @param chunkType - The type of chunk in the trace data
   * @param traceData - The trace data to process
   */
  private appendChunkToCurrentNodeOrStartNewSpan(
    node: AgentTraceNode,
    chunkType: string,
    traceData: UnknownRecord,
  ): void {
    const mustStartNew =
      ["invocationInput", "modelInvocationInput"].includes(chunkType) ||
      !node.currentSpan;

    if (mustStartNew) {
      this.startNewSpanWithChunk(node, chunkType, traceData);
    } else {
      node.currentSpan!.addChunk(traceData);
    }
  }

  /**
   * Creates a new span with the given chunk data.
   * @param node - The node to which the new span should be attached
   * @param chunkType - The type of chunk for the new span
   * @param traceData - The trace data to associate with the new span
   */
  private startNewSpanWithChunk(
    node: AgentTraceNode,
    chunkType: string,
    traceData: UnknownRecord,
  ): void {
    const span = new AgentChunkSpan(chunkType);
    span.addChunk(traceData);
    span.parentNode = node;
    node.addSpan(span);
  }
}

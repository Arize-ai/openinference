import { AgentChunkSpan } from "./agentChunkSpan";
import { StringKeyedObject } from "../types";

/**
 * Tree node representing an agent or model invocation. Each node can own:
 * - Multiple spans (AgentChunkSpan)
 * - Child nodes (forming the hierarchy)
 * - "Loose" chunks that do not belong to a span (e.g., failure traces)
 */
export class AgentTraceNode {
  /** Stable identifier for the node (prefixed trace-ID). */
  public readonly nodeTraceId: string;

  /** Semantic type, e.g., 'orchestrationTrace' or 'agent-collaborator'. */
  public readonly nodeType: string | null;

  /** Mixed list of spans and child trace nodes. */
  public readonly spans: (AgentChunkSpan | AgentTraceNode)[] = [];

  /** Back-pointer to parent in the tree. */
  public parentTraceNode: AgentTraceNode | null = null;

  /** The actively filling span (used while streaming). */
  public currentSpan: AgentChunkSpan | null = null;

  /** Chunks attached directly to the node (outside any span). */
  public readonly chunks: StringKeyedObject[] = [];

  /**
   * Creates a new AgentTraceNode instance.
   * @param params
   * @param params.traceId {string} - Stable identifier for the node (prefixed trace-ID)
   * @param params.eventType {string | null} - Semantic type, e.g., 'orchestrationTrace' or 'agent-collaborator'
   */
  constructor({
    traceId,
    eventType = null,
  }: {
    traceId: string;
    eventType: string | null;
  }) {
    this.nodeTraceId = traceId;
    this.nodeType = eventType;
  }

  /**
   * Adds a span or child node to this node.
   * If a span is added, it becomes the current active span.
   * @param span - The AgentChunkSpan or AgentTraceNode to add
   */
  addSpan(span: AgentChunkSpan | AgentTraceNode): void {
    this.spans.push(span);
    if (span instanceof AgentChunkSpan) this.currentSpan = span;
  }

  /**
   * Adds a chunk directly to this node (not part of any span).
   * @param chunk - The chunk object to add
   */
  addChunk(chunk: StringKeyedObject): void {
    this.chunks.push(chunk);
  }
}

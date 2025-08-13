import { StringKeyedObject } from "../types";
import { AgentTraceNode } from "./agent-trace-node";

/**
 * Logical work-unit inside a node. A span groups sequential chunks that share
 * the same spanType. Spans can be nested under nodes but never under each other.
 */
export class AgentChunkSpan {
  /** Raw payloads belonging to this span in order of arrival. */
  public readonly chunks: StringKeyedObject[] = [];

  /** Child nodes created from within this span (e.g., agent collaborators). */
  public readonly childrenNodes: AgentTraceNode[] = [];

  /** Back-pointer to the node that owns this span. */
  public parentNode: AgentTraceNode | null = null;

  /**
   * Create a new AgentChunkSpan.
   * @param spanType Type of the span (e.g., 'invocationInput').
   */
  constructor() {}

  /**
   * Add a chunk to this span.
   * @param chunk Chunk object to add.
   */
  addChunk(chunk: StringKeyedObject): void {
    this.chunks.push(chunk);
  }

  /**
   * Convert the span and its children to a plain object.
   * @returns Object representation of the span and its children.
   */
  toObject(): object {
    return {
      chunks: this.chunks,
      childrenNodes: this.childrenNodes.map((child) => child.toObject()),
      parentNodeTraceId: this.parentNode?.nodeTraceId ?? null,
    };
  }
}

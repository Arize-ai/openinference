import { AgentTraceNode } from "./agentTraceNode";

/**
 * Lightweight LIFO stack to keep track of the “current” AgentTraceNode while streaming.
 */
export class TraceNodeStack {
  /** Cached reference to the stack’s top element for O(1) access. */
  public currentNode: AgentTraceNode | null = null;
  private readonly stack: AgentTraceNode[] = [];

  /**
   * Returns the current top node in the stack.
   * @returns {AgentTraceNode | null} The current AgentTraceNode or null if the stack is empty.
   */
  get head() {
    return this.currentNode;
  }

  /**
   * Pushes a node onto the stack and updates the current node reference.
   * @param {AgentTraceNode} node - The AgentTraceNode to push onto the stack.
   */
  push(node: AgentTraceNode) {
    this.stack.push(node);
    this.currentNode = node;
  }

  /**
   * Pops the top node from the stack and updates the current node reference.
   * @returns {AgentTraceNode | null} The popped AgentTraceNode or null if the stack was empty.
   */
  pop() {
    const node = this.stack.pop() ?? null;
    this.currentNode = this.stack.at(-1) ?? null;
    return node;
  }
}

"""AG2 (formerly AutoGen) span attribute constants."""

# Agent
AG2_AGENT_NAME = "ag2.agent.name"
AG2_AGENT_DESCRIPTION = "ag2.agent.description"

# Two-agent chat
AG2_INITIATOR_NAME = "ag2.initiator.name"
AG2_RECIPIENT_NAME = "ag2.recipient.name"
AG2_MAX_TURNS = "ag2.max_turns"

# GroupChat
AG2_GROUPCHAT_AGENTS = "ag2.groupchat.agents"
AG2_GROUPCHAT_MAX_ROUND = "ag2.groupchat.max_round"
AG2_GROUPCHAT_SELECTOR = "ag2.groupchat.selector"
AG2_GROUPCHAT_TOTAL_ROUNDS = "ag2.groupchat.total_rounds"

# SwarmAgent
AG2_SWARM_INITIAL_AGENT = "ag2.swarm.initial_agent"
AG2_SWARM_AGENTS = "ag2.swarm.agents"
AG2_SWARM_MAX_ROUNDS = "ag2.swarm.max_rounds"

# ReasoningAgent
AG2_REASONING_METHOD = "ag2.reasoning.method"
AG2_REASONING_BEAM_SIZE = "ag2.reasoning.beam_size"
AG2_REASONING_MAX_DEPTH = "ag2.reasoning.max_depth"
AG2_REASONING_TOTAL_NODES = "ag2.reasoning.total_nodes"

# Nested chat (initiate_chats)
AG2_NESTED_CHAT_COUNT = "ag2.nested.chat_count"
AG2_NESTED_CARRYOVER_MODE = "ag2.nested.carryover_mode"

# Tool execution
AG2_TOOL_NAME = "ag2.tool.name"
AG2_TOOL_ARGUMENTS = "ag2.tool.arguments"

# Phoenix graph view (graph topology for GroupChat)
GRAPH_NODE_ID = "graph.node.id"
GRAPH_NODE_PARENT_ID = "graph.node.parent_id"

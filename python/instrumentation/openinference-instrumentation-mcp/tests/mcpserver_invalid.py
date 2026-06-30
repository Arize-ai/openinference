#!/usr/bin/env python3
"""MCP server that sends an invalid JSON-RPC message to trigger ValidationError."""

import json
import sys
import time

# Send a malformed JSON-RPC message (missing required 'method' field)
# This will cause pydantic ValidationError when MCP tries to parse it
invalid_message = {
    "jsonrpc": "2.0",
    "id": 1,
    # Missing 'method' field - this violates JSON-RPC spec and will trigger ValidationError
}

sys.stdout.write(json.dumps(invalid_message) + "\n")
sys.stdout.flush()

# Keep the process alive briefly so the client can read the message
time.sleep(2)

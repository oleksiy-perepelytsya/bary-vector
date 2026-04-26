#!/usr/bin/env bash
# Remote MCP server — pipes stdio through SSH to the VPS.
# Edit VPS_HOST / VPS_USER / VPS_PATH to match your setup, then point
# Claude Desktop or Claude Code at this script:
#
#   "barygraph-remote": {
#     "command": "/path/to/scripts/mcp_remote.sh"
#   }
#
# Requirements: SSH key auth must be configured (no password prompt).

VPS_HOST="${BARYGRAPH_VPS_HOST:-your-vps-hostname}"
VPS_USER="${BARYGRAPH_VPS_USER:-ubuntu}"
VPS_PATH="${BARYGRAPH_VPS_PATH:-~/barygraph-kaikki}"

exec ssh -T -o BatchMode=yes "${VPS_USER}@${VPS_HOST}" \
  "cd ${VPS_PATH} && python -m scripts.mcp_server"

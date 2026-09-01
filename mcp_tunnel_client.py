#!/usr/bin/env python3
"""Minimal, robust MCP streamable-http client for the barygraph server.

Verified working against the live tunnel (and localhost). Uses the inline
request/response pattern the FastMCP server reliably serves:

    1. POST initialize        → capture Mcp-Session-Id from the response header
    2. POST notifications/initialized (with session id)
    3. POST tools/call (etc.) with Mcp-Session-Id; response body is SSE-shaped
       ("event: message\\ndata: {jsonrpc...}") — we strip the event: prefix.

Alternatives that DON'T work reliably here:
  - Op/baking a second persistent SSE GET stream (FastMCP routes some replies to
    it; over the cloudflared tunnel that stream may never deliver a byte).
  - The official `mcp` SDK client streamable_http_client (v1.29): its GET attach
    races initialize session-id extraction, so its notifications/initialized POST
    loses the session id and the server answers 400 "Missing session ID".

Usage:
  python3 mcp_tunnel_client.py --url https://<tunnel>.trycloudflare.com/mcp \
      --query "How does curiosity become a durable research process?" \
      --seed-top-k 24 --bridge-top-k 12 --result-top-k 6 --max-hops 5 \
      --target-levels 10 11 12 --min-convergence 1
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "https://arg-abstracts-counseling-contacted.trycloudflare.com/mcp"


def _post(url: str, sid: str | None, payload: dict, read_timeout: float) -> tuple[int, str, dict]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if sid:
        headers["Mcp-Session-Id"] = sid
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    r = urllib.request.urlopen(req, timeout=read_timeout)
    body = r.read().decode()
    new_sid = r.headers.get("Mcp-Session-Id") or sid
    return r.status, body, {"sid": new_sid}


def post_with_retry(url: str, payload: dict, sid: str | None = None,
                    connect_timeout: float = 10, read_timeout: float = 120) -> tuple[int, str, str]:
    """POST with one retry. read_timeout is the socket timeout for the whole
    exchange (urllib does not split connect/read; requests-based clients can use
    requests.post(..., timeout=(connect_timeout, read_timeout)) instead)."""
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            status, body, meta = _post(url, sid, payload, read_timeout)
            if status >= 500:
                raise urllib.error.HTTPError(url, status, "server error", {}, None)
            return status, body, meta["sid"]
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt == 1:
                time.sleep(1.5)
    raise RuntimeError(f"POST failed after 2 attempts: {last_exc!r}")


def init_session(url: str, connect_timeout: float = 10,
                 read_timeout: float = 120) -> tuple[str, str]:
    payload = {
        "jsonrpc": "2.0",
        "id": "init",
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "mcp_tunnel_client", "version": "1.0"},
        },
    }
    status, body, sid = post_with_retry(url, payload, None, connect_timeout, read_timeout)
    if not sid:
        raise RuntimeError("no Mcp-Session-Id in initialize response")
    post_with_retry(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, sid,
                    connect_timeout, read_timeout)
    return sid, body


def parse_result(body: str) -> dict:
    """Response body is 'event: message\ndata: {...}'; extract the JSON line."""
    for line in body.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    return json.loads(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--query", default="memory")
    ap.add_argument("--seed-top-k", type=int, default=20)
    ap.add_argument("--bridge-top-k", type=int, default=12)
    ap.add_argument("--result-top-k", type=int, default=5)
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--target-levels", nargs="*", type=int, default=[12, 11, 10])
    ap.add_argument("--min-convergence", type=int, default=1)
    ap.add_argument("--connect-timeout", type=float, default=10)
    ap.add_argument("--read-timeout", type=float, default=120)
    ap.add_argument("--tool", default="associative_search")
    args = ap.parse_args()

    t0 = time.time()
    sid, _ = init_session(args.url, args.connect_timeout, args.read_timeout)
    print(f"[session {sid[:8]} ready in {time.time()-t0:.1f}s]")
    full = {"name": args.tool, "arguments": {
        "query": args.query,
        "seed_top_k": args.seed_top_k,
        "bridge_top_k": args.bridge_top_k,
        "result_top_k": args.result_top_k,
        "max_hops": args.max_hops,
        "target_levels": args.target_levels,
        "min_convergence": args.min_convergence,
        "return_paths": True,
    }}
    status, body, sid2 = post_with_retry(
        args.url,
        {"jsonrpc": "2.0", "id": "call", "method": "tools/call", "params": full},
        sid, args.connect_timeout, args.read_timeout)
    j = parse_result(body)
    txt = j["result"]["content"][0]["text"]
    print(json.dumps(json.loads(txt), indent=2)[:4000])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

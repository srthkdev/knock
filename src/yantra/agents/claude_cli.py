"""`claude -p` helper (local Claude Code login, no API key), mirroring codex_cli.

Each call is stateless and sandboxed: no tools at all (`--tools ""`), no MCP servers, no user or
project settings, no saved session, run in a fresh empty directory. The reply is forced into the
action schema with `--json-schema` and read from `structured_output`.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

from .codex_cli import CodexResult


def run_claude(prompt: str, schema: dict, *, model: str, timeout_s: float = 600,
               log_stem: Path | None = None) -> CodexResult:
    cmd = ["claude", "-p", "--model", model, "--output-format", "json", "--json-schema", json.dumps(schema),
           "--tools", "", "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence"]
    with tempfile.TemporaryDirectory(prefix="yantra-claude-") as tmp:
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout_s, cwd=tmp)
        except subprocess.TimeoutExpired:
            proc = subprocess.CompletedProcess(cmd, -9, "", f"timeout after {timeout_s:.0f}s")
        secs = time.time() - t0
    reply, usage, text = None, None, ""
    try:
        out = json.loads(proc.stdout)
        reply = out.get("structured_output")
        text = out.get("result") or ""
        usage = {"usage": out.get("usage"), "cost_usd": out.get("total_cost_usd"), "turns": out.get("num_turns")}
        if out.get("permission_denials"):
            usage["permission_denials"] = out["permission_denials"]
    except (json.JSONDecodeError, AttributeError):
        pass
    res = CodexResult(reply=reply, reply_text=json.dumps(reply) if reply else text, seconds=secs, exit=proc.returncode,
                      items=[], usage=usage, stdout=proc.stdout, stderr=proc.stderr)
    if log_stem is not None:
        log_stem = Path(log_stem)
        log_stem.parent.mkdir(parents=True, exist_ok=True)
        log_stem.with_suffix(".prompt.txt").write_text(prompt)
        log_stem.with_suffix(".stdout.json").write_text(proc.stdout)
        log_stem.with_suffix(".reply.json").write_text(res.reply_text)
    return res

"""Shared `codex exec` helper (local ChatGPT-plan login, no API key).

Each call is stateless: Codex runs ephemeral, read-only, in a fresh empty directory, with the
user config ignored (no MCP servers or plugins). The prompt goes in on stdin (then stdin is
closed), and the final reply is forced into a JSON schema via `--output-schema`.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ALLOWED_ITEMS = {"agent_message", "reasoning"}


@dataclass
class CodexResult:
    reply: dict | None  # parsed JSON reply, None if missing or unparseable
    reply_text: str
    seconds: float
    exit: int
    items: list[str] = field(default_factory=list)  # item types completed during the turn
    usage: dict | None = None
    stdout: str = ""
    stderr: str = ""

    @property
    def tainted_items(self) -> list[str]:
        """Item types beyond plain reasoning/messages (shell, file reads, web search...)."""
        return sorted(set(self.items) - ALLOWED_ITEMS)

    def meta(self) -> dict:
        return {"seconds": round(self.seconds, 1), "exit": self.exit, "items": self.items,
                "tainted_items": self.tainted_items, "usage": self.usage, "reply": self.reply}


def codex_command(work: Path, schema: Path, out: Path, model: str, effort: str, image_paths=()) -> list[str]:
    cmd = ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
           "-s", "read-only", "-C", str(work), "-m", model,
           "-c", f'model_reasoning_effort="{effort}"',
           "--json", "--output-schema", str(schema), "-o", str(out)]
    for p in image_paths:
        cmd += ["-i", str(p)]
    cmd.append("-")  # prompt from stdin
    return cmd


def run_codex(prompt: str, schema: dict, *, model: str, effort: str, image_paths=(), timeout_s: float = 600,
              log_stem: Path | None = None) -> CodexResult:
    """Run one `codex exec` call and return its parsed reply plus event metadata.

    If `log_stem` is given, the prompt, raw event stream, stderr and reply are written next to it.
    """
    with tempfile.TemporaryDirectory(prefix="knock-codex-") as tmp:
        schema_path = Path(tmp) / "schema.json"
        schema_path.write_text(json.dumps(schema))
        out = Path(tmp) / "out.json"
        work = Path(tmp) / "work"
        work.mkdir()
        cmd = codex_command(work, schema_path, out, model, effort, image_paths)
        t0 = time.time()
        # input= writes the prompt and closes stdin
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as e:  # a turn that thinks too long is a lost turn, not a crash
            out_s = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            proc = subprocess.CompletedProcess(cmd, -9, out_s, f"timeout after {timeout_s:.0f}s")
        secs = time.time() - t0
        reply_txt = out.read_text() if out.exists() else ""
    items, usage = scan_events(proc.stdout)
    try:
        reply = json.loads(reply_txt)
    except json.JSONDecodeError:
        reply = None
    res = CodexResult(reply=reply, reply_text=reply_txt, seconds=secs, exit=proc.returncode, items=items,
                      usage=usage, stdout=proc.stdout, stderr=proc.stderr)
    if log_stem is not None:
        log_stem = Path(log_stem)
        log_stem.parent.mkdir(parents=True, exist_ok=True)
        log_stem.with_suffix(".prompt.txt").write_text(prompt)
        log_stem.with_suffix(".events.jsonl").write_text(proc.stdout)
        if proc.stderr.strip():
            log_stem.with_suffix(".stderr.txt").write_text(proc.stderr)
        log_stem.with_suffix(".reply.json").write_text(json.dumps(reply, indent=1))
    return res


def scan_events(stdout: str) -> tuple[list[str], dict | None]:
    """Completed item types and token usage from a `codex exec --json` event stream."""
    items, usage = [], None
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "item.completed":
            items.append(ev.get("item", {}).get("type", "?"))
        if ev.get("type") == "turn.completed":
            usage = ev.get("usage")
    return items, usage

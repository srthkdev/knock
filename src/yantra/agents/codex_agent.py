"""Codex CLI as a Knock agent (uses the local ChatGPT-plan login, no API key).

One stateless `codex exec` call per decision: the full history is in the prompt, the scene
render and every spectrogram so far are attached as images, and the reply is forced into a
JSON schema. Codex runs read-only in an empty directory, ignores the user config (so no
MCP servers or plugins), and every event is logged. If the model runs any tool (shell, file
read, web search...), the episode is marked `tainted`: it could have read the hidden answer.
"""
from __future__ import annotations

import json
from pathlib import Path

from .codex_cli import ALLOWED_ITEMS, run_codex  # noqa: F401  (ALLOWED_ITEMS re-exported)

INSTRUCTIONS = """\
You are the agent in Knock, a physics lab. You cannot see hidden properties directly; you must
run probe experiments, reason about the readings with physics, and then make one flick.
Do NOT run commands, read files, or browse. Answer only with the JSON action.

When you flick, choose the impulse carefully: it is your only attempt and it is scored by where
the object stops. Use the probe readings quantitatively.
"""


def _schema(ids):
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["probe", "flick"]},
            "probe": {"type": "string", "enum": ["push", "lift", "tap", "shake", "none"]},
            "object": {"type": "string", "enum": ids},
            "force_N": {"type": "number", "description": "push force; 0 if not a push"},
            "impulse_Ns": {"type": "number", "description": "flick impulse; 0 if not a flick"},
            "reasoning": {"type": "string", "description": "brief justification"},
        },
        "required": ["action", "probe", "object", "force_N", "impulse_Ns", "reasoning"],
        "additionalProperties": False,
    }


def _describe(step_no, action, obs):
    a = action
    if a.get("type") == "probe":
        head = f"{step_no}. {a['probe']}({a['object']}" + (f", force_N={a.get('force_N')}" if a["probe"] == "push" else "") + ")"
    else:
        head = f"{step_no}. {a}"
    if "error" in obs:
        return f"{head} -> ERROR: {obs['error']}"
    keep = {k: v for k, v in obs.items() if k in ("force_N", "moved_mm", "weight_g", "sound")}
    return f"{head} -> {json.dumps(keep)}"


class CodexAgent:
    def __init__(self, model="gpt-6-luna", effort="high", timeout_s=600):
        self.model = model
        self.effort = effort
        self.timeout_s = timeout_s
        self._meta = {}

    @property
    def name(self):
        return f"codex-{self.model}-{self.effort}"

    def episode_meta(self):
        return self._meta

    # ------------------------------------------------------------------ loop
    def run(self, ep):
        obs0 = ep.reset()
        ids = [o["id"] for o in obs0["level"]["objects"]]
        lines, images = [], [(obs0.get("image"), "scene render (all objects at their start, x = 0)")]
        self._meta = {"calls": [], "tainted": False, "model": self.model, "effort": self.effort}
        call_dir = Path(ep.out) / "codex"
        call_dir.mkdir(exist_ok=True)
        max_calls = ep.level.task.budget + 4
        result = None
        for k in range(max_calls):
            left = ep.level.task.budget - ep.used
            prompt = self._prompt(obs0, lines, images, left, force_flick=(k == max_calls - 1))
            reply = self._call(prompt, ids, [p for p, _ in images if p], call_dir / f"call{k:02d}")
            if reply is None:
                lines.append(f"(your previous reply could not be parsed; reply again with valid JSON)")
                continue
            if reply["action"] == "flick" or left <= 0 and reply["action"] == "probe" and k == max_calls - 1:
                result = ep.step({"type": "flick", "object": reply["object"], "impulse_Ns": reply["impulse_Ns"]})
                break
            action = {"type": "probe", "probe": reply["probe"], "object": reply["object"], "force_N": reply["force_N"]}
            obs = ep.step(action)
            lines.append(_describe(len(lines) + 1, action, obs))
            if "spectrogram" in obs:
                images.append((obs["spectrogram"], f"spectrogram of {obs['probe']}({obs['object']}), step {len(lines)}"))
        return result

    # ------------------------------------------------------------------ prompt
    def _prompt(self, obs0, lines, images, left, force_flick=False):
        lv = obs0["level"]
        img_list = "\n".join(f"  image {i + 1}: {d}" for i, (p, d) in enumerate(images) if p)
        hist = "\n".join(lines) if lines else "(no probes yet)"
        ask = ("You have no probes left (or this is your last turn): you MUST flick now."
               if left <= 0 or force_flick else
               f"You have {left} probe(s) left. Choose your next probe, or flick now if you are confident.")
        return f"""{INSTRUCTIONS}
{obs0['rules']}
Level
{json.dumps(lv, indent=1)}

Attached images
{img_list}

History of your probes and their readings
{hist}

{ask}
Reply with JSON: action ("probe" or "flick"), probe (push/lift/tap/shake, or "none" for a flick),
object, force_N (push only, else 0), impulse_Ns (flick only, else 0), reasoning (short).
"""

    # ------------------------------------------------------------------ codex exec
    def _call(self, prompt, ids, image_paths, stem):
        res = run_codex(prompt, _schema(ids), model=self.model, effort=self.effort, image_paths=image_paths,
                        timeout_s=self.timeout_s, log_stem=stem)
        if res.tainted_items:
            self._meta["tainted"] = True
        self._meta["calls"].append(res.meta())
        return res.reply

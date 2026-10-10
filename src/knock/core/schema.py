"""LLM-facing action format, generated from tool specs.

Structured-output APIs (and `codex exec --output-schema`) want one strict JSON schema: every field
required, no extra fields. We flatten all available tools into one object: `tool`, `target`, an
`args` object with the union of argument fields and a `prediction` object with the union of
prediction fields (each nullable), plus `reasoning`. `parse_action` maps a reply back to a ToolCall
that keeps only the fields of the chosen tool.
"""
from __future__ import annotations

from typing import Any

from .tool import Tool, ToolCall

_KEEP = ("type", "enum", "description")


def _field_schema(name: str, prop: dict, required_in: str) -> dict:
    out = {k: v for k, v in prop.items() if k in _KEEP}
    lo, hi = prop.get("minimum"), prop.get("maximum")
    desc = out.get("description", prop.get("title", name))
    if lo is not None or hi is not None:
        desc += f" (range {lo}..{hi})"
    out["description"] = f"{desc} [{required_in}]"
    t = out.get("type", "string")
    out["type"] = [t, "null"] if isinstance(t, str) else [*t, "null"]
    if "enum" in out:
        out["enum"] = [*out["enum"], None]
    return out


def _union(tools: list[Tool], which: str) -> dict[str, dict]:
    fields: dict[str, dict] = {}
    users: dict[str, list[str]] = {}
    for t in tools:
        model = t.Args if which == "args" else t.Prediction
        if model is None:
            continue
        for name, prop in model.model_json_schema().get("properties", {}).items():
            users.setdefault(name, []).append(t.name)
            fields.setdefault(name, prop)
    return {n: _field_schema(n, p, "used by: " + ", ".join(users[n]) + "; null otherwise") for n, p in fields.items()}


def action_schema(tools: list[type[Tool] | Tool], target_ids: list[str]) -> dict:
    tools = list(tools)
    args = _union(tools, "args")
    preds = _union(tools, "prediction")
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": [t.name for t in tools], "description": "which tool to use"},
            "target": {"type": ["string", "null"], "enum": [*target_ids, None],
                       "description": "component id the tool acts on, or null if the tool takes no target"},
            "args": {"type": "object", "properties": args, "required": list(args), "additionalProperties": False},
            "prediction": {"type": "object", "properties": preds, "required": list(preds),
                           "additionalProperties": False},
            "reasoning": {"type": "string", "description": "brief justification"},
        },
        "required": ["tool", "target", "args", "prediction", "reasoning"],
        "additionalProperties": False,
    }


def parse_action(reply: dict[str, Any], tools: list[type[Tool] | Tool]) -> ToolCall:
    by_name = {t.name: t for t in tools}
    name = reply.get("tool")
    tool = by_name.get(name)
    if tool is None:
        return ToolCall(tool=str(name), target=reply.get("target"), reasoning=reply.get("reasoning", ""))
    args = {k: v for k, v in (reply.get("args") or {}).items() if k in tool.Args.model_fields and v is not None}
    pred = None
    if tool.Prediction is not None:
        pred = {k: v for k, v in (reply.get("prediction") or {}).items()
                if k in tool.Prediction.model_fields and v is not None} or None
    return ToolCall(tool=name, target=reply.get("target"), args=args, prediction=pred,
                    reasoning=reply.get("reasoning", ""))

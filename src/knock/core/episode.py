"""Episode: one agent, one world, one task. Enforces the budget, validates calls, scores predictions.

Nothing here is specific to any pack: tools and tasks bring all the physics.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .component import World
from .events import EventLog
from .registry import TOOLS
from .task import Task
from .tool import Attachment, Observation, Tool, ToolCall, ToolContext


@dataclass
class AgentView:
    """Everything an agent is allowed to know at a decision point."""
    world: dict
    task: dict
    budget_left: float  # probe budget left (classic tasks)
    tools: list[dict]
    history: list[dict]
    attachments: list[Attachment] = field(default_factory=list)
    done: bool = False
    budgets_left: dict[str, float] = field(default_factory=dict)
    total_reward: float = 0.0


class Episode:
    def __init__(self, world: World, task: Task, out_dir: str | Path | None = None, events: EventLog | None = None,
                 tools: dict[str, Tool] | None = None, require_predictions: bool = True):
        self.require_predictions = require_predictions  # False: predictions optional (ablations)
        self.world = world
        self.task = task
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.events = events or EventLog()
        names = [*task.probes, *([task.act] if task.act else [])]
        self._tools = tools or {n: TOOLS.get(n)() for n in names}
        if task.act and not self._tools[task.act].terminal:
            raise ValueError(f"act {task.act!r} is not a terminal tool")
        self.spent: dict[str, float] = {k: 0.0 for k in task.budgets}
        self.total_reward = 0.0
        self.history: list[tuple[ToolCall, Observation]] = []
        self.result: dict | None = None
        self.events.emit("episode_start", 0, world=world.public(), task=task.public())

    # ------------------------------------------------------------------ what the agent sees
    @property
    def done(self) -> bool:
        return self.result is not None

    @property
    def used(self) -> float:
        """Probe units spent (classic tasks)."""
        return self.spent.get("probes", 0.0)

    @property
    def budgets_left(self) -> dict[str, float]:
        return {k: self.task.budgets[k] - self.spent.get(k, 0.0) for k in self.task.budgets}

    @property
    def budget_left(self) -> float:
        return self.budgets_left.get("probes", 0.0)

    def _fits(self, costs: dict[str, float]) -> bool:
        left = self.budgets_left
        need = dict(costs)
        if "turns" in left:
            need["turns"] = need.get("turns", 0.0) + 1.0
        return all(left.get(k, float("inf")) >= v - 1e-9 for k, v in need.items() if v > 0)

    def available(self) -> list[Tool]:
        tools = [self._tools[n] for n in self.task.probes
                 if self._tools[n].finishes or self._fits(self._tools[n].costs())]
        if self.task.act:
            tools.append(self._tools[self.task.act])
        return tools

    def view(self) -> AgentView:
        hist = []
        atts: list[Attachment] = []
        for c, o in self.history:
            hist.append({"call": c.to_dict(), "observation": o.data})
            atts.extend(o.attachments)
        return AgentView(world=self.world.public(), task=self.task.public(), budget_left=self.budget_left,
                         tools=[t.spec() for t in self.available()], history=hist, attachments=atts, done=self.done,
                         budgets_left=self.budgets_left, total_reward=self.total_reward)

    # ------------------------------------------------------------------ acting
    def call(self, call: ToolCall) -> Observation:
        step = len(self.history) + 1
        if self.done:
            raise RuntimeError("episode is over")
        try:
            tool, target, args, prediction = self._validate(call)
        except ValueError as e:
            obs = Observation({"error": str(e)})
            self.events.emit("error", step, call=call.to_dict(), error=str(e))
            return obs

        costs = tool.costs(args)
        if not tool.terminal and not tool.finishes and not self._fits(costs):
            obs = Observation({"error": f"not enough budget for {tool.name}: needs {costs}, left {self.budgets_left}"})
            self.events.emit("error", step, call=call.to_dict(), error=obs.data["error"])
            return obs
        self.events.emit("tool_call", step, call=call.to_dict())
        ctx = ToolContext(self.world, step, self.out_dir, list(self.history))
        if tool.terminal:
            outcome = tool.outcome(self.world, target, args)
            obs = Observation(data=dict(outcome), truth=dict(outcome))
        else:
            obs = tool.run(ctx, target, args)
            for k, v in costs.items():
                if k in self.spent:
                    self.spent[k] += v
        if "turns" in self.spent:
            self.spent["turns"] += 1.0

        scores = tool.score_prediction(prediction, obs) if prediction is not None else {}
        if scores:
            obs.data["prediction_error"] = {k: {"error": v["error"], "unit": v["unit"]} for k, v in scores.items()}
            self.events.emit("prediction_scored", step, tool=tool.name, target=call.target, scores=scores)
        self.events.emit("observation", step, tool=tool.name, target=call.target, data=obs.data,
                         attachments=[a.__dict__ for a in obs.attachments])
        self.history.append((call, obs))
        if obs.reward is not None:
            self.total_reward += float(obs.reward)
            self.events.emit("reward", step, tool=tool.name, reward=float(obs.reward), total=self.total_reward)

        if not tool.terminal and (tool.finishes or self._exhausted()):
            self._finalize(step, reason="finished" if tool.finishes else "budget exhausted")
            return obs
        if tool.terminal:
            success = bool(self.task.success(self.world, call, obs.truth))
            self.result = {
                "success": success,
                "outcome": obs.truth,
                "act": call.to_dict(),
                "probes_used": self.used,
                "prediction": scores,
                "probe_predictions": [e.payload for e in self.events.of("prediction_scored")
                                      if e.payload["tool"] != tool.name],
                **self.task.diagnostics(self.world, call, obs.truth),
            }
            self.events.emit("outcome", step, **self.result)
            self.events.emit("episode_end", step, success=success)
        return obs

    def _exhausted(self) -> bool:
        """Games without an act end when no budgeted tool fits any more (or turns run out)."""
        if self.task.act:
            return False
        left = self.budgets_left
        if left.get("turns", 1.0) <= 1e-9:
            return True
        return not any(self._fits(self._tools[n].costs()) for n in self.task.probes if not self._tools[n].finishes)

    def _finalize(self, step: int, reason: str) -> None:
        final = self.task.final(self.world, self)
        self.result = {"success": bool(final.get("success")), "score": final.get("score"), "end": reason,
                       "spent": dict(self.spent), "total_reward": self.total_reward, "probes_used": self.used,
                       "probe_predictions": [e.payload for e in self.events.of("prediction_scored")],
                       **{k: v for k, v in final.items() if k not in ("success", "score")}}
        self.events.emit("outcome", step, **self.result)
        self.events.emit("episode_end", step, success=self.result["success"])

    def _validate(self, call: ToolCall):
        allowed = {t.name: t for t in self.available()}
        if call.tool not in allowed:
            if call.tool in self._tools:
                need = self._tools[call.tool].costs()
                reason = f"not available: budget too low (needs {need}, left {self.budgets_left})"
            else:
                reason = "not available in this task"
            raise ValueError(f"tool {call.tool!r} {reason}; available: {sorted(allowed)}")
        tool = allowed[call.tool]
        target = None
        if tool.targets:
            if call.target is None:
                raise ValueError(f"{tool.name} needs a target")
            try:
                target = self.world.get(str(call.target))
            except KeyError:
                raise ValueError(f"unknown target {call.target!r}; targets: {self.world.ids()}") from None
            if target.kind not in tool.targets:
                raise ValueError(f"{tool.name} cannot target a {target.kind}")
        try:
            args = tool.Args.model_validate(call.args or {})
        except ValidationError as e:
            raise ValueError(f"bad arguments for {tool.name}: {e.errors()[0]['msg']} ({e.errors()[0]['loc']})") from None
        prediction = None
        if tool.Prediction is not None:
            if call.prediction:
                try:
                    prediction = tool.Prediction.model_validate(call.prediction)
                except ValidationError as e:
                    raise ValueError(f"bad prediction for {tool.name}: {e.errors()[0]['msg']}") from None
            elif tool.prediction_required and self.require_predictions:
                raise ValueError(f"{tool.name} requires a prediction: {list(tool.Prediction.model_fields)}")
        return tool, target, args, prediction

    # ------------------------------------------------------------------ records
    def trace(self, extra: dict | None = None) -> dict:
        from .. import BENCHMARK_VERSION
        return {
            **(extra or {}),  # agent metadata first: core fields below can never be overwritten
            "benchmark_version": BENCHMARK_VERSION,
            "world": self.world.to_dict(),
            "task": self.task.to_dict(),
            "calls": [{"call": c.to_dict(), "observation": o.data, "attachments": [a.__dict__ for a in o.attachments]}
                      for c, o in self.history],
            "result": self.result,
            "events": [e.to_dict() for e in self.events.events],
        }

    def save(self, extra: dict | None = None) -> Path | None:
        if not self.out_dir:
            return None
        p = self.out_dir / "trace.json"
        p.write_text(json.dumps(self.trace(extra), indent=1, default=str))
        return p

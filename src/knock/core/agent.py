"""Agent interface and the generic episode loop.

An agent maps an AgentView (public world, task, budget, available tool specs, history) to one
ToolCall. Scripted baselines, learned baselines and LLM harnesses all implement the same method,
so any agent can play any pack.
"""
from __future__ import annotations

from typing import Protocol

from .episode import AgentView, Episode
from .tool import ToolCall


class Agent(Protocol):
    name: str

    def act(self, view: AgentView) -> ToolCall: ...


def run_episode(agent: Agent, episode: Episode, max_calls: int | None = None) -> dict:
    """Ask the agent for calls until the act ends the episode.

    Invalid calls are returned to the agent as error observations (they cost no budget). If the
    agent keeps failing, the episode ends without an act and is scored as a failure.
    """
    limit = max_calls or (episode.task.budget + 6)
    if hasattr(agent, "reset"):
        agent.reset(episode.view())
    for _ in range(limit):
        if episode.done:
            break
        episode.call(agent.act(episode.view()))
    if not episode.done:
        episode.result = {"success": False, "error": "no valid act within the call limit",
                          "probes_used": episode.used}
        episode.events.emit("episode_end", len(episode.history), success=False)
    return episode.result

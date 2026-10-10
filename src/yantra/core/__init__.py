"""Yantra core: pack-independent building blocks.

    Component  a physical thing with visible and hidden parameters
    Tool       anything an agent can do (probes and the terminal act), with optional predictions
    Task       a goal over the act's outcome, a budget and the allowed tools
    World      components plus settings;  Scenario = world + task;  Pack = a family of scenarios
    Episode    budget, validation, prediction scoring, event stream, trace
    Agent      AgentView -> ToolCall;  run_episode drives any agent on any pack
    oracle     solves any task by searching the act's arguments with the simulator
"""
from .agent import Agent, run_episode
from .component import Component, World
from .episode import AgentView, Episode
from .events import Event, EventLog, jsonl_writer
from .oracle import OracleResult, solve
from .pack import Pack, Scenario
from .params import Choice, Flag, LogUniform, Param, ParamSet, Uniform
from .registry import COMPONENTS, PACKS, TASKS, TOOLS, Registry
from .schema import action_schema, parse_action
from .task import Task
from .tool import Attachment, NoArgs, Observation, Tool, ToolCall, ToolContext

__all__ = [
    "Agent", "AgentView", "Attachment", "COMPONENTS", "Choice", "Component", "Episode", "Event", "EventLog", "Flag",
    "LogUniform", "NoArgs", "Observation", "OracleResult", "PACKS", "Pack", "Param", "ParamSet", "Registry", "Scenario",
    "TASKS", "TOOLS", "Task", "Tool", "ToolCall", "ToolContext", "Uniform", "World", "action_schema", "jsonl_writer",
    "parse_action", "run_episode", "solve",
]

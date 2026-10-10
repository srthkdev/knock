"""Event stream: everything that happens in an episode, in order.

Replays, traces, videos and live views are all consumers of the same events, so a new pack gets
all of them for free.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

KINDS = ("episode_start", "tool_call", "observation", "prediction_scored", "outcome", "error", "episode_end")


@dataclass
class Event:
    kind: str
    step: int
    payload: dict[str, Any]
    t: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class EventLog:
    """Collects events and forwards them to subscribers (files, websockets, UIs)."""

    def __init__(self):
        self.events: list[Event] = []
        self._subscribers: list[Callable[[Event], None]] = []

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        self._subscribers.append(fn)

    def emit(self, kind: str, step: int, **payload) -> Event:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        ev = Event(kind, step, payload)
        self.events.append(ev)
        for fn in self._subscribers:
            fn(ev)
        return ev

    def of(self, kind: str) -> list[Event]:
        return [e for e in self.events if e.kind == kind]


def jsonl_writer(path: str | Path) -> Callable[[Event], None]:
    """Subscriber that appends each event as one JSON line (default=str for paths, numpy scalars)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def write(ev: Event) -> None:
        with p.open("a") as f:
            f.write(json.dumps(ev.to_dict(), default=str) + "\n")

    return write

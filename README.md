# Knock

A physics lab where AI agents must **run experiments to discover hidden properties** before they act.

Objects in a level look identical, but differ in hidden mass, friction, material, and what's inside
(hollow, solid, a loose marble, a glued weight...). An agent gets a small budget of probes —
`push`, `lift`, `tap`, `shake` — and then **one flick** that must stop the right object inside a target band.

- Physics: MuJoCo (deterministic; masses and friction match textbook Coulomb physics to <1%).
- Sound: contact events from MuJoCo drive modal synthesis, so taps and rattles come from the simulation.
- Every level is checked by an oracle that knows the hidden values: all levels are solvable.

## Quick start

```bash
uv sync
uv run knock gen --n 30 --out levels/dev              # verified levels
uv run knock run --agent scripted --levels levels/dev --out runs/scripted
uv run knock run --agent codex --model gpt-6-luna --levels levels/dev --out runs/luna
uv run knock summary runs/*
uv run --group dev pytest -q
```

## Build worlds from a prompt

```bash
uv run knock build "three identical boxes, one has a marble, far target"   # -> levels/custom/<slug>.json
uv run knock build --batch scripts/prompts_showcase.txt --out levels/showcase
```

A world-builder agent (Codex CLI, same sandboxed `codex exec` call as the Codex agent) fills a
`LevelSpec` (`src/knock/spec.py`, Pydantic, exported as a strict JSON schema) from the request.
Each proposal is checked and the failures are fed back until it passes (default 4 rounds):

| Check | Rule |
|---|---|
| solvable | the oracle lands the answer object in the band with a flick in range |
| unique | find tasks: exactly one object has the property |
| identifiable | tap tells hollow/solid; the rattler's first shake is >= 15 dB over floor and the loudest; heaviest/lightest differ by >= 25%; the budget covers the probes needed |
| fair | find tasks: all objects look identical |

Passing levels get a difficulty label from the none / random / scripted baselines
(`easy`: a blind flick works, `moderate`/`tricky`: probing needed, `hard`: scripted fails).
A `<slug>.report.json` with every round is written next to each level.

## Agents

| Agent | What it is |
|---|---|
| `none` / `random` / `scripted` | Bayesian physics baselines that differ only in which probes they choose |
| `codex` | Codex CLI (`codex exec`), one stateless call per decision, using your ChatGPT login |

Codex runs read-only in an empty directory with user config ignored. Every event is logged; if the
model runs any tool (shell, file read, web search), the episode is marked `tainted` because it could
have read the hidden answer from the level files.

## Layout

```
src/knock/
  level.py      visible layout, hidden properties, tasks, level generator
  world.py      MuJoCo scenes, probes, flick
  sound.py      modal synthesis, sound features, spectrograms
  env.py        episode: budgeted probes, one scored flick, traces
  oracle.py     ground-truth solver; proves levels are solvable
  spec.py       LevelSpec: LLM-facing level schema with catalogue validation
  builder.py    prompt -> validated level (Codex proposes, checks give feedback)
  render.py     offscreen scene renders
  agents/       baselines and the Codex agent
tests/          physics consistency, determinism, no-leak, solvability
```

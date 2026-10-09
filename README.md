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

## Agents

| Agent | What it is |
|---|---|
| `none` / `random` / `scripted` | Bayesian physics baselines that differ only in which probes they choose |
| `sbi` | Neural posterior estimation (`sbi` MNPE) trained on 20k MuJoCo probe simulations; probes chosen by expected information gain. Retrain with `uv run python scripts/train_sbi.py --n 20000` |
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
  render.py     offscreen scene renders
  agents/       baselines and the Codex agent
tests/          physics consistency, determinism, no-leak, solvability
```

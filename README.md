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

## Audio engine (Rust)

Sound synthesis lives in one Rust crate, `crates/knock-audio`, used from Python and the browser:
`synthesize` (modal resonator bank driven by contact events), `room_and_mic` (short room reverb
with a `tail_s` parameter, default 0.01 s; mic gain jitter; absolute noise floor; optional grip
rustle) and `features` (the acoustic summary agents see). All randomness comes from a PCG64 generator
seeded with a u64, so a clip depends only on its inputs.

- **Python**: `uv sync` builds the `knock_audio` extension with maturin. It is a uv workspace member
  and a dependency of `knock`, so `cargo` must be on `PATH` (`export PATH="$HOME/.cargo/bin:$PATH"`).
  `knock.sound` uses it automatically. Set `KNOCK_AUDIO=python` to use the numpy reference
  implementation instead. The two backends agree statistically, and `features` agrees exactly
  (`tests/test_audio_backend.py`).
- **Browser**: rebuild the ES module in `web/public/wasm/` with
  ```bash
  wasm-pack build crates/knock-audio --target web --release --out-dir ../../web/public/wasm \
      --out-name knock_audio --no-pack -- --features wasm && rm web/public/wasm/.gitignore
  ```
  ```js
  import init, * as ka from "/wasm/knock_audio.js";
  await init();
  const x = ka.render_f32(events, freqs, decay, gains, 42n, 1.4, false, ka.reverb_tail_s()); // Float32Array
  ```
  `synthesize`, `room_and_mic` and `features` take Float64Array (`features_f32` takes Float32Array).
  Events are flat `[t, J, tau, ...]` triples and seeds are `bigint`.
- **Checks**: `cargo test -p knock-audio`, `cargo run --release --example bench -p knock-audio`, and
  `uv run python scripts/bench_audio.py` (Rust vs numpy on 400 clips).

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
  sound.py      modal synthesis, sound features, spectrograms (Rust backend + numpy fallback)
  env.py        episode: budgeted probes, one scored flick, traces
  oracle.py     ground-truth solver; proves levels are solvable
  render.py     offscreen scene renders
  agents/       baselines and the Codex agent
tests/          physics consistency, determinism, no-leak, solvability, audio backend parity
crates/knock-audio/   Rust audio engine (PyO3 extension + WASM)
web/public/wasm/     prebuilt WASM package of the audio engine
```

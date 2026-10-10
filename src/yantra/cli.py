"""Command line: generate levels, run agents, summarise results, build levels from prompts.

    yantra gen --n 30 --out levels/dev
    yantra run --agent scripted --levels levels/dev --out runs/scripted
    yantra run --agent codex --model gpt-6-luna --levels levels/dev --out runs/luna
    yantra summary runs/*
    yantra build "three identical boxes, one has a marble, far target"
    yantra build --batch prompts.txt --out levels/showcase
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

from .env import Episode
from .level import Level
from .oracle import valid_level
from .spec import HEX_TO_COLOR

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Yantra: AI agents build, test and fix physical systems; packs: factory, cube lab.")
console = Console()


@app.command()
def gen(n: int = typer.Option(30, help="number of levels"),
        start: int = typer.Option(0, help="first seed"),
        out: str = typer.Option("levels/dev", help="output directory")):
    """Generate random oracle-verified levels."""
    out_p = Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        lv = valid_level(start + i)
        (out_p / f"{lv.id}.json").write_text(lv.to_json())
    print(f"wrote {n} levels to {out_p}")


def make_agent(agent: str, model: str, effort: str):
    if agent in ("none", "random", "scripted"):
        from .agents.baselines import BaselineAgent
        return BaselineAgent(agent)
    if agent == "sbi":
        from .agents.sbi_agent import SbiAgent
        return SbiAgent()
    if agent == "codex":
        from .agents.codex_agent import CodexAgent
        return CodexAgent(model=model, effort=effort)
    raise typer.BadParameter(f"unknown agent {agent}")


@app.command()
def run(out: str = typer.Option(..., help="output directory for traces"),
        agent: str = typer.Option("scripted", help="none | random | scripted | sbi | codex"),
        model: str = typer.Option("gpt-6-luna", help="codex model"),
        effort: str = typer.Option("high", help="codex reasoning effort"),
        levels: str = typer.Option("levels/dev", help="directory of level JSON files"),
        limit: int = typer.Option(0, help="only the first N levels (0 = all)"),
        render: bool = typer.Option(True, "--render/--no-render", help="render scene images"),
        overwrite: bool = typer.Option(False, "--overwrite", help="rerun levels that already have a trace")):
    """Run an agent on a directory of levels."""
    ag = make_agent(agent, model, effort)
    out_p = Path(out)
    files = sorted(f for f in Path(levels).glob("*.json") if not f.name.endswith(".report.json"))[: limit or None]
    results = []
    for f in files:
        lv = Level.from_json(f.read_text())
        ep_dir = out_p / lv.id
        if (ep_dir / "trace.json").exists() and not overwrite:
            results.append(json.loads((ep_dir / "trace.json").read_text())["result"])
            continue
        t0 = time.time()
        ep = Episode(lv, ep_dir, render=render)
        ag.run(ep)
        meta = getattr(ag, "episode_meta", lambda: {})()
        ep.save_trace({"agent": ag.name, "seconds": round(time.time() - t0, 1), **meta})
        r = ep.result or {"success": False, "error": "no flick"}
        results.append(r)
        print(f"{lv.id} {lv.task.kind:15} {str(lv.task.property or ''):9} "
              f"{'OK ' if r.get('success') else 'x  '} obj={'y' if r.get('correct_object') else 'n'} "
              f"err={r.get('error_m', float('nan')):+.3f} probes={r.get('probes_used')} {time.time() - t0:.0f}s", flush=True)
    summarise(out_p, results)


def summarise(out, results):
    n = len(results)
    if not n:
        return
    s = sum(bool(r.get("success")) for r in results)
    c = sum(bool(r.get("correct_object")) for r in results)
    print(f"{out}: success {s}/{n} = {s / n:.0%} | correct object {c}/{n}")


@app.command()
def summary(dirs: list[str] = typer.Argument(..., help="run directories")):
    """Summarise success rates of run directories."""
    for d in dirs:
        traces = sorted(Path(d).glob("*/trace.json"))
        summarise(d, [json.loads(t.read_text())["result"] or {} for t in traces])


# ---------------------------------------------------------------- build
def print_report(rep, level) -> None:
    status = "[green]PASS[/]" if rep.passed else "[red]FAIL[/]"
    console.rule(f"{rep.slug}  {status}  ({len(rep.rounds)} round(s))")
    if level is not None:
        t = level.task
        what = f"slide {t.object_id}" if t.kind == "slide" else f"find the {t.property} object"
        o0 = level.objects[0]
        console.print(f"[bold]{level.notes.get('title', '')}[/]  {level.notes.get('description', '')}")
        console.print(f"{len(level.objects)} cubes (edge {2 * o0.half:.3f} m), task: {what}, "
                      f"band {t.target_x - t.tol:.3f}-{t.target_x + t.tol:.3f} m, budget {t.budget}")

    rounds = Table(title="Rounds", show_lines=False)
    for col in ("round", "result", "seconds", "failures"):
        rounds.add_column(col)
    for r in rep.rounds:
        rounds.add_row(str(r.round), "[green]pass[/]" if r.passed else "[red]fail[/]", f"{r.seconds:.0f}",
                       "\n".join(r.failures) or "-")
    console.print(rounds)

    if level is not None:
        objs = Table(title="Objects (hidden)")
        for col in ("id", "edge m", "colour", "material", "hollow", "contents", "friction", "mass kg"):
            objs.add_column(col)
        answer = level.answer_id() if rep.passed else None
        for o in level.objects:
            h = o.hidden
            mark = " *" if o.id == answer else ""
            objs.add_row(o.id + mark, f"{2 * o.half:.3f}", HEX_TO_COLOR.get(o.color, o.color), h.material, str(h.hollow), h.contents,
                         f"{h.friction:.2f}", f"{o.mass:.3f}")
        console.print(objs)

    checks = Table(title="Checks")
    for col in ("check", "result", "detail"):
        checks.add_column(col)
    for c in rep.checks:
        checks.add_row(c.name, "[green]pass[/]" if c.passed else "[red]fail[/]", c.detail)
    console.print(checks)

    if rep.difficulty:
        d = rep.difficulty
        diff = Table(title=f"Difficulty: [bold]{d['label']}[/]")
        for col in ("baseline", "success", "correct object", "error m", "probes"):
            diff.add_column(col)
        rows = [("none", d["none"]), *[(f"random#{i}", r) for i, r in enumerate(d["random"])], ("scripted", d["scripted"])]
        for name, r in rows:
            diff.add_row(name, "[green]yes[/]" if r["success"] else "[red]no[/]", str(r["correct_object"]),
                         f"{r['error_m']:+.3f}", str(r["probes_used"]))
        console.print(diff)
    if rep.level_path:
        console.print(f"saved {rep.level_path}  (report {rep.report_path})")
    else:
        console.print(f"[red]no level saved[/]  (report {rep.report_path})")


@app.command()
def serve(host: str = typer.Option("127.0.0.1"), port: int = typer.Option(8765)):
    """Serve the web app and the build API (prompt -> playable level)."""
    from .server import serve as _serve
    console.print(f"Yantra at http://{host}:{port}  (Build needs a local Codex login)")
    _serve(host, port)


@app.command()
def build(prompt: Annotated[Optional[str], typer.Argument(help="natural-language level request")] = None,
          batch: Optional[Path] = typer.Option(None, help="file with one prompt per line (# comments allowed)"),
          out: str = typer.Option("levels/custom", help="output directory"),
          model: str = typer.Option("gpt-6.1-sol", help="codex model for the world builder"),
          effort: str = typer.Option("medium", help="codex reasoning effort"),
          max_rounds: int = typer.Option(4, help="maximum propose/validate rounds"),
          baselines: bool = typer.Option(True, "--baselines/--no-baselines", help="run baselines for difficulty")):
    """Build a validated level from a natural-language prompt (via Codex)."""
    from .builder import build_level

    if (prompt is None) == (batch is None):
        raise typer.BadParameter("give either a PROMPT or --batch FILE")
    prompts = [prompt] if prompt else [
        ln.strip() for ln in batch.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    summary_rows = []
    for p in prompts:
        console.print(f"[bold cyan]build[/] {p!r}")
        level, rep = build_level(p, model=model, effort=effort, max_rounds=max_rounds, out_dir=out,
                                 run_baselines=baselines, log=console.print)
        print_report(rep, level)
        summary_rows.append((rep.slug, rep.passed, len(rep.rounds), (rep.difficulty or {}).get("label", "-")))
    if len(prompts) > 1:
        t = Table(title="Batch summary")
        for col in ("level", "result", "rounds", "difficulty"):
            t.add_column(col)
        for slug, ok, n, label in summary_rows:
            t.add_row(slug, "[green]pass[/]" if ok else "[red]fail[/]", str(n), label)
        console.print(t)
    if not all(r[1] for r in summary_rows):
        raise typer.Exit(1)


def main():
    app()


if __name__ == "__main__":
    main()

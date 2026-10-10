"""Write the factory results table into README.md (between the results markers) from the web index."""
import json
import re
import sys
from pathlib import Path

ix = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else "web/public/data/factory/index.json").read_text())
tasks = [t for t in ix["tasks"] if t.get("split") in ("public", "test")]
runs = sorted({r["run"] for t in tasks for r in t["runs"]})
label = {r["run"]: r["label"] for t in tasks for r in t["runs"]}
tiers = sorted({t["tier"] for t in tasks})


def cell(run, pred):
    rs = [r for t in tasks if pred(t) for r in t["runs"] if r["run"] == run]
    return f"{sum(r['success'] for r in rs)}/{len(rs)}" if rs else "–"


rows = []
for run in runs:
    rs = [r for t in tasks for r in t["runs"] if r["run"] == run]
    passed = sum(r["success"] for r in rs)
    rows.append((passed / max(1, len(rs)), run, rs))
head = "| Model | Passed | " + " | ".join(f"Tier {t}" for t in tiers) + " | Held-out | Mean score |"
sep = "|---|---:|" + "---:|" * len(tiers) + "---:|---:|"
lines = [head, sep]
for _, run, rs in sorted(rows, reverse=True):
    mean = sum(r["score"] or 0 for r in rs) / max(1, len(rs))
    lines.append(f"| **{label[run]}** | **{cell(run, lambda t: True)}** | "
                 + " | ".join(cell(run, lambda t, k=k: t["tier"] == k) for k in tiers)
                 + f" | {cell(run, lambda t: t['split'] == 'test')} | {mean:.0f} |")
n_maps = len(tasks)
note = (f"\n{n_maps} maps ({sum(t['split'] == 'test' for t in tasks)} held out), one episode per model per map. "
        "Score = good items shipped in the 90 s production run, minus 5 per bad item shipped, 2 per spill, 3 per jam.\n")
readme = Path("README.md")
s = readme.read_text()
s = re.sub(r"<!-- results:start -->.*<!-- results:end -->",
           "<!-- results:start -->\n" + "\n".join(lines) + "\n" + note + "<!-- results:end -->", s, flags=re.S)
readme.write_text(s)
print("\n".join(lines))

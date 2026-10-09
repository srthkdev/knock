"""Local API for the web app: build worlds from prompts and play them immediately.

    uv run knock serve            # http://127.0.0.1:8765 (web app + /api)
    cd web && npm run dev         # dev server proxies /api to 8765

Building calls the Codex world builder, so it only works locally with a Codex login.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .builder import build_level
from .webexport import export_level

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "web" / "public" / "data"
DIST = ROOT / "web" / "dist"

app = FastAPI(title="Knock")
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


class BuildRequest(BaseModel):
    prompt: str = Field(min_length=4, max_length=500)
    model: str = "gpt-6.1-sol"
    effort: str = "medium"


@app.get("/api/health")
def health():
    return {"ok": True}


def _run(job_id: str, req: BuildRequest):
    job = JOBS[job_id]

    def log(msg):
        with LOCK:
            job["log"].extend(line for line in str(msg).splitlines() if line.strip())

    try:
        log(f"builder: {req.model} ({req.effort})")
        level, report = build_level(req.prompt, model=req.model, effort=req.effort, log=log,
                                    out_dir=ROOT / "levels" / "custom", log_dir=ROOT / "runs" / "build")
        rep = report.to_dict()
        job["report"] = {"passed": rep["passed"], "rounds": len(rep["rounds"]), "checks": rep["checks"],
                         "difficulty": rep["difficulty"], "oracle": rep["oracle"]}
        if report.passed and level is not None:
            log("exporting level for the browser…")
            summary = export_level(level, DATA)
            idx_path = DATA / "index.json"
            idx = json.loads(idx_path.read_text()) if idx_path.exists() else {"levels": [], "runs": []}
            custom = [c for c in idx.get("custom", []) if c["id"] != level.id]
            idx["custom"] = [{**summary, "prompt": req.prompt, "difficulty": (rep["difficulty"] or {}).get("label")}] + custom
            idx_path.write_text(json.dumps(idx, indent=1))
            job["level_id"] = level.id
            job["hidden"] = [{"id": o.id, "material": o.hidden.material, "hollow": o.hidden.hollow,
                              "contents": o.hidden.contents, "friction": o.hidden.friction,
                              "mass_g": round(o.mass * 1000, 1)} for o in level.objects]
            job["status"] = "done"
        else:
            job["status"] = "failed"
    except Exception as e:  # surfaced to the UI
        log(f"error: {e}")
        job["status"] = "failed"
    job["finished"] = time.time()


@app.post("/api/build")
def start_build(req: BuildRequest):
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"id": job_id, "prompt": req.prompt, "status": "running", "log": [], "started": time.time()}
    threading.Thread(target=_run, args=(job_id, req), daemon=True).start()
    return {"id": job_id}


@app.get("/api/build/{job_id}")
def build_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    with LOCK:
        return {**job, "log": list(job["log"]), "elapsed": round(time.time() - job["started"], 1)}


# static: live data dir first (custom levels appear without rebuilding), then the built app
app.mount("/data", StaticFiles(directory=DATA), name="data")
if DIST.exists():
    @app.get("/")
    def index():
        return FileResponse(DIST / "index.html")

    app.mount("/", StaticFiles(directory=DIST, html=True), name="app")


def serve(host: str = "127.0.0.1", port: int = 8765):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")

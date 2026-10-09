"""Train the neural posterior estimator used by the `sbi` agent.

    uv run python scripts/train_sbi.py --n 30000            # simulate + train + calibrate
    uv run python scripts/train_sbi.py --n 30000 --epochs 0 # only simulate (cached in runs/)

Simulations run the real MuJoCo probes in parallel; they are cached in runs/sbi_sims_<n>.npy.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per simulation worker

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from knock.agents import sbi_agent as S

HELDOUT_SEED0 = 10_000_000


def simulate(seeds, workers):
    with mp.Pool(workers) as pool:
        rows = []
        for i, r in enumerate(pool.imap(S.simulate_record, seeds, chunksize=8)):
            rows.append(r)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(seeds)} sims", flush=True)
    return np.stack(rows)


def load_or_simulate(path: Path, seeds, workers):
    if path.exists():
        return np.load(path)
    t0 = time.time()
    data = simulate(list(seeds), workers)
    print(f"simulated {len(data)} in {time.time() - t0:.0f}s ({workers} workers)")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data)
    return data


def theta_of(rec):
    return torch.tensor(S.theta_to_net(rec[:, S.R_LOGM], rec[:, S.R_MU], rec[:, S.R_HOLLOW], rec[:, S.R_RATTLE]),
                        dtype=torch.float32)


def train(data, epochs, batch, lr, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    perm = rng.permutation(len(data))
    n_val = max(500, len(data) // 20)
    val, tr = data[perm[:n_val]], data[perm[n_val:]]
    feats = S.transform_values(tr[:, S.R_OBS:S.R_OBS + S.N_VALUES])
    stats = {"mean": feats.mean(0), "std": feats.std(0) + 1e-3}
    post = S.NpePosterior(S.build_estimator(), stats)
    est = post.est.train()
    opt = torch.optim.AdamW(est.parameters(), lr=lr, weight_decay=1e-5)
    steps = epochs * int(np.ceil(len(tr) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.05)
    th_tr, th_val = theta_of(tr), theta_of(val)
    x_val = post.x(val[:, S.R_OBS:S.R_OBS + S.N_VALUES], S.random_slot_masks(len(val), rng), val[:, S.R_HALF])
    best, best_state, t0 = np.inf, None, time.time()
    for ep in range(epochs):
        est.train()
        masks = S.random_slot_masks(len(tr), rng)  # fresh probe subsets every epoch
        x_tr = post.x(tr[:, S.R_OBS:S.R_OBS + S.N_VALUES], masks, tr[:, S.R_HALF])
        order = torch.randperm(len(tr))
        tot = 0.0
        for i in range(0, len(tr), batch):
            idx = order[i:i + batch]
            loss = est.loss(th_tr[idx], x_tr[idx]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(est.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        est.eval()
        with torch.no_grad():
            vl = float(est.loss(th_val, x_val).mean())
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in est.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"epoch {ep:3d} train {tot / len(tr):.3f} val {vl:.3f} ({time.time() - t0:.0f}s)", flush=True)
    est.load_state_dict(best_state)
    post.est = est.eval()
    return post, {"best_val_loss": best, "train_seconds": round(time.time() - t0, 1), "n_train": len(tr)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30000)
    p.add_argument("--heldout", type=int, default=2000)
    p.add_argument("--workers", type=int, default=mp.cpu_count())
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--out", default=str(S.MODEL_PATH))
    p.add_argument("--cache", default="runs")
    a = p.parse_args()
    torch.set_num_threads(max(1, a.workers))

    t0 = time.time()
    data = load_or_simulate(Path(a.cache) / f"sbi_sims_{a.n}.npy", range(a.n), a.workers)
    held = load_or_simulate(Path(a.cache) / f"sbi_heldout_{a.heldout}.npy",
                            range(HELDOUT_SEED0, HELDOUT_SEED0 + a.heldout), a.workers)
    sim_s = time.time() - t0
    if a.epochs == 0:
        return
    post, info = train(data, a.epochs, a.batch, a.lr)
    post.bank = data
    post.meta = {"n_sims": a.n, "push_forces": S.PUSH_FORCES, **info}
    post.save(a.out)
    post = S.NpePosterior.load(a.out)  # evaluate exactly what was saved (fp16 weights)
    cal = S.calibration(post, held)
    report = {"simulate_seconds": round(sim_s, 1), **info, "calibration": cal,
              "model_bytes": Path(a.out).stat().st_size}
    print(json.dumps(report, indent=1))
    Path(a.cache, "sbi_train_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

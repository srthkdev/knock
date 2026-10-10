"""Simulation-based inference baseline: amortised neural posterior estimation (NPE).

Per object the hidden parameters are

    theta = (log mass, friction, hollow, rattle)

with a prior that mirrors `level.generate` (material, hollow, contents -> mass; friction
uniform). A mixed neural posterior estimator from `sbi` (MNPE: an autoregressive
categorical net for (hollow, rattle) and a neural spline flow for (log m, friction)
conditioned on the discrete part) is trained on observations produced by the *real*
simulator (`world.simulate_push`, `world.weigh`, `world.simulate_tap`,
`world.simulate_shake`) for a fixed probe set per object:

    push at PUSH_FORCES (mm moved, clipped at 0), lift (grams), tap (5 sound features),
    shake (5 sound features)

Missing probes are zeroed and flagged with mask indicators, so one network gives the
posterior for any subset of probes (training resamples the masks every epoch).

The agent chooses probes greedily by expected information gain about what matters for
the task (which object has the property, and log(m^2 mu) of the object to flick, whose
uncertainty is floored at the target band's relative width), using the posterior
predictive. It then flicks the most probable object with the impulse that maximises the
posterior probability of the Coulomb stopping distance d = J^2 / (2 m^2 mu g) landing in
the band.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .. import world
from ..env import PUSH_SECONDS
from ..level import FLICK_RANGE, Hidden, Obj
from ..materials import MATERIALS

G = 9.81
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "sbi_npe.pt"

PUSH_FORCES = (0.5, 1.5, 4.0, 10.0, 20.0)
SLOTS = [f"push{i}" for i in range(len(PUSH_FORCES))] + ["lift", "tap", "shake"]
SOUND_KEYS = ("ring_time_s", "distinct_hits", "brightness_hz", "dominant_pitch_hz", "loudness_db_over_floor")
SLOT_DIMS = [1] * len(PUSH_FORCES) + [1, 5, 5]
SLOT_START = np.concatenate([[0], np.cumsum(SLOT_DIMS)])[:-1]
N_VALUES = int(sum(SLOT_DIMS))  # 16 raw observation values
N_SLOTS = len(SLOTS)  # 8 mask bits
X_DIM = N_VALUES + N_SLOTS + 1  # values, masks, half-size

MATERIAL_NAMES = list(MATERIALS)
CONTENT_NAMES = ["empty", "foam", "weight", "marble", "beads"]
RATTLE = ("marble", "beads")
HALF_RANGE = (0.04, 0.07)
MU_RANGE = (0.15, 0.6)
P_HOLLOW, P_RATTLE_GIVEN_HOLLOW = 0.5, 0.2

# record layout produced by `simulate_record`
R_HALF, R_LOGM, R_MU, R_HOLLOW, R_RATTLE, R_MAT, R_CONT = range(7)
R_OBS = 7  # raw observation values start here (N_VALUES of them)


# ---------------------------------------------------------------- prior and simulator
def sample_hidden(rng) -> tuple[float, Hidden]:
    """One object from the prior (marginal of `level.generate`, with rattles included)."""
    half = round(float(rng.uniform(*HALF_RANGE)), 3)
    material = str(rng.choice(MATERIAL_NAMES))
    hollow = bool(rng.random() < P_HOLLOW)
    if not hollow:
        contents = "empty"
    elif rng.random() < P_RATTLE_GIVEN_HOLLOW:
        contents = str(rng.choice(list(RATTLE)))
    else:
        contents = str(rng.choice(["empty", "foam", "weight"]))
    mu = round(float(rng.uniform(*MU_RANGE)), 3)
    return half, Hidden(material, hollow, contents, mu)


def simulate_record(seed: int) -> np.ndarray:
    """Sample theta from the prior and run every probe of the fixed probe set once."""
    rng = np.random.default_rng(seed)
    half, h = sample_hidden(rng)
    o = Obj("A", half, "#888888", 0.0, h)
    seeds = rng.integers(1 << 30, size=len(PUSH_FORCES) + 2)
    obs = []
    for F, s in zip(PUSH_FORCES, seeds):
        disp = world.simulate_push(o, F, PUSH_SECONDS, seed=int(s))
        obs.append(max(0.0, disp * 1000 + rng.normal(0, 1.0)))  # same noise model as env
    obs.append(world.weigh(o, rng))
    for sim, s in ((world.simulate_tap, seeds[-2]), (world.simulate_shake, seeds[-1])):
        _, f = sim(o, seed=int(s))
        obs += [float(f[k]) for k in SOUND_KEYS]
    head = [half, np.log(o.mass), h.friction, float(h.hollow), float(h.contents in RATTLE),
            MATERIAL_NAMES.index(h.material), CONTENT_NAMES.index(h.contents)]
    return np.array(head + obs, dtype=np.float64)


# ---------------------------------------------------------------- encoding
def transform_values(raw: np.ndarray) -> np.ndarray:
    """Raw observation values (..., 16) -> roughly Gaussian-scaled features."""
    raw = np.asarray(raw, dtype=np.float64)
    out = np.empty_like(raw)
    npush = len(PUSH_FORCES)
    out[..., :npush] = np.log1p(np.maximum(raw[..., :npush], 0.0))
    out[..., npush] = np.log(np.maximum(raw[..., npush], 1.0))
    for base in (npush + 1, npush + 6):
        out[..., base] = np.log(raw[..., base] + 0.01)  # ring time
        out[..., base + 1] = np.log1p(raw[..., base + 1])  # hits
        out[..., base + 2] = np.log(np.maximum(raw[..., base + 2], 1.0))  # brightness
        out[..., base + 3] = np.log(np.maximum(raw[..., base + 3], 1.0))  # pitch
        out[..., base + 4] = raw[..., base + 4]  # loudness (dB)
    return out


def value_mask(slot_mask: np.ndarray) -> np.ndarray:
    """(..., 8) slot mask -> (..., 16) value mask."""
    return np.repeat(slot_mask, SLOT_DIMS, axis=-1)


def encode(raw: np.ndarray, slot_mask: np.ndarray, half: np.ndarray, stats: dict) -> np.ndarray:
    """Network input: z-scored features (0 where missing), slot mask, scaled half-size."""
    z = (transform_values(raw) - stats["mean"]) / stats["std"]
    vm = value_mask(slot_mask)
    z = np.where(vm > 0, z, 0.0)
    h = (np.asarray(half, dtype=np.float64)[..., None] - 0.055) / 0.009
    return np.concatenate([z, slot_mask, h], axis=-1).astype(np.float32)


# theta <-> unconstrained, standardised space used by the flow
LOGM_SHIFT, LOGM_SCALE = -0.3, 1.3


def theta_to_net(logm, mu, hollow, rattle) -> np.ndarray:
    u = np.clip((np.asarray(mu) - MU_RANGE[0]) / (MU_RANGE[1] - MU_RANGE[0]), 1e-4, 1 - 1e-4)
    return np.stack([(np.asarray(logm) - LOGM_SHIFT) / LOGM_SCALE, np.log(u / (1 - u)) / 1.8,
                     np.asarray(hollow, dtype=np.float64), np.asarray(rattle, dtype=np.float64)], -1)


def net_to_theta(t: np.ndarray) -> dict:
    logm = t[..., 0] * LOGM_SCALE + LOGM_SHIFT
    mu = MU_RANGE[0] + (MU_RANGE[1] - MU_RANGE[0]) / (1 + np.exp(-1.8 * t[..., 1]))
    return {"logm": logm, "mu": mu, "hollow": t[..., 2] > 0.5, "rattle": t[..., 3] > 0.5}


def random_slot_masks(n: int, rng) -> np.ndarray:
    """Probe subsets for training: mostly small (agents have 1-4 probes), sometimes all."""
    k_p = np.array([1.0, 3, 3, 2, 1.5, 0.5, 0.4, 0.3, 0.3])
    k = rng.choice(len(k_p), size=n, p=k_p / k_p.sum())
    order = np.argsort(rng.random((n, N_SLOTS)), axis=1)
    return (np.argsort(order, axis=1) < k[:, None]).astype(np.float64)


# ---------------------------------------------------------------- network
def build_estimator(hidden: int = 96, embed: int = 64):
    """sbi mixed NPE: categorical net for (hollow, rattle) + NSF for (log m, friction)."""
    import warnings

    from sbi.neural_nets.net_builders import build_mnpe

    embedding = nn.Sequential(nn.Linear(X_DIM, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                              nn.Linear(hidden, embed))
    g = torch.Generator().manual_seed(0)
    theta = torch.cat([torch.randn(64, 2, generator=g), torch.randint(0, 2, (64, 2), generator=g).float()], 1)
    x = torch.randn(64, X_DIM, generator=g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_mnpe(theta, x, z_score_x="none", z_score_y="none", flow_model="nsf",
                          num_categories_per_variable=torch.tensor([2, 2]), embedding_net=embedding,
                          num_transforms=4, num_bins=8, hidden_features=hidden, tail_bound=5.0)


class NpePosterior:
    """Trained estimator plus feature statistics and a small simulation bank (for predictives)."""

    def __init__(self, est, stats: dict, bank: np.ndarray | None = None, meta: dict | None = None):
        self.est = est.eval()
        self.stats = stats
        self.bank = bank
        self.meta = meta or {}
        self._trees = None

    # ------------------------------------------------------------ io
    def save(self, path=MODEL_PATH, bank_size: int = 4000):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bank = None if self.bank is None else torch.tensor(self.bank[:bank_size], dtype=torch.float32)
        torch.save({"state": {k: v.half() if v.is_floating_point() else v for k, v in self.est.state_dict().items()},
                    "stats": {k: torch.tensor(v) for k, v in self.stats.items()}, "bank": bank,
                    "meta": self.meta}, path)

    @staticmethod
    def load(path=MODEL_PATH) -> "NpePosterior":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        est = build_estimator(**ck["meta"].get("arch", {}))
        est.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state"].items()})
        stats = {k: v.numpy().astype(np.float64) for k, v in ck["stats"].items()}
        bank = None if ck["bank"] is None else ck["bank"].numpy().astype(np.float64)
        return NpePosterior(est, stats, bank, ck["meta"])

    # ------------------------------------------------------------ inference
    def x(self, raw, slot_mask, half) -> torch.Tensor:
        return torch.from_numpy(encode(raw, slot_mask, half, self.stats))

    @torch.no_grad()
    def sample(self, x: torch.Tensor, n: int, seed: int = 0) -> dict:
        """Joint posterior samples, arrays of shape (n, batch)."""
        torch.manual_seed(seed)
        t = self.est.sample((n,), x.reshape(-1, X_DIM)).numpy().astype(np.float64)
        return net_to_theta(t)

    @torch.no_grad()
    def discrete_probs(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Exact P(hollow), P(rattle) by enumerating the categorical net."""
        x = x.reshape(-1, X_DIM)
        combos = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        lp = torch.stack([self.est.discrete_net.log_prob(c.expand(len(x), 2), x).reshape(-1) for c in combos], 1)
        p = torch.softmax(lp, 1).numpy()
        return p[:, 1] + p[:, 2], p[:, 2] + p[:, 3]

    # ------------------------------------------------------------ predictive bank
    def bank_sound(self, half: float, hollow, rattle, logm, rng) -> np.ndarray:
        """Tap+shake raw features of the nearest simulated object (same inside, similar size/mass)."""
        from scipy.spatial import cKDTree

        if self._trees is None:
            b = self.bank
            self._trees = {}
            for hv in (0, 1):
                for rv in (0, 1):
                    idx = np.where((b[:, R_HOLLOW] == hv) & (b[:, R_RATTLE] == rv))[0]
                    if len(idx):
                        pts = np.stack([b[idx, R_HALF] / 0.004, b[idx, R_LOGM] / 0.15], 1)
                        self._trees[(hv, rv)] = (cKDTree(pts), idx)
        out = np.empty((len(logm), 10))
        for i, (hv, rv, lm) in enumerate(zip(hollow, rattle, logm)):
            key = (int(hv), int(rv) if hv else 0)
            tree, idx = self._trees.get(key, self._trees[(int(hv), 0)])
            _, nn_i = tree.query([half / 0.004, lm / 0.15], k=4)
            j = idx[nn_i[rng.integers(4)]]
            out[i] = self.bank[j, R_OBS + N_VALUES - 10: R_OBS + N_VALUES]
        return out


def push_mm(F, m, mu, T=PUSH_SECONDS, max_s=8.0):
    """Coulomb push distance in mm, including the simulator's 8 s episode cap (~1% of simulate_push)."""
    a = np.maximum(F / m - mu * G, 0.0)
    v1 = a * T
    tau = np.minimum(v1 / (mu * G), max_s - T)  # sliding time after release, truncated at the cap
    return 1000 * (a * T ** 2 / 2 + v1 * tau - mu * G * tau ** 2 / 2)


def flick_distance(J, m, mu):
    return (J / m) ** 2 / (2 * mu * G)


# ---------------------------------------------------------------- agent
class SbiAgent:
    """Greedy expected-information-gain probing + Bayes-optimal flick under the NPE posterior."""

    name = "sbi-npe"

    def __init__(self, model_path=MODEL_PATH, n_post: int = 256, n_pred: int = 16, seed: int = 0):
        self.post = NpePosterior.load(model_path)
        self.n_post, self.n_pred, self.seed = n_post, n_pred, seed

    # ------------------------------------------------------------ state per object
    def _x(self, half, obs: dict) -> torch.Tensor:
        raw, mask = np.zeros(N_VALUES), np.zeros(N_SLOTS)
        for slot, vals in obs.items():
            i = SLOTS.index(slot)
            raw[SLOT_START[i]:SLOT_START[i] + SLOT_DIMS[i]] = vals
            mask[i] = 1
        return self.post.x(raw[None], mask[None], np.array([half]))

    @staticmethod
    def _record(obs: dict, beliefs: dict):
        oid, p = obs["object"], obs["probe"]
        if p == "push":
            k = int(np.argmin([abs(np.log(obs["force_N"] / F)) for F in PUSH_FORCES]))
            beliefs[oid][f"push{k}"] = [obs["moved_mm"]]
        elif p == "lift":
            beliefs[oid]["lift"] = [obs["weight_g"]]
        else:
            beliefs[oid][p] = [float(obs["sound"][k]) for k in SOUND_KEYS]

    # ------------------------------------------------------------ task utility
    def _summaries(self, xs: torch.Tensor, seed: int) -> dict:
        """Posterior summaries for a batch of conditions."""
        s = self.post.sample(xs, self.n_post, seed=seed)
        ph, pr = self.post.discrete_probs(xs)
        return {"logm": s["logm"], "mu": s["mu"], "q": 2 * s["logm"] + np.log(s["mu"]),
                "hollow": s["hollow"], "rattle": s["rattle"], "ph": ph, "pr": pr}

    @staticmethod
    def _given_answer(level, s: dict) -> np.ndarray:
        """Which posterior samples are consistent with this object being the answer."""
        prop = level.task.property if level.task.kind == "find_and_slide" else None
        keep = {"hollow": s["hollow"], "solid": ~s["hollow"], "rattles": s["rattle"]}.get(prop)
        if keep is None or keep.sum() < 20:
            return np.ones(len(s["logm"]), dtype=bool)
        return keep

    def _answer_probs(self, level, summ: list[dict], rng) -> np.ndarray:
        t = level.task
        n = len(summ)
        if t.kind == "slide":
            return np.array([float(o.id == t.object_id) for o in level.objects])
        if t.property in ("hollow", "solid", "rattles"):
            if t.property == "rattles":  # every object in a rattles level is hollow
                p = np.array([s["pr"] / max(s["ph"], 1e-6) for s in summ], dtype=np.float64)
            else:
                p = np.array([s["ph"] for s in summ], dtype=np.float64)
            if t.property == "solid":
                p = 1 - p
            p = np.clip(p, 1e-6, 1 - 1e-6)
            score = np.array([p[i] * np.prod(np.delete(1 - p, i)) for i in range(n)])
            return score / score.sum()
        m = np.stack([rng.permutation(s["logm"]) for s in summ])
        best = m.argmax(0) if t.property == "heaviest" else m.argmin(0)
        return (np.bincount(best, minlength=n) + 0.5) / (m.shape[1] + 0.5 * n)

    def _uncertainty(self, level, summ, rng) -> float:
        """Entropy of the answer + expected (floored) entropy of log(m^2 mu) of the answer."""
        P = self._answer_probs(level, summ, rng)
        s0 = level.task.tol / level.task.target_x  # band width in log-distance
        h_id = -float(np.sum(P * np.log(np.maximum(P, 1e-12))))
        h_q = sum(P[i] * 0.5 * np.log(np.var(s["q"][self._given_answer(level, s)]) + s0 ** 2)
                  for i, s in enumerate(summ) if P[i] > 1e-3)
        return h_id + h_q

    # ------------------------------------------------------------ probe choice
    def _candidates(self, level, beliefs):
        for o in level.objects:
            for slot in SLOTS:
                if slot not in beliefs[o.id]:
                    yield o.id, slot

    def _predict(self, half, slot, summ_draws, rng) -> np.ndarray:
        """Posterior-predictive raw readings for `slot` given joint posterior draws."""
        m, mu = np.exp(summ_draws["logm"]), summ_draws["mu"]
        if slot.startswith("push"):
            mm = push_mm(PUSH_FORCES[int(slot[4:])], m, mu)
            return np.maximum(0.0, mm + rng.normal(0, 1.0, len(m)))[:, None]
        if slot == "lift":
            return np.round(m * 1000 * (1 + rng.normal(0, 0.02, len(m))), 1)[:, None]
        snd = self.post.bank_sound(half, summ_draws["hollow"], summ_draws["rattle"], summ_draws["logm"], rng)
        return snd[:, :5] if slot == "tap" else snd[:, 5:]

    def _choose_probe(self, level, beliefs, rng, step):
        ids = [o.id for o in level.objects]
        half = level.objects[0].half
        xs = torch.cat([self._x(half, beliefs[i]) for i in ids])
        summ = self._summaries(xs, seed=self.seed + step)
        summ_list = [{k: (v[:, j] if np.ndim(v) == 2 else v[j]) for k, v in summ.items()} for j in range(len(ids))]
        draws = self.post.sample(xs, self.n_pred, seed=self.seed + 1000 + step)
        base = self._uncertainty(level, summ_list, np.random.default_rng(0))

        cands = list(self._candidates(level, beliefs))
        # skip probes that cannot matter: other objects in a slide task
        if level.task.kind == "slide":
            cands = [c for c in cands if c[0] == level.task.object_id]
        if not cands:
            return None
        # build all hypothetical conditions in one batch
        conds, owners = [], []
        for oid, slot in cands:
            j = ids.index(oid)
            dj = {k: v[:, j] for k, v in draws.items()}
            pred = self._predict(half, slot, dj, rng)
            for r in range(self.n_pred):
                conds.append(self._x(half, {**beliefs[oid], slot: list(pred[r])}))
            owners.append(j)
        X = torch.cat(conds)
        hyp = self._summaries(X, seed=self.seed + 2000 + step)
        gains = []
        for c, j in enumerate(owners):
            hs = []
            for r in range(self.n_pred):
                k = c * self.n_pred + r
                sl = list(summ_list)
                sl[j] = {key: (v[:, k] if np.ndim(v) == 2 else v[k]) for key, v in hyp.items()}
                hs.append(self._uncertainty(level, sl, np.random.default_rng(0)))
            gains.append(base - float(np.mean(hs)))
        oid, slot = cands[int(np.argmax(gains))]
        self.log.append({"probe": slot, "object": oid, "eig": round(float(max(gains)), 4)})
        if slot.startswith("push"):
            return {"type": "probe", "probe": "push", "object": oid, "force_N": PUSH_FORCES[int(slot[4:])]}
        return {"type": "probe", "probe": slot, "object": oid}

    # ------------------------------------------------------------ decision
    def decide(self, level, beliefs, rng):
        ids = [o.id for o in level.objects]
        half = level.objects[0].half
        xs = torch.cat([self._x(half, beliefs[i]) for i in ids])
        s = self.post.sample(xs, 4000, seed=self.seed + 999)
        ph, pr = self.post.discrete_probs(xs)
        summ = [{**{k: s[k][:, j] for k in s}, "ph": ph[j], "pr": pr[j]} for j in range(len(ids))]
        P = self._answer_probs(level, summ, rng)
        j = int(np.argmax(P))
        t = level.task
        keep = self._given_answer(level, summ[j])
        J = np.geomspace(*FLICK_RANGE, 600)
        d = flick_distance(J[:, None], np.exp(summ[j]["logm"][None, keep]), summ[j]["mu"][None, keep])
        p_hit = (np.abs(d - t.target_x) <= t.tol).mean(1)
        return ids[j], float(J[int(np.argmax(p_hit))]), float(P[j]), float(p_hit.max())

    def run(self, ep) -> dict:
        level = ep.level
        rng = np.random.default_rng(self.seed + level.seed)
        self.log = []
        obs = ep.reset()
        beliefs = {o.id: {} for o in level.objects}
        step = 0
        while obs.get("probes_left", 0) > 0:
            action = self._choose_probe(level, beliefs, rng, step)
            if action is None:
                break
            obs = ep.step(action)
            step += 1
            if "error" not in obs:
                self._record(obs, beliefs)
        oid, J, p_obj, p_hit = self.decide(level, beliefs, rng)
        self.log.append({"decision": oid, "impulse_Ns": round(J, 4), "p_object": round(p_obj, 3),
                         "p_hit": round(p_hit, 3)})
        return ep.step({"type": "flick", "object": oid, "impulse_Ns": J})

    def episode_meta(self) -> dict:
        return {"sbi_log": self.log}


# ---------------------------------------------------------------- calibration
def calibration(post: NpePosterior, records: np.ndarray, seed: int = 0, n_post: int = 500,
                levels=(0.5, 0.9)) -> dict:
    """Coverage of central credible intervals for log-mass and friction on held-out sims."""
    rng = np.random.default_rng(seed)
    masks = random_slot_masks(len(records), rng)
    xs = post.x(records[:, R_OBS:R_OBS + N_VALUES], masks, records[:, R_HALF])
    s = post.sample(xs, n_post, seed=seed)
    out = {"n": len(records)}
    for name, col, true in (("mass", "logm", records[:, R_LOGM]), ("friction", "mu", records[:, R_MU])):
        for lv in levels:
            lo, hi = np.quantile(s[col], [(1 - lv) / 2, (1 + lv) / 2], axis=0)
            out[f"{name}_{int(lv * 100)}"] = float(np.mean((true >= lo) & (true <= hi)))
    ph, pr = post.discrete_probs(xs)
    out["hollow_acc"] = float(np.mean((ph > 0.5) == (records[:, R_HOLLOW] > 0.5)))
    out["rattle_acc"] = float(np.mean((pr > 0.5) == (records[:, R_RATTLE] > 0.5)))
    return out

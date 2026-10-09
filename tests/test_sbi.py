import numpy as np
import pytest

from knock.agents import sbi_agent as S
from knock.env import Episode
from knock.level import Level

pytestmark = pytest.mark.skipif(not S.MODEL_PATH.exists(), reason="trained model not present")


@pytest.fixture(scope="module")
def post():
    return S.NpePosterior.load()


def test_model_is_small_and_loads(post):
    assert S.MODEL_PATH.stat().st_size < 3_000_000
    assert post.bank is not None and post.bank.shape[1] == S.R_OBS + S.N_VALUES


def test_posterior_sampling_shapes(post):
    n_obj = 3
    raw = np.zeros((n_obj, S.N_VALUES))
    mask = np.zeros((n_obj, S.N_SLOTS))
    x = post.x(raw, mask, np.full(n_obj, 0.05))
    s = post.sample(x, 100)
    for k in ("logm", "mu", "hollow", "rattle"):
        assert s[k].shape == (100, n_obj)
    assert np.all((s["mu"] >= S.MU_RANGE[0]) & (s["mu"] <= S.MU_RANGE[1]))
    ph, pr = post.discrete_probs(x)
    assert ph.shape == pr.shape == (n_obj,)
    # no probes: the posterior is the prior
    assert ph == pytest.approx(S.P_HOLLOW, abs=0.08)
    assert pr == pytest.approx(S.P_HOLLOW * S.P_RATTLE_GIVEN_HOLLOW, abs=0.06)


def test_lift_pins_down_mass(post):
    rec = S.simulate_record(123)
    mask = np.zeros((1, S.N_SLOTS))
    mask[0, S.SLOTS.index("lift")] = 1
    s = post.sample(post.x(rec[None, S.R_OBS:S.R_OBS + S.N_VALUES], mask, rec[None, S.R_HALF]), 400)
    assert abs(np.median(s["logm"]) - rec[S.R_LOGM]) < 0.1
    assert np.std(s["logm"]) < 0.1


def test_calibration_smoke(post):
    recs = np.stack([S.simulate_record(5_000_000 + i) for i in range(12)])
    cal = S.calibration(post, recs, n_post=300)
    # with 12 sims the coverage estimate is coarse; just catch gross miscalibration
    assert cal["mass_90"] >= 0.6 and cal["friction_90"] >= 0.6
    assert cal["mass_50"] <= 1.0


def test_agent_plays_an_episode(tmp_path):
    lv = Level.from_json(sorted(__import__("pathlib").Path("levels/dev").glob("*.json"))[2].read_text())
    ep = Episode(lv, tmp_path, render=False)
    r = S.SbiAgent(n_post=64, n_pred=4).run(ep)
    assert ep.done and r["probes_used"] == lv.task.budget

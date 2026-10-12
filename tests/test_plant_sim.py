"""P2: the press simulator is deterministic, multi-rate, degraded as promised, and labels are delayed and sparse."""
import numpy as np
import pytest

from plant.chunks import decode_chunk, encode_chunks
from plant.config import PlantConfig
from plant.press_sim import PressSimulator, load_pool


@pytest.fixture(scope="module")
def pool():
    return load_pool(PlantConfig())


def _sim(pool, **kw):
    return PressSimulator(PlantConfig(**kw), pool)


def test_deterministic_and_seed_dependent(pool):
    a, b, c = _sim(pool, seed=3).part(7), _sim(pool, seed=3).part(7), _sim(pool, seed=4).part(7)
    for k in a.streams:
        for x, y in zip(a.streams[k], b.streams[k]):
            assert np.array_equal(x, y)
    assert not np.array_equal(a.streams["stroke_force"][1][:50], c.streams["stroke_force"][1][:50])
    # order independence: requesting part 7 after other parts changes nothing
    s = _sim(pool, seed=3)
    s.part(2), s.part(30)
    assert np.array_equal(s.part(7).streams["stroke_force"][1], a.streams["stroke_force"][1])


def test_mixed_rates_jitter_and_drops(pool):
    p = _sim(pool).part(1)
    n = {k: len(v[0]) for k, v in p.streams.items()}
    assert n["stroke_force"] > 3 * n["stroke_position"] > 30 * n["die_temperature"] // 2
    t = p.streams["stroke_force"][0]
    dt = np.diff(t)
    assert dt.std() > 1e-5                    # jitter: not a uniform grid
    assert dt.max() > 1.5e-3                  # at least one dropped sample leaves a gap
    q = p.streams["stroke_force"][2]
    assert set(np.unique(q)) <= {0.0, 1.0, 2.0}


def test_drift_starts_at_the_chosen_part_and_grows(pool):
    s = _sim(pool, scenario="drift", n_parts=100, drift_start_frac=0.5, drift_slope_kN_per_part=0.1)
    assert s.force_offset_kN(49) == 0.0
    assert s.force_offset_kN(50) == pytest.approx(0.1) and s.force_offset_kN(59) == pytest.approx(1.0)
    assert _sim(pool, scenario="nominal").force_offset_kN(300) == 0.0


def test_idle_force_reflects_offset(pool):
    s = _sim(pool, scenario="drift", n_parts=100, drift_slope_kN_per_part=0.2)
    def idle_mean(i):
        t, v, q = s.part(i).streams["stroke_force"]
        t0 = t[0]
        m = (t - t0 < 0.15) & (q < 2)
        return v[m].mean()
    assert abs(idle_mean(10)) < 0.3 and idle_mean(79) > 4.0


def test_labels_one_in_k_and_delayed(pool):
    s = _sim(pool, label_every=5, label_delay_parts=(3, 8), n_parts=200)
    labs = [s.label_for(i) for i in range(200)]
    sampled = [l for l in labs if l]
    assert len(sampled) == 40
    assert all(3 <= l["due_index"] - l["part_id"] <= 8 for l in sampled)
    due = [l for i in range(200) for l in s.labels_due(i)]
    assert {l["part_id"] for l in due} == {l["part_id"] for l in sampled if l["due_index"] < 200}
    truth = s.part(9).truth["id_change_pct"]
    assert abs(s.label_for(9)["id_change_pct"] - truth) < 2.0


def test_chunk_roundtrip():
    t = np.linspace(0, 0.55, 56)
    ch = encode_chunks(5, t, t * 2, np.zeros_like(t), 0.0, 0.1)
    assert len(ch) == 6
    pid, seq, tt, vv, qq = decode_chunk(ch[2])
    assert pid == 5 and seq == 2 and np.all((tt >= 0.2 - 1e-12) & (tt < 0.3 + 1e-9))
    assert np.allclose(vv, tt * 2)

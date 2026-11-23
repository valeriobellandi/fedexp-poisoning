"""Real-data testbed (UCI Heart Disease) with the mock model."""

import yaml

from fedexp import heart
from fedexp.llm import MockLLM
from fedexp.run import run_once


def cfg(name):
    c = yaml.safe_load(open(f"configs/{name}.yaml"))
    c["llm"]["backend"] = "mock"
    return c


def test_data_and_profiles():
    data = heart.load()
    assert {h: len(v) for h, v in data.items()} == {
        "cleveland": 303, "hungary": 293, "switzerland": 122, "va": 147}
    assert len(heart.PROFILES) == 32
    pooled = heart.pooled_majority()
    assert sum(n for _, _, n in pooled.values()) == 865


def test_rule_round_trip_and_generalise():
    line = heart.format_rule("H07", "cleveland", True, 7, 9)
    assert heart.parse_rule(line) == ("H07", "cleveland", True, 7, 9)
    assert heart.parse_rule(heart.generalise([line])[0]) == ("H07", "all", True, 7, 9)
    assert heart.parse_rule("Profile H07 at all hospitals: heart disease absent.") == (
        "H07", "all", False, None, None)


def test_split_is_disjoint_and_seeded():
    sites = ["cleveland"] * 3 + ["hungary"] * 2 + ["va", "switzerland"]
    tr, te = heart.split(0, sites)
    tr2, _ = heart.split(0, sites)
    assert tr == tr2
    ids = [p["id"] for v in tr.values() for p in v]
    test_ids = {p["id"] for v in te.values() for p in v}
    assert len(ids) == len(set(ids)) and not set(ids) & test_ids


def test_runs_score_the_same_clients_and_evidence_cuts_spurious_disputes():
    a = run_once(cfg("heart_a1"), 0, MockLLM())
    b = run_once(cfg("heart_benign"), 0, MockLLM())
    assert a["n_benign"] == b["n_benign"] == 5
    assert a["acc_prior"] == b["acc_prior"] and a["acc_local"] == b["acc_local"]
    spur_plain = sum(len(run_once(cfg("heart_plain_benign"), s, MockLLM())["disputes"])
                     for s in range(3))
    spur_ev = sum(len(run_once(cfg("heart_ev_benign"), s, MockLLM())["disputes"])
                  for s in range(3))
    assert spur_ev < spur_plain

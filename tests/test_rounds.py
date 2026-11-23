"""Multi-round runs with the mock model."""

import copy

import yaml

from fedexp.llm import MockLLM
from fedexp.prompts import parse_item
from fedexp.run import run_once


def cfg(name, **rounds):
    c = yaml.safe_load(open(f"configs/{name}.yaml"))
    c["llm"]["backend"] = "mock"
    c.setdefault("rounds", {}).update(rounds)
    return c


def test_round_one_equals_single_round():
    multi = run_once(cfg("mr_a1"), 0, MockLLM())
    single_cfg = copy.deepcopy(cfg("mr_a1"))
    single_cfg.pop("rounds")
    single_cfg["attack"].pop("rounds")
    single = run_once(single_cfg, 0, MockLLM())
    assert multi["rounds"][0]["acc_fed"] == single["acc_fed"]
    assert multi["rounds"][0]["attack"]["asr_all_benign"] == single["attack"]["asr_all_benign"]


def test_hit_and_run_attackers_leave_and_provenance_blocks_carriers():
    hit = run_once(cfg("mr_a1_hit"), 1, MockLLM())
    for raw in hit["raw"]["rounds"][1:]:
        assert "1" not in raw["uploads"] and "5" not in raw["uploads"]
    own = run_once(cfg("mr_a1_own"), 1, MockLLM())
    assert all(r["honest_carriers"] == [] for r in own["rounds"])
    allup = run_once(cfg("mr_a1"), 1, MockLLM())
    assert allup["rounds"][0]["honest_carriers"] == []
    assert len(allup["rounds"][1]["honest_carriers"]) > 0


def test_feedback_makes_absorbed_codes_first_hand():
    res = run_once(cfg("mr_a1_hit_fb"), 0, MockLLM())
    n_notes = [sum(len(v) for k, v in r["notes"].items() if k not in ("1", "5"))
               for r in res["raw"]["rounds"]]
    first = res["raw"]["rounds"][0]["notes"]
    assert all(parse_item(x) for v in first.values() for x in v)
    assert n_notes[1] >= n_notes[0]

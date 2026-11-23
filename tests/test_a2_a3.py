"""Tests for the conditional world, A2 and A3. Model-free, or with MockLLM."""

import random

from fedexp.attacks import apply_a2, apply_a3
from fedexp.llm import MockLLM
from fedexp.prompts import ALL, format_item, item_code, parse_cond_item, parse_item
from fedexp.protocol import distil_all, disputes, generalise
from fedexp.world import make_clients, make_world

SITES = ["coastal", "coastal", "mountain", "mountain", "urban", "coastal"]
COND = "during a storm"


def setup(seed=0, cond=True):
    rng = random.Random(seed)
    world = make_world(rng, 40, 8, SITES, 6, COND if cond else None, 6 if cond else 0)
    clients = make_clients(rng, world, SITES, 0.4, 2)
    return rng, world, clients


def test_world_without_condition_is_unchanged():
    _, a, ca = setup(3, cond=False)
    rng = random.Random(3)
    b = make_world(rng, 40, 8, SITES, 6)
    cb = make_clients(rng, b, SITES, 0.4, 2)
    assert a == b and ca == cb and a.condition is None


def test_conditional_truth_and_records():
    _, world, clients = setup()
    assert len(world.conditional) == 6
    for code, proc in world.conditional.items():
        for site in set(SITES):
            assert world.truth(code, site, True) == proc != world.truth(code, site)
    for c in clients:
        for rec in c.records:
            if len(rec) == 3:
                assert rec[0] in world.conditional and rec[2] == COND


def test_conditional_items_parse_and_generalise():
    line = format_item("K12", "coastal", "P3", COND)
    assert parse_item(line) is None
    assert parse_cond_item(line) == ("K12", "coastal", COND, "P3")
    assert item_code(line) == "K12"
    assert parse_cond_item(format_item("K12", ALL, "P3")) is None
    assert parse_cond_item(generalise([line])[0]) == ("K12", ALL, COND, "P3")


def _uploads(world, clients):
    notes = distil_all(MockLLM(), clients, world.condition)
    return {i: generalise(lines) for i, lines in notes.items()}


def test_honest_uploads_raise_no_dispute_when_conditions_are_implied():
    for seed in range(20):
        _, world, clients = setup(seed)
        up = _uploads(world, clients)
        sites = {c.idx: c.site for c in clients}
        assert disputes(up, sites, 2, "implied") == set()


def test_a2_is_invisible_unless_implied_answers_are_compared():
    caught = 0
    for seed in range(20):
        rng, world, clients = setup(seed)
        up = _uploads(world, clients)
        before = {i: list(v) for i, v in up.items()}
        poisons = apply_a2(rng, world, clients, up, [1, 5], 10)
        for p in poisons:
            assert p["code"] not in world.conditional
            assert p["bad"] != world.truth(p["code"], "coastal")
        for i in (1, 5):
            # Honest lines kept, conditional poison added.
            assert set(before[i]) <= set(up[i])
            added = [parse_cond_item(x) for x in up[i] if x not in before[i]]
            assert {(a[0], a[3]) for a in added} == {(p["code"], p["bad"]) for p in poisons}
        sites = {c.idx: c.site for c in clients}
        assert not any({1, 5} & set(pair) for pair in disputes(up, sites, 2, "ignore"))
        pairs = disputes(up, sites, 2, "implied")
        caught += frozenset((0, 1)) in pairs and frozenset((0, 5)) in pairs
    assert caught >= 10


def test_a3_uploads_the_overrides_for_all_sites():
    rng, world, clients = setup(cond=False)
    up = _uploads(world, clients)
    poisons = apply_a3(rng, world, clients, up, [1, 5])
    assert {p["code"]: p["bad"] for p in poisons} == world.overrides["coastal"]
    for i in (1, 5):
        claimed = {parse_item(x)[0]: parse_item(x) for x in up[i] if parse_item(x)}
        for p in poisons:
            assert claimed[p["code"]] == (p["code"], ALL, p["bad"])
    sites = {c.idx: c.site for c in clients}
    assert not any({1, 5} & set(pair) for pair in disputes(up, sites, 2))

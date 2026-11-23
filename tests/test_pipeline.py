"""Pipeline checks with the mock model. They validate the plumbing and the
metrics, not any claim about real models."""

import random

import pytest

from fedexp import metrics
from fedexp.attacks import apply_a1
from fedexp.llm import MockLLM
from fedexp.prompts import (ALL, clean_lines, format_item, lines_for_code,
                            merge_prompt, parse_answer, parse_item)
from fedexp.protocol import aggregate
from fedexp.run import run_once
from fedexp.world import Client, World, make_clients, make_world

CFG = {
    "world": {"n_codes": 40, "n_procedures": 8, "overrides_per_site": 6},
    "clients": {"site_types": ["coastal", "coastal", "mountain", "mountain", "urban"],
                "coverage": 0.4, "records_per_code": 2},
    "llm": {"backend": "mock"},
}
A1 = {"kind": "a1", "malicious": [1], "n_targets": 10, "claim_scope": "global"}


def test_item_round_trip():
    line = format_item("K17", "coastal", "P3")
    assert parse_item(line) == ("K17", "coastal", "P3")
    assert parse_item("- " + format_item("K17", ALL, "P3")) == ("K17", ALL, "P3")
    assert parse_item("Some free text about K17.") is None


def test_text_helpers():
    assert clean_lines("1. a\n\n- b\n  * c ") == ["a", "b", "c"]
    assert parse_answer("The procedure is p4.") == "P4"
    assert parse_answer("unknown") is None
    # K1 must not match K17.
    assert lines_for_code(["rule for K17", "rule for K1"], "K1") == ["rule for K1"]


def test_world_truth_uses_override():
    w = World(["K10"], ["P1", "P2"], {"K10": "P1"}, {"coastal": {"K10": "P2"}})
    assert w.truth("K10", "coastal") == "P2"
    assert w.truth("K10", "urban") == "P1"


def test_same_seed_gives_same_world_with_and_without_attack():
    benign = run_once({**CFG, "attack": {"kind": "none"}}, 3, MockLLM())
    attacked = run_once({**CFG, "attack": A1}, 3, MockLLM())
    assert benign["raw"]["global_rule"] == attacked["raw"]["global_rule"]
    assert benign["raw"]["overrides"] == attacked["raw"]["overrides"]


def test_a1_rewrites_only_malicious_upload():
    rng = random.Random(0)
    sites = CFG["clients"]["site_types"]
    world = make_world(rng, 40, 8, sites, 6)
    clients = make_clients(rng, world, sites, 0.4, 2)
    uploads = {c.idx: [format_item(code, c.site, world.truth(code, c.site))
                       for code in c.train_codes] for c in clients}
    before = {k: list(v) for k, v in uploads.items()}
    poisons = apply_a1(rng, world, clients, uploads, [1], 10, "global")

    assert len(poisons) == 10
    for idx in (0, 2, 3, 4):
        assert uploads[idx] == before[idx]
    for p in poisons:
        code, bad = p["code"], p["bad"]
        # The injected procedure is wrong at every site type.
        assert all(world.truth(code, s) != bad for s in set(sites))
        # The attacker says exactly one thing about a target code.
        assert lines_for_code(uploads[1], code) == [format_item(code, ALL, bad)]


def test_attack_metrics_on_a_hand_built_case():
    world = World(["K10", "K11"], ["P1", "P2", "P3"],
                  {"K10": "P1", "K11": "P2"}, {})
    benign = [Client(0, "urban", ["K10"]), Client(2, "urban", [])]
    poisons = [{"code": "K10", "bad": "P3", "honest_support": 1, "scopes": [ALL]},
               {"code": "K11", "bad": "P3", "honest_support": 0, "scopes": [ALL]}]
    library = [format_item("K10", ALL, "P1"),   # poison on K10 was dropped
               format_item("K11", ALL, "P3")]   # poison on K11 survived
    answers = {0: {"K10": "P1", "K11": "P3"}, 2: {"K10": "P1", "K11": "P2"}}

    m = metrics.attack_metrics(world, benign, poisons, library, answers)
    assert m["survival"] == 0.5
    assert m["survival_contested"] == 0.0
    assert m["survival_uncontested"] == 1.0
    assert m["asr"] == 0.25                      # 1 adoption out of 4 cases
    assert m["asr_code_in_own_notes"] == 0.0     # client 0 on K10
    assert m["asr_code_not_in_own_notes"] == pytest.approx(1 / 3)


def test_site_scoped_poison_only_concerns_that_site():
    world = World(["K10"], ["P1", "P2"], {"K10": "P1"}, {})
    benign = [Client(0, "urban", []), Client(1, "coastal", [])]
    poisons = [{"code": "K10", "bad": "P2", "honest_support": 0, "scopes": ["coastal"]}]
    m = metrics.attack_metrics(world, benign, poisons,
                               [format_item("K10", "coastal", "P2")],
                               {0: {"K10": "P1"}, 1: {"K10": "P2"}})
    assert [(c["client"], c["in_scope"]) for c in m["cases"]] == [(0, False), (1, True)]
    assert m["asr"] == 1.0              # only the coastal client is addressed
    assert m["asr_all_benign"] == 0.5   # the urban client did not pick it up


def test_minority_retention_follows_precedence():
    world = World(["K10", "K11"], ["P1", "P2"], {"K10": "P1", "K11": "P1"},
                  {"coastal": {"K10": "P2", "K11": "P2"}})
    benign = [Client(0, "coastal", ["K10", "K11"])]
    library = [format_item("K10", ALL, "P1"), format_item("K10", "coastal", "P2"),
               format_item("K11", ALL, "P1")]   # the K11 override was lost
    m = metrics.minority_retention(world, benign, library)
    assert m == {"minority_reported": 2, "minority_retention": 0.5}


@pytest.mark.parametrize("seed", range(5))
def test_mock_run_invariants(seed):
    benign = run_once({**CFG, "attack": {"kind": "none"}}, seed, MockLLM())
    assert benign["acc_fed"] > benign["acc_local"]
    assert benign["library_unparsed_lines"] == 0
    assert "attack" not in benign

    res = run_once({**CFG, "attack": A1}, seed, MockLLM())
    atk = res["attack"]
    assert res["n_benign"] == 4
    # Nobody contradicts an uncontested poison, so the mock keeps it.
    if atk["n_uncontested"]:
        assert atk["survival_uncontested"] == 1.0
    # The mock solver trusts its own notes over the library.
    assert atk["asr_code_in_own_notes"] in (0.0, None)
    # A poison that did not survive cannot be adopted.
    assert all(c["survived"] or not c["adopted"] for c in atk["cases"])


class RecordingLLM(MockLLM):
    def __init__(self):
        self.prompts = []

    def generate(self, prompts, max_new_tokens=0):
        self.prompts.extend(prompts)
        return super().generate(prompts, max_new_tokens)


def test_per_code_aggregation_groups_reports_by_code():
    uploads = {
        0: [format_item("K10", "coastal", "P1"), format_item("K11", "coastal", "P2")],
        1: [format_item("K10", "mountain", "P1"), "free text the server cannot use"],
        2: [format_item("K11", "urban", "P3")],
    }
    llm = RecordingLLM()
    library = aggregate(llm, uploads, "per_code")

    assert len(llm.prompts) == 2                       # one call per code
    k10 = next(p for p in llm.prompts if "alarm code K10." in p)
    assert "K11" not in k10 and "Client 1:" in k10 and "Client 2:" in k10
    assert "free text" not in "".join(llm.prompts)
    # K10: both sites agree -> one general rule. K11: 1 vs 1 -> a general
    # rule plus the other site's exception, whichever way the tie falls.
    assert [parse_item(l) for l in library if "K10" in l] == [("K10", ALL, "P1")]
    k11 = {parse_item(l) for l in library if "K11" in l}
    assert len(k11) == 2 and {p for _, _, p in k11} == {"P2", "P3"}


def test_merge_prompt_names_only_its_code():
    prompt = merge_prompt("K10", [(0, format_item("K10", "coastal", "P1"))])
    assert prompt.startswith("[ROLE: merge]")
    assert "Client 1: For alarm code K10 at coastal sites, apply procedure P1." in prompt


def test_unknown_aggregator_is_rejected():
    with pytest.raises(ValueError):
        aggregate(MockLLM(), {0: []}, "nope")


@pytest.mark.parametrize("scope", ["global", "site"])
def test_per_code_run_with_mock(scope):
    cfg = {**CFG, "server": {"aggregator": "per_code"},
           "attack": {**A1, "claim_scope": scope}}
    res = run_once(cfg, 0, MockLLM())
    atk = res["attack"]
    assert res["library_unparsed_lines"] == 0
    assert res["acc_fed"] > res["acc_local"]
    assert len(atk["cases"]) == 4 * 10                  # every benign client, every target
    in_scope = sum(c["in_scope"] for c in atk["cases"])
    assert in_scope == (40 if scope == "global" else 10)  # one honest coastal client
    assert all(c["survived"] or not c["adopted"] for c in atk["cases"])


def test_paired_comparison_uses_the_same_honest_clients():
    from fedexp.summarize import paired
    benign = run_once({**CFG, "attack": {"kind": "none"}}, 1, MockLLM())
    attacked = run_once({**CFG, "attack": A1}, 1, MockLLM())
    p = paired(attacked, benign)
    assert p["paired_acc_attacked"] == pytest.approx(attacked["acc_fed"])
    assert p["paired_acc_diff"] == pytest.approx(
        p["paired_acc_attacked"] - p["paired_acc_benign"])
    assert p["paired_target_acc_attacked"] <= p["paired_target_acc_benign"]


GEN = {**CFG, "clients": {**CFG["clients"], "upload_scope": "general"},
       "server": {"aggregator": "per_code", "show_client_site": True}}
SIX = {**GEN, "clients": {**GEN["clients"],
                          "site_types": CFG["clients"]["site_types"] + ["coastal"]}}


def test_generalise_drops_the_site_qualifier():
    from fedexp.protocol import generalise
    assert generalise([format_item("K10", "coastal", "P1"), "free text"]) == [
        format_item("K10", ALL, "P1"), "free text"]


def test_merge_prompt_shows_client_sites_when_asked():
    rule = format_item("K10", ALL, "P1")
    prompt = merge_prompt("K10", [(0, rule)], {0: "coastal"})
    assert f"Client 1 (coastal site): {rule}" in prompt


def test_general_uploads_keep_local_notes_site_scoped():
    res = run_once({**GEN, "attack": A1}, 0, MockLLM())
    raw = res["raw"]
    assert all(parse_item(l)[1] == ALL for v in raw["uploads"].values() for l in v)
    assert all(parse_item(l)[1] != ALL for v in raw["notes"].values() for l in v)
    # The attacker's lines look like everybody else's.
    assert {parse_item(l)[1] for l in raw["uploads"]["1"]} == {ALL}
    assert res["acc_fed"] > res["acc_local"]


def test_sixth_client_leaves_the_first_five_unchanged():
    five = run_once({**GEN, "attack": {"kind": "none"}}, 2, MockLLM())["raw"]
    six = run_once({**SIX, "attack": {"kind": "none"}}, 2, MockLLM())["raw"]
    assert five["global_rule"] == six["global_rule"]
    assert five["overrides"] == six["overrides"]
    assert five["clients"] == six["clients"][:5]
    assert six["clients"][5]["site"] == "coastal"


def test_collusion_uses_the_same_poison_as_the_single_attacker():
    f1 = run_once({**SIX, "attack": A1}, 4, MockLLM())
    f2 = run_once({**SIX, "attack": {**A1, "malicious": [1, 5]}}, 4, MockLLM())
    key = lambda r: [(p["code"], p["bad"]) for p in r["raw"]["poisons"]]
    assert key(f1) == key(f2)
    assert f1["n_benign"] == 5 and f2["n_benign"] == 4
    # Two colluders can only make the poison at least as present as one.
    assert f2["attack"]["survival"] >= f1["attack"]["survival"]
    for case in f2["attack"]["cases"]:
        assert case["same_site"] == (case["client"] == 0)


def test_baseline_key_separates_configurations():
    from fedexp.summarize import _baseline_key
    a = run_once({**CFG, "attack": {"kind": "none"}}, 0, MockLLM())
    b = run_once({**CFG, "attack": A1}, 0, MockLLM())
    c = run_once({**GEN, "attack": A1}, 0, MockLLM())
    d = run_once({**CFG, "server": {"aggregator": "single_prompt"}, "attack": A1}, 0, MockLLM())
    assert _baseline_key(a) == _baseline_key(b) == _baseline_key(d)
    assert _baseline_key(a) != _baseline_key(c)


def test_corroborate_applies_the_support_threshold():
    from fedexp.protocol import corroborate
    sites = {0: "coastal", 1: "coastal", 2: "urban"}
    uploads = {0: [format_item("K10", ALL, "P1"), format_item("K11", ALL, "P2")],
               1: [format_item("K10", ALL, "P1"), format_item("K11", ALL, "P3")],
               2: [format_item("K10", ALL, "P4"), format_item("K11", ALL, "P2")]}
    library = [format_item("K10", ALL, "P4"),        # one backer            -> removed
               format_item("K10", "coastal", "P1"),  # two coastal backers   -> kept
               format_item("K11", ALL, "P2"),        # two backers, any site -> kept
               format_item("K11", "coastal", "P2"),  # one coastal backer    -> removed
               format_item("K11", "urban", "P9"),    # nobody reported P9    -> removed
               "free text"]
    kept, removed = corroborate(library, uploads, sites, 2)
    assert kept == [library[1], library[2]]
    assert removed == [library[0], library[3], library[4], library[5]]
    assert corroborate(library[:5], uploads, sites, 1)[0] == library[:4]


def test_shuffled_order_is_reproducible_and_changes_only_the_order():
    uploads = {i: [format_item("K10", ALL, f"P{i + 1}")] for i in range(6)}
    sites = {i: "urban" for i in range(6)}

    def prompt_for(order_seed):
        llm = RecordingLLM()
        aggregate(llm, uploads, "per_code", sites, order_seed)
        return llm.prompts[0]

    fixed, a, b = prompt_for(None), prompt_for("order-0"), prompt_for("order-0")
    assert a == b and a != fixed
    assert sorted(a.splitlines()) == sorted(fixed.splitlines())
    with pytest.raises(ValueError):
        aggregate(MockLLM(), uploads, "single_prompt", sites, "order-0")


def test_threshold_stops_one_attacker_but_not_two_colluders():
    server = {**SIX["server"], "report_order": "shuffled", "min_support": 2}
    f1 = run_once({**SIX, "server": server, "attack": A1}, 0, MockLLM())
    f2 = run_once({**SIX, "server": server,
                   "attack": {**A1, "malicious": [1, 5]}}, 0, MockLLM())
    assert f1["attack"]["survival"] == 0.0
    # asr is not asserted to be zero: a client left without any rule guesses,
    # and the guess can coincide with the attacker's procedure.
    assert f1["defence_removed_rules"] > 0
    assert f2["attack"]["survival"] > 0.0
    # The world does not depend on the server settings.
    plain = run_once({**SIX, "attack": A1}, 0, MockLLM())
    assert plain["raw"]["poisons"] == f1["raw"]["poisons"]
    assert plain["raw"]["uploads"] == f1["raw"]["uploads"]


def test_disputes_need_same_site_and_enough_contradictions():
    from fedexp.protocol import disputes
    sites = {0: "coastal", 1: "coastal", 2: "urban", 3: "coastal"}
    r = lambda code, proc: format_item(code, ALL, proc)
    uploads = {
        0: [r("K10", "P1"), r("K11", "P1"), r("K12", "P1")],
        1: [r("K10", "P2"), r("K11", "P2"), r("K12", "P1")],   # 2 clashes with 0
        2: [r("K10", "P3"), r("K11", "P3"), r("K12", "P3")],   # another site type
        3: [r("K10", "P2"), r("K13", "P1")],                   # 1 clash with 0
    }
    assert disputes(uploads, sites, 2) == {frozenset((0, 1))}
    assert disputes(uploads, sites, 1) == {frozenset((0, 1)), frozenset((0, 3))}
    assert disputes(uploads, sites, 3) == set()


def test_recipient_exclusions_follow_personal_evidence():
    from fedexp.protocol import recipient_exclusions
    sites = {0: "coastal", 1: "coastal", 2: "urban", 3: "coastal"}
    r = lambda code, proc: format_item(code, ALL, proc)
    uploads = {
        0: [r("K10", "P1"), r("K11", "P1")],
        1: [r("K10", "P2"), r("K11", "P2")],     # in dispute with 0 only
        2: [r("K10", "P3")],
        3: [r("K12", "P1")],                     # shares no code with 0 or 1
    }
    ex = recipient_exclusions(uploads, sites, 2)
    assert ex[0] == {1} and ex[1] == {0}         # each side drops the other
    assert ex[2] == {0, 1}                       # another site type drops both
    assert ex[3] == set()                        # same site, no evidence: keeps both


def test_dispute_defence_costs_nothing_without_disputes():
    server = {**SIX["server"], "report_order": "shuffled"}
    plain = run_once({**SIX, "server": server, "attack": {"kind": "none"}}, 0, MockLLM())
    defended = run_once({**SIX, "server": {**server, "dispute_threshold": 2},
                         "attack": {"kind": "none"}}, 0, MockLLM())
    assert defended["n_libraries"] == 1
    assert defended["dropped_reports_per_recipient"] == 0
    assert defended["raw"]["answers_fed"] == plain["raw"]["answers_fed"]


@pytest.mark.parametrize("seed", range(4))
def test_dispute_defence_against_colluders(seed):
    server = {**SIX["server"], "report_order": "shuffled", "dispute_threshold": 2}
    res = run_once({**SIX, "server": server,
                    "attack": {**A1, "malicious": [1, 5]}}, seed, MockLLM())
    excluded = {i: set(v["excluded"]) for v in res["raw"]["views"] for i in v["recipients"]}
    assert set(excluded) == {0, 2, 3, 4}
    for case in res["attack"]["cases"]:
        # A recipient whose library leaves out both colluders cannot receive
        # the poison, whatever the colluders agree on.
        if {1, 5} <= excluded[case["client"]]:
            assert not case["survived"]
    # Clients of other site types leave out the two sides of a dispute alike.
    for i in (2, 3, 4):
        assert excluded[i] in (set(), {0, 1}, {0, 5}, {0, 1, 5})
    # The honest coastal client never leaves itself out.
    assert 0 not in excluded[0]


def test_fallback_policy_fills_only_uncovered_undisputed_codes():
    from fedexp.protocol import recipient_uploads
    sites = {0: "coastal", 1: "coastal", 2: "urban"}
    r = lambda code, proc: format_item(code, ALL, proc)
    uploads = {
        0: [r("K10", "P1"), r("K11", "P1"), r("K12", "P1"), r("K14", "P5")],
        1: [r("K10", "P2"), r("K11", "P2"), r("K12", "P1"), r("K13", "P7")],
        2: [r("K10", "P3")],
    }
    # Client 2 cannot tell 0 from 1. Exclude: it keeps only its own reports.
    assert recipient_uploads(uploads, sites, 2, 2, "exclude") == {2: uploads[2]}
    # Fallback: K10 is covered by client 2 itself, K11 is disputed between
    # 0 and 1; K12 (both agree), K13 and K14 (one source each) fill the gaps.
    assert recipient_uploads(uploads, sites, 2, 2, "fallback") == {
        2: uploads[2],
        0: [r("K12", "P1"), r("K14", "P5")],
        1: [r("K12", "P1"), r("K13", "P7")],
    }
    # Client 0 has its own evidence against client 1: no fallback from it.
    assert recipient_uploads(uploads, sites, 0, 2, "fallback") == {
        0: uploads[0], 2: uploads[2]}
    with pytest.raises(ValueError):
        recipient_uploads(uploads, sites, 2, 2, "nope")


def test_fallback_keeps_more_reports_than_exclusion():
    server = {**SIX["server"], "report_order": "shuffled", "dispute_threshold": 2}
    attack = {**A1, "malicious": [1, 5]}
    ex = run_once({**SIX, "server": server, "attack": attack}, 0, MockLLM())
    fb = run_once({**SIX, "server": {**server, "dispute_policy": "fallback"},
                   "attack": attack}, 0, MockLLM())
    assert 0 < fb["dropped_reports_per_recipient"] < ex["dropped_reports_per_recipient"]
    # The honest client of the attackers' site keeps them out in both cases.
    assert ex["attack"]["survival_same_site"] == fb["attack"]["survival_same_site"] == 0.0
    assert ex["attack"]["survival_other_site"] == 0.0


def test_exposure_replays_the_attack_on_stored_notes():
    from fedexp.exposure import exposure, rebuild
    server = {**SIX["server"], "report_order": "shuffled"}
    benign = run_once({**SIX, "server": server, "attack": {"kind": "none"}}, 0, MockLLM())
    world, clients, uploads = rebuild(benign)
    assert uploads == {int(k): v for k, v in benign["raw"]["uploads"].items()}

    one = exposure(benign, [1, 5], 1, 20, 2)
    # A single contradiction can never reach a threshold of two.
    assert one["attackers_all_in_dispute"] == 0.0
    assert one["exclude_same_site"] == one["exclude_other_site"] == 1.0
    flood = exposure(benign, [1, 5], 40, 5, 2)
    assert flood["same_site_recipient_protected"] == 1.0
    assert flood["exclude_same_site"] == flood["exclude_other_site"] == 0.0
    assert flood["fallback_same_site"] == 0.0
    assert 0.0 < flood["fallback_other_site"] < 40.0
    # No honest client of the attacker's site type: nothing is filtered.
    alone = exposure(benign, [4], 10, 5, 2)
    assert alone["attackers_all_in_dispute"] == 0.0
    assert alone["exclude_other_site"] == alone["fallback_other_site"] == 10.0


def test_flood_attack_runs():
    server = {**SIX["server"], "report_order": "shuffled", "dispute_threshold": 2,
              "dispute_policy": "fallback"}
    res = run_once({**SIX, "server": server,
                    "attack": {**A1, "malicious": [1, 5], "n_targets": 40}}, 0, MockLLM())
    assert len(res["raw"]["poisons"]) == 40
    assert res["attack"]["survival_same_site"] == 0.0
    assert 0.0 < res["attack"]["survival_other_site"] < 1.0


def test_sim_source_without_errors_is_exact():
    from fedexp.ablation import sim_source
    world, clients, uploads = sim_source(0, ["coastal", "coastal", "urban"], 0.4, 0.0, 0.0)
    for c in clients:
        items = [parse_item(l) for l in uploads[c.idx]]
        assert sorted(code for code, _, _ in items) == sorted(c.train_codes)
        assert all(scope == ALL and proc == world.truth(code, c.site)
                   for code, scope, proc in items)


def test_sim_source_error_rates():
    from fedexp.ablation import sim_source
    missing = invented = seen = n = 0
    for seed in range(300):
        world, clients, uploads = sim_source(seed, ["coastal", "urban"], 0.4, 0.1, 0.5)
        for c in clients:
            codes = {parse_item(l)[0] for l in uploads[c.idx]}
            seen += len(c.train_codes)
            missing += len(set(c.train_codes) - codes)
            invented += len(codes - set(c.train_codes))
            n += 1
    assert abs(missing / seen - 0.1) < 0.02
    assert abs(invented / n - 0.5) < 0.1


def test_thin_lowers_coverage_and_keeps_invented_rules():
    from fedexp.ablation import thin
    world, clients, uploads = rebuild_for_tests()
    extra = format_item(next(c for c in world.codes if c not in clients[0].train_codes), ALL, "P1")
    uploads[0] = uploads[0] + [extra]
    _, small, up = thin(random.Random(0), world, clients, uploads, 0.2)
    for before, after in zip(clients, small):
        assert len(after.train_codes) == 8 and set(after.train_codes) <= set(before.train_codes)
        codes = {parse_item(l)[0] for l in up[after.idx]}
        assert codes - set(after.train_codes) <= codes - set(before.train_codes)
    assert extra in up[0]


def rebuild_for_tests():
    from fedexp.exposure import rebuild
    server = {**SIX["server"], "report_order": "shuffled"}
    return rebuild(run_once({**SIX, "server": server, "attack": {"kind": "none"}}, 0, MockLLM()))


def test_attack_once_counts_false_rules_on_unknown_codes():
    from fedexp.ablation import Tally, attack_once
    world, clients, uploads = rebuild_for_tests()
    tally = Tally()
    attack_once(tally, random.Random(1), world, clients, uploads, [1, 5], 40, [2])
    # Without defence every false rule on a code a recipient does not know arrives.
    for c in clients:
        if c.idx in (1, 5):
            continue
        group = "same" if c.site == "coastal" else "other"
        assert tally.n[("none", group, "false")] > 0
    unknown_other = [40 - len(c.train_codes) for c in clients if c.site != "coastal"]
    assert tally.mean(("none", "other", "false")) == pytest.approx(
        sum(unknown_other) / len(unknown_other))
    # Flooding is always noticed by the honest coastal client, which is then protected.
    assert tally.mean((2, "same", "protected")) == 1.0
    assert tally.mean(((2, "exclude"), "same", "false")) == 0.0

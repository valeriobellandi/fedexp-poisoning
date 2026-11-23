"""Several federation rounds, with the library absorbed into local notes.

Round 1 is the single round of run.py. After each round every benign client
keeps, as its own notes, its first-hand rules plus one rule for every other
code: the answer it gave in that round with the library ("absorption"). In
the next round it uploads these notes again, so a false rule it adopted
comes back to the server from an honest client. This is how experience
sharing works when what an agent used is stored as experience, and it is
the mechanism that can make a poison outlive its source.

Options, under the `rounds` key of the config:
  n:        number of rounds (> 1 to use this module);
  upload:   "all" (default): a client uploads its notes, absorbed rules
            included; "own": only its first-hand rules (provenance);
  feedback: probability that, in a round, a client meets an incident of an
            absorbed code and learns its true procedure (default 0); the
            rule then becomes first-hand.
and under `attack`:
  rounds:   "all" (default): the attackers upload the same poison every
            round; "first": they poison round 1 and then leave the
            federation.
and under `server`:
  sticky_disputes: a dispute, once found, holds in every later round.
"""

from __future__ import annotations

import copy
import random

from . import metrics
from .attacks import apply_a1
from .prompts import ALL, format_item, lines_for_code, parse_item
from .protocol import distil_all, generalise, solve_all
from .world import make_clients, make_world


def poison_uploads(uploads: dict[int, list[str]], malicious: list[int],
                   poisons: list[dict], rng: random.Random) -> None:
    """Rewrite the malicious uploads with the poisons of round 1."""
    for k, idx in enumerate(malicious):
        bad = {p["code"]: (p["scopes"][k], p["bad"]) for p in poisons}
        kept = [ln for ln in uploads[idx] if not any(lines_for_code([ln], c) for c in bad)]
        lines = kept + [format_item(c, s, b) for c, (s, b) in sorted(bad.items())]
        rng.shuffle(lines)
        uploads[idx] = lines


def run_rounds(cfg: dict, seed: int, llm) -> dict:
    from .run import build_views  # run.py imports this module lazily

    rng = random.Random(seed)
    wcfg, ccfg, acfg = cfg["world"], cfg["clients"], cfg.get("attack", {"kind": "none"})
    rcfg, scfg = cfg.get("rounds", {}), cfg.get("server", {})
    n_rounds = rcfg["n"]
    upload_mode = rcfg.get("upload", "all")
    feedback = rcfg.get("feedback", 0.0)
    attack_rounds = acfg.get("rounds", "all")
    if upload_mode not in ("all", "own") or attack_rounds not in ("all", "first"):
        raise ValueError("bad rounds.upload or attack.rounds")
    if acfg["kind"] not in ("none", "a1"):
        raise ValueError("multi-round runs support the A1 attack only")
    if wcfg.get("condition"):
        raise ValueError("multi-round runs do not support conditions")

    sites = ccfg["site_types"]
    world = make_world(rng, wcfg["n_codes"], wcfg["n_procedures"], sites,
                       wcfg["overrides_per_site"])
    clients = make_clients(rng, world, sites, ccfg["coverage"], ccfg["records_per_code"])
    malicious = acfg.get("malicious", []) if acfg["kind"] != "none" else []
    benign = [c for c in clients if c.idx not in malicious]
    general = ccfg.get("upload_scope", "site") == "general"

    distilled = distil_all(llm, clients)
    # First-hand rules of each client, by code. They grow only with feedback.
    own: dict[int, dict[str, str]] = {}
    for c in clients:
        own[c.idx] = {}
        for line in distilled[c.idx]:
            item = parse_item(line)
            if item:
                own[c.idx].setdefault(item[0], line)
    absorbed: dict[int, dict[str, str]] = {c.idx: {} for c in clients}
    fb_rng = random.Random(f"feedback-{seed}")

    local = solve_all(llm, benign, world.codes, distilled, None)
    acc_local = metrics.accuracy(world, benign, local)

    poisons: list[dict] = []
    known_pairs: set = set()
    per_round = []
    raw_rounds = []
    for r in range(1, n_rounds + 1):
        notes = {c.idx: [own[c.idx][k] for k in sorted(own[c.idx])]
                 + [absorbed[c.idx][k] for k in sorted(absorbed[c.idx])
                    if k not in own[c.idx]]
                 for c in clients}
        if r == 1:
            notes = {i: list(v) for i, v in distilled.items()}  # exactly round 1 of run.py
        upload_src = notes if upload_mode == "all" else {
            c.idx: [own[c.idx][k] for k in sorted(own[c.idx])] for c in clients}
        if r == 1:
            upload_src = notes
        uploads = copy.deepcopy(upload_src)
        if general:
            uploads = {i: generalise(v) for i, v in uploads.items()}

        attackers_in = bool(malicious) and (attack_rounds == "all" or r == 1)
        if acfg["kind"] == "a1":
            if r == 1:
                poisons = apply_a1(rng, world, clients, uploads, malicious,
                                   acfg["n_targets"], acfg["claim_scope"])
            elif attackers_in:
                poison_uploads(uploads, malicious, poisons, random.Random(f"poison-{seed}-{r}"))
            else:
                for idx in malicious:
                    uploads.pop(idx)

        sv = build_views(cfg, seed, llm, clients, benign, uploads, None,
                         order_seed=f"order-{seed}" if r == 1 else f"order-{seed}-r{r}",
                         known_pairs=known_pairs if scfg.get("sticky_disputes") else None)
        if scfg.get("sticky_disputes"):
            known_pairs |= sv["pairs"]
        views = sv["views"]
        fed = solve_all(llm, benign, world.codes, notes, views)

        bad = {(p["code"], p["bad"]) for p in poisons}
        carriers = [c.idx for c in benign
                    if any((i[0], i[2]) in bad
                           for i in map(parse_item, uploads.get(c.idx, [])) if i)]
        row = {
            "round": r,
            "acc_fed": metrics.accuracy(world, benign, fed),
            "attackers_in": attackers_in,
            "disputes": sorted(sorted(p) for p in sv["pairs"]),
            "dropped_reports_per_recipient": sum(sv["dropped"].values()) / len(benign),
            "honest_carriers": carriers,
            "honest_false_reports": sum(
                (i[0], i[2]) in bad for c in benign
                for i in map(parse_item, uploads.get(c.idx, [])) if i),
            **metrics.minority_retention(world, benign, views),
        }
        if poisons:
            atk = metrics.attack_metrics(world, benign, poisons, views, fed)
            row["attack"] = {k: v for k, v in atk.items() if k != "cases"}
            row["attack"]["cases"] = atk["cases"]
        per_round.append(row)
        raw_rounds.append({
            "round": r,
            "notes": {str(k): v for k, v in notes.items()},
            "uploads": {str(k): v for k, v in uploads.items()},
            "views": [{"recipients": [c.idx for c in benign if sv["key_of"][c.idx] == key],
                       "library": sv["final"][key]} for key in sv["final"]],
            "answers_fed": {str(k): v for k, v in fed.items()},
        })

        # Absorption: the answer given with the library becomes a note.
        for c in benign:
            for code in world.codes:
                if code in own[c.idx]:
                    continue
                ans = fed[c.idx][code]
                if ans is None:
                    continue
                if feedback and fb_rng.random() < feedback:
                    # An incident of this code reveals the true procedure.
                    own[c.idx][code] = format_item(code, c.site, world.truth(code, c.site))
                    absorbed[c.idx].pop(code, None)
                else:
                    absorbed[c.idx][code] = format_item(code, c.site, ans)
        # Malicious clients keep their own notes: they do not need the library.

    last = per_round[-1]
    result = {
        "seed": seed,
        "config": cfg,
        "n_benign": len(benign),
        "acc_local": acc_local,
        "acc_fed": last["acc_fed"],
        "collaboration_gain": last["acc_fed"] - acc_local,
        "knowable_share": metrics.knowable_share(world, clients, benign),
        "minority_retention": last["minority_retention"],
        "rounds": per_round,
    }
    if poisons:
        result["attack"] = last["attack"]
    result["raw"] = {
        "global_rule": world.global_rule,
        "overrides": world.overrides,
        "condition": None,
        "conditional": {},
        "clients": [{"idx": c.idx, "site": c.site, "train_codes": c.train_codes,
                     "malicious": c.idx in malicious} for c in clients],
        "poisons": poisons,
        "notes": {str(k): v for k, v in distilled.items()},
        "uploads": raw_rounds[0]["uploads"],
        "library": raw_rounds[-1]["views"][0]["library"],
        "answers_local": {str(k): v for k, v in local.items()},
        "answers_fed": raw_rounds[-1]["answers_fed"],
        "rounds": raw_rounds,
    }
    return result

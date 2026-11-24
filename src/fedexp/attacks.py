"""Attacks on the uploads of malicious clients.

A1: direct poisoning, a wrong procedure for a few codes.
A2: trigger-conditioned poisoning, a wrong procedure under a condition only.
A3: locally correct rules of the attacker's site type, claimed for all sites.
"""

from __future__ import annotations

import random

from .prompts import ALL, format_item, lines_for_code
from .world import Client, World


def apply_a1(rng: random.Random, world: World, clients: list[Client],
             uploads: dict[int, list[str]], malicious: list[int],
             n_targets: int, claim_scope: str,
             targeting: str = "random") -> list[dict]:
    """Rewrite the malicious uploads in place and return the injected items.

    The attacker keeps its honest rules for the other codes, so the upload
    looks ordinary. claim_scope is "global" (the rule is claimed for all
    sites) or "site" (claimed for the attacker's own site type).
    All malicious clients push the same poison: this is plain collusion.

    targeting is "random" (codes drawn uniformly) or "uncovered": an oracle
    attacker that knows which codes the honest clients of its own site type
    have seen and picks those that the fewest of them know, so that its false
    rules meet as few contradictions as possible (ties: fewest honest clients
    overall, then at random). This is the worst case for the dispute defence.
    """
    honest = [c for c in clients if c.idx not in malicious]
    attacker_sites = {clients[idx].site for idx in malicious}
    if targeting == "random":
        targets = sorted(rng.sample(world.codes, n_targets))
    elif targeting == "uncovered":
        tie = {c: rng.random() for c in world.codes}
        rank = sorted(world.codes, key=lambda c: (
            sum(c in h.train_codes for h in honest if h.site in attacker_sites),
            sum(c in h.train_codes for h in honest), tie[c]))
        targets = sorted(rank[:n_targets])
    else:
        raise ValueError(f"unknown targeting {targeting}")
    poisons = []
    for code in targets:
        legit = {world.global_rule[code]} | {
            ov[code] for ov in world.overrides.values() if code in ov}
        bad = rng.choice([p for p in world.procedures if p not in legit])
        poisons.append({
            "code": code,
            "bad": bad,
            "honest_support": sum(code in c.train_codes for c in honest),
            "same_site_support": sum(code in c.train_codes for c in honest
                                     if c.site in attacker_sites),
            "attacker_sites": sorted(attacker_sites),
        })

    for idx in malicious:
        site = clients[idx].site
        scope = ALL if claim_scope == "global" else site
        kept = [ln for ln in uploads[idx]
                if not any(lines_for_code([ln], p["code"]) for p in poisons)]
        injected = [format_item(p["code"], scope, p["bad"]) for p in poisons]
        lines = kept + injected
        rng.shuffle(lines)
        uploads[idx] = lines
        for p in poisons:
            p.setdefault("scopes", []).append(scope)
    return poisons


def apply_a2(rng: random.Random, world: World, clients: list[Client],
             uploads: dict[int, list[str]], malicious: list[int],
             n_targets: int) -> list[dict]:
    """A2, trigger-conditioned poisoning.

    The trigger is the world's own condition, which honest clients also use
    for legitimate conditional rules. For each target code, which has no
    legitimate conditional rule, every malicious client adds the rule
    "under the condition, apply <bad>" for all sites, and keeps its honest
    plain rule for the code. Nothing the attacker uploads contradicts a plain
    rule, and without the trigger the poisoned code keeps its right answer.
    """
    if not world.condition:
        raise ValueError("A2 needs a world with a condition")
    honest = [c for c in clients if c.idx not in malicious]
    attacker_sites = {clients[idx].site for idx in malicious}
    candidates = [c for c in world.codes if c not in world.conditional]
    targets = sorted(rng.sample(candidates, n_targets))
    poisons = []
    for code in targets:
        legit = {world.global_rule[code]} | {
            ov[code] for ov in world.overrides.values() if code in ov}
        bad = rng.choice([p for p in world.procedures if p not in legit])
        poisons.append({
            "code": code, "bad": bad, "condition": world.condition,
            "honest_support": sum(code in c.train_codes for c in honest),
            "same_site_support": sum(code in c.train_codes for c in honest
                                     if c.site in attacker_sites),
            "attacker_sites": sorted(attacker_sites), "scopes": [],
        })
    for idx in malicious:
        lines = list(uploads[idx]) + [format_item(p["code"], ALL, p["bad"], world.condition)
                                      for p in poisons]
        rng.shuffle(lines)
        uploads[idx] = lines
        for p in poisons:
            p["scopes"].append(ALL)
    return poisons


def apply_a3(rng: random.Random, world: World, clients: list[Client],
             uploads: dict[int, list[str]], malicious: list[int]) -> list[dict]:
    """A3, locally correct but non-transferable rules.

    Every override of the attackers' site type is true at that site type and
    false elsewhere. Each malicious client uploads all of them, worded for all
    sites, in place of whatever it would have said about those codes. Every
    rule it uploads is correct for its own site, so no client of its site
    type can contradict it. The "bad" procedure of a target is the override:
    the attack succeeds on a client of another site type that answers it.
    """
    sites = {clients[idx].site for idx in malicious}
    if len(sites) != 1:
        raise ValueError("A3 needs all malicious clients at one site type")
    site = sites.pop()
    honest = [c for c in clients if c.idx not in malicious]
    poisons = []
    for code, proc in sorted(world.overrides.get(site, {}).items()):
        poisons.append({
            "code": code, "bad": proc,
            "honest_support": sum(code in c.train_codes for c in honest),
            "same_site_support": sum(code in c.train_codes for c in honest if c.site == site),
            "attacker_sites": [site], "scopes": [],
            # Site types where the override is also the right answer.
            "right_at": sorted(s for s in {c.site for c in clients}
                               if world.truth(code, s) == proc),
        })
    for idx in malicious:
        kept = [ln for ln in uploads[idx]
                if not any(lines_for_code([ln], p["code"]) for p in poisons)]
        lines = kept + [format_item(p["code"], ALL, p["bad"]) for p in poisons]
        rng.shuffle(lines)
        uploads[idx] = lines
        for p in poisons:
            p["scopes"].append(ALL)
    return poisons

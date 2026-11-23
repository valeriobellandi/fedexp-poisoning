"""Synthetic world with hidden, checkable operating rules.

Every alarm code maps to one procedure (the global rule). Each site type
overrides the procedure for a few codes: this is the legitimate minority
knowledge. Codes and procedures are arbitrary identifiers, so the model has
no prior to lean on and every answer is scored by exact match.

Optionally the world has one condition (for example "during a storm") under
which a few codes take a different procedure at every site type. This is
legitimate conditional knowledge: clients that saw such a code also saw it
under the condition. Every other code is unaffected by the condition.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class World:
    codes: list[str]
    procedures: list[str]
    global_rule: dict[str, str]
    overrides: dict[str, dict[str, str]]  # site type -> {code: procedure}
    condition: str | None = None
    conditional: dict[str, str] = field(default_factory=dict)  # code -> procedure

    def truth(self, code: str, site: str, cond: bool = False) -> str:
        if cond and code in self.conditional:
            return self.conditional[code]
        return self.overrides.get(site, {}).get(code, self.global_rule[code])


@dataclass
class Client:
    idx: int
    site: str
    train_codes: list[str]
    # (code, procedure), or (code, procedure, condition) for an incident
    # that happened under the world's condition.
    records: list[tuple] = field(default_factory=list)


def make_world(rng: random.Random, n_codes: int, n_procedures: int,
               site_types: list[str], overrides_per_site: int,
               condition: str | None = None, n_conditional: int = 0) -> World:
    codes = [f"K{10 + i}" for i in range(n_codes)]
    procedures = [f"P{i + 1}" for i in range(n_procedures)]
    global_rule = {c: rng.choice(procedures) for c in codes}
    overrides: dict[str, dict[str, str]] = {}
    for site in sorted(set(site_types)):
        chosen = rng.sample(codes, overrides_per_site)
        overrides[site] = {
            c: rng.choice([p for p in procedures if p != global_rule[c]])
            for c in chosen
        }
    conditional: dict[str, str] = {}
    if condition and n_conditional:
        # Drawn after everything else, so worlds without a condition are
        # the same as before for the same seed.
        for c in sorted(rng.sample(codes, n_conditional)):
            plain = {global_rule[c]} | {ov[c] for ov in overrides.values() if c in ov}
            conditional[c] = rng.choice([p for p in procedures if p not in plain])
    return World(codes, procedures, global_rule, overrides,
                 condition if conditional else None, conditional)


def make_clients(rng: random.Random, world: World, site_types: list[str],
                 coverage: float, records_per_code: int) -> list[Client]:
    """Each client sees only a fraction of the codes, at its own site."""
    n_seen = max(1, round(coverage * len(world.codes)))
    clients = []
    for idx, site in enumerate(site_types):
        seen = sorted(rng.sample(world.codes, n_seen))
        records = []
        for c in seen:
            for k in range(records_per_code):
                if k == records_per_code - 1 and c in world.conditional:
                    # The last incident of a conditional code happened under
                    # the condition; the others did not.
                    records.append((c, world.truth(c, site, True), world.condition))
                else:
                    records.append((c, world.truth(c, site)))
        rng.shuffle(records)
        clients.append(Client(idx, site, seen, records))
    return clients

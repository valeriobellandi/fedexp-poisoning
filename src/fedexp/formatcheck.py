"""Quick check that a model follows the output formats, before a full run.

    PYTHONPATH=src python -m fedexp.formatcheck configs/llama_shuf6_benign.yaml

Runs the distillation of seed 0 and one solve task per seen code for every
client, and reports per client: lines returned, lines in the rule format,
rules correct for the client's site, and solve answers that parse and are
right. A model fit for the full run parses nearly every line and answer.
"""

from __future__ import annotations

import random
import sys

import yaml

from .llm import make_llm
from .prompts import parse_cond_item, parse_item
from .protocol import distil_all, solve_all
from .world import make_clients, make_world


def main() -> None:
    cfg = yaml.safe_load(open(sys.argv[1]))
    llm = make_llm(cfg["llm"])
    rng = random.Random(0)
    w, c = cfg["world"], cfg["clients"]
    world = make_world(rng, w["n_codes"], w["n_procedures"], c["site_types"],
                       w["overrides_per_site"], w.get("condition"), w.get("n_conditional", 0))
    clients = make_clients(rng, world, c["site_types"], c["coverage"], c["records_per_code"])
    notes = distil_all(llm, clients, world.condition)
    print("client site      lines parsed correct | solve parsed correct")
    tot = [0] * 5
    cond_ok = [0, 0, 0]
    answers = solve_all(llm, clients, list(world.codes), notes, None, world.condition)
    for cl in clients:
        lines = notes[cl.idx]
        items = [it for it in map(parse_item, lines) if it]
        conds = [it for it in map(parse_cond_item, lines) if it]
        good = (sum(world.truth(code, cl.site) == proc for code, _, proc in items)
                + sum(world.truth(code, cl.site, True) == proc for code, _, _, proc in conds))
        want_cond = sorted(set(cl.train_codes) & set(world.conditional))
        got_cond = sorted({code for code, _, _, proc in conds
                           if world.truth(code, cl.site, True) == proc})
        cond_ok[0] += len(set(want_cond) & set(got_cond))
        cond_ok[1] += len(want_cond)
        cond_ok[2] += len(set(got_cond) - set(want_cond))
        items = items + conds
        seen = sorted(set(cl.train_codes))
        ans = [answers[cl.idx][k] for k in seen]
        parsed = sum(a is not None for a in ans)
        right = sum(a == world.truth(k, cl.site) for a, k in zip(ans, seen))
        row = [len(lines), len(items), good, parsed, right]
        tot = [a + b for a, b in zip(tot, row)]
        print(f"{cl.idx:6d} {cl.site:9s} {row[0]:5d} {row[1]:6d} {row[2]:7d} | "
              f"{row[3]:3d}/{len(seen):<3d} {row[4]:4d}")
    print(f"total            {tot[0]:5d} {tot[1]:6d} {tot[2]:7d} | {tot[3]:7d} {tot[4]:4d}")
    if world.condition:
        print(f"conditional rules: {cond_ok[0]}/{cond_ok[1]} expected found, "
              f"{cond_ok[2]} for codes without a conditional rule")
    print("\nfirst client, raw notes:")
    print("\n".join(notes[clients[0].idx][:8]))
    if world.condition and cond_ok[1] and cond_ok[0] < 0.8 * cond_ok[1]:
        tot[1] = -1  # conditional rules are not being written: not fit for A2
    ok = tot[1] >= 0.95 * tot[0] and tot[3] >= 0.95 * sum(len(set(x.train_codes)) for x in clients)
    print("\nFORMAT OK" if ok else "\nFORMAT PROBLEM: send this output before the full run")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

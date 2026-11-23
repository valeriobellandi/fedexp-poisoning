"""Predict the outcome of runs that have not been made, from the proxies.

The prediction for a configuration is written before the configuration is
run with the model, so that the run checks the proxies out of sample.

    cd experiments
    PYTHONPATH=src python -m fedexp.predict results predictions/cov2.json

For each configuration it reports, as means over benign recipients:
  acc_fed      predicted accuracy with the library, over all codes:
               (own * a_own + unknown * c0 + c1 * useful + c2 * false) / codes
               where c0, c1, c2 come from the fit of ablation.validate_proxies
               and a_own is the accuracy measured on the codes in the
               client's own notes;
  disputed     share of attacks in which every attacker is in dispute;
  useful_share, false_share  the proxies, per unknown code.
Two sources of uploads are used: simulated worlds, and real notes thinned
to the target coverage.
"""

from __future__ import annotations

import copy
import json
import random
import statistics
import sys
from pathlib import Path

from .ablation import (SITES6, _recipient, error_model, real_sources, sim_source,
                       thin, validate_proxies)
from .attacks import apply_a1
from .prompts import parse_item
from .protocol import disputes, recipient_uploads

TARGETS = {
    # name: (malicious, n_targets, dispute threshold, policy)
    "cov2_benign": ([], 0, 0, None),
    "cov2_a1_f2": ([1, 5], 10, 0, None),
    "cov2_trust_a1_f2": ([1, 5], 10, 2, "exclude"),
    "cov2_trustfb_a1_f2": ([1, 5], 10, 2, "fallback"),
}
COVERAGE = 0.2


def own_accuracy(folder: Path) -> float:
    """Accuracy with the library on the codes present in the client's own
    notes, pooled over the runs with general sharing and K=6."""
    hits = total = 0
    for name in ("shuf6_benign", "shuf6_a1_f2", "trust6_a1_f2", "trustfb6_a1_f2"):
        for path in sorted(folder.glob(f"{name}__hf__seed*.json")):
            raw = json.loads(path.read_text())["raw"]
            mal = {c["idx"] for c in raw["clients"] if c["malicious"]}
            for c in raw["clients"]:
                if c["idx"] in mal:
                    continue
                own = {it[0] for it in map(parse_item, raw["uploads"][str(c["idx"])]) if it}
                for code in own:
                    truth = raw["overrides"].get(c["site"], {}).get(code, raw["global_rule"][code])
                    total += 1
                    hits += raw["answers_fed"][str(c["idx"])][code] == truth
    return hits / total


def predict(sources, malicious, n, theta, policy, draws, coef, a_own, tag,
            thin_to=None) -> dict:
    accs, useful, false, disputed = [], [], [], []
    for s, (world, clients, uploads) in enumerate(sources):
        for d in range(draws):
            rng = random.Random(f"predict/{tag}/{s}/{d}")
            w, cl, up = world, clients, uploads
            if thin_to is not None:
                w, cl, up = thin(rng, world, clients, uploads, thin_to)
            up = copy.deepcopy(up)
            bad: set[tuple[str, str]] = set()
            if malicious and n:
                poisons = apply_a1(rng, w, cl, up, malicious, n, "global")
                bad = {(p["code"], p["bad"]) for p in poisons}
            sites = {c.idx: c.site for c in cl}
            mal = set(malicious)
            pairs = disputes(up, sites, theta) if theta else set()
            if mal:
                disputed.append(mal <= {i for pair in pairs for i in pair})
            n_codes = len(w.codes)
            for c in cl:
                if c.idx in mal:
                    continue
                kept = (recipient_uploads(up, sites, c.idx, theta, policy, pairs)
                        if theta else up)
                f, u, unk = _recipient(w, c.site, up[c.idx], kept, c.idx, mal, bad, n_codes)
                own = n_codes - unk
                accs.append((own * a_own + unk * coef[0] + coef[1] * u + coef[2] * f) / n_codes)
                if unk:
                    useful.append(u / unk)
                    false.append(f / unk)
    return {"acc_fed": statistics.mean(accs),
            "useful_share": statistics.mean(useful),
            "false_share": statistics.mean(false),
            "disputed": statistics.mean(disputed) if disputed else None}


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    target = Path(sys.argv[2] if len(sys.argv) > 2 else "predictions/cov2.json")
    err = error_model(folder)
    val = validate_proxies(folder)
    a_own = own_accuracy(folder)
    real = real_sources(folder)
    sims = [sim_source(i, SITES6, COVERAGE, err["p_miss"], err["lam"]) for i in range(1000)]
    out = {"coverage": COVERAGE, "coef": val["coef"], "a_own": a_own,
           "error_model": err, "configs": {}}
    for name, (malicious, n, theta, policy) in TARGETS.items():
        out["configs"][name] = {
            "sim": predict(sims, malicious, n, theta, policy, 1, val["coef"], a_own, f"sim-{name}"),
            "real_thinned": predict(real, malicious, n, theta, policy, 200, val["coef"], a_own,
                                    f"real-{name}", thin_to=COVERAGE),
        }
        r = out["configs"][name]
        print(f"{name:20s} acc_fed sim {r['sim']['acc_fed']:.3f} real-thinned "
              f"{r['real_thinned']['acc_fed']:.3f} | disputed sim {r['sim']['disputed']} "
              f"real-thinned {r['real_thinned']['disputed']}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2))
    print(f"wrote {target}")


if __name__ == "__main__":
    main()

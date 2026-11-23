"""Run one configuration for one or more seeds and write one JSON per run.

    cd experiments
    PYTHONPATH=src python -m fedexp.run --config configs/pilot_benign.yaml configs/pilot_a1.yaml --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import yaml

from . import metrics
from .attacks import apply_a1, apply_a2, apply_a3
from .llm import make_llm
from .protocol import (COND_SUFFIX, aggregate, corroborate, disputes, distil_all,
                       generalise, recipient_uploads, solve_all)
from .world import make_clients, make_world


def build_views(cfg: dict, seed: int, llm, clients, benign, uploads: dict[int, list[str]],
                condition: str | None, order_seed: str | None = None,
                known_pairs: set | None = None) -> dict:
    """Server side of one round: optional disputes and per-recipient
    exclusions, aggregation, optional support threshold. Returns the library
    of every benign client and the bookkeeping needed for the metrics.

    order_seed: seed of the report order (default: from the run seed).
    known_pairs: disputes remembered from earlier rounds, added to those of
        this round (sticky disputes).
    """
    scfg = cfg.get("server", {})
    sites_by_client = ({c.idx: c.site for c in clients}
                       if scfg.get("show_client_site", False) else None)
    report_order = scfg.get("report_order", "fixed")
    if report_order not in ("fixed", "shuffled"):
        raise ValueError(f"unknown report_order: {report_order}")
    mode = scfg.get("aggregator", "single_prompt")
    if order_seed is None:
        order_seed = f"order-{seed}"
    if report_order != "shuffled":
        order_seed = None
    client_sites = {c.idx: c.site for c in clients}

    # Optional defence before the merge: one library per recipient, built
    # without the reports that recipient has a reason to distrust. Libraries
    # are only built for the benign clients, the ones we evaluate.
    threshold = scfg.get("dispute_threshold", 0)
    policy = scfg.get("dispute_policy", "exclude")
    cond_mode = scfg.get("dispute_conditions", "ignore")
    pairs = set()
    if threshold:
        pairs = disputes(uploads, client_sites, threshold, cond_mode) | set(known_pairs or ())
        # A remembered dispute only matters while both clients take part.
        pairs = {p for p in pairs if p <= set(uploads)}
    kept_for: dict[int, dict[int, list[str]]] = {}
    for c in benign:
        kept_for[c.idx] = (recipient_uploads(uploads, client_sites, c.idx, threshold, policy,
                                             pairs=pairs, conditions=cond_mode)
                           if threshold else uploads)
    key_of = {idx: json.dumps(sorted(kept.items())) for idx, kept in kept_for.items()}
    merged: dict[str, list[str]] = {}
    for c in benign:
        if key_of[c.idx] not in merged:
            merged[key_of[c.idx]] = aggregate(llm, kept_for[c.idx], mode,
                                              sites_by_client, order_seed, condition)

    # Optional defence after the merge: support threshold.
    min_support = scfg.get("min_support", 1)
    final: dict[str, list[str]] = {}
    removed_counts = []
    for c in benign:
        key = key_of[c.idx]
        if key in final:
            continue
        if min_support > 1:
            final[key], removed = corroborate(merged[key], kept_for[c.idx],
                                              client_sites, min_support)
        else:
            final[key], removed = merged[key], []
        removed_counts.append(len(removed))
    views = {c.idx: final[key_of[c.idx]] for c in benign}

    n_up = n_lines(uploads)
    dropped = {c.idx: n_up - n_lines(kept_for[c.idx]) for c in benign}
    # The library shown in the raw output is the one built from most reports.
    main_idx = min(benign, key=lambda c: (dropped[c.idx], c.idx)).idx
    return {"views": views, "kept_for": kept_for, "key_of": key_of, "final": final,
            "merged": merged, "dropped": dropped, "removed_counts": removed_counts,
            "pairs": pairs, "library": views[main_idx],
            "merged_library": merged[key_of[main_idx]]}


def n_lines(reports: dict[int, list[str]]) -> int:
    return sum(len(lines) for lines in reports.values())


def run_once(cfg: dict, seed: int, llm) -> dict:
    if cfg.get("domain") == "heart":
        from .heart import run_heart
        return run_heart(cfg, seed, llm)
    if cfg.get("rounds", {}).get("n", 1) > 1:
        from .rounds import run_rounds
        return run_rounds(cfg, seed, llm)

    rng = random.Random(seed)
    wcfg, ccfg, acfg = cfg["world"], cfg["clients"], cfg.get("attack", {"kind": "none"})
    sites = ccfg["site_types"]
    world = make_world(rng, wcfg["n_codes"], wcfg["n_procedures"], sites,
                       wcfg["overrides_per_site"], wcfg.get("condition"),
                       wcfg.get("n_conditional", 0))
    condition = world.condition
    clients = make_clients(rng, world, sites, ccfg["coverage"],
                           ccfg["records_per_code"])

    malicious = acfg.get("malicious", []) if acfg["kind"] != "none" else []
    benign = [c for c in clients if c.idx not in malicious]

    # 1. Local distillation. Benign notes are what each client keeps for itself.
    notes = distil_all(llm, clients, condition)
    uploads = copy.deepcopy(notes)
    upload_scope = ccfg.get("upload_scope", "site")
    if upload_scope == "general":
        uploads = {idx: generalise(lines) for idx, lines in uploads.items()}
    elif upload_scope != "site":
        raise ValueError(f"unknown upload_scope: {upload_scope}")

    # 2. Attack on the uploads of the malicious clients.
    poisons = []
    if acfg["kind"] == "a1":
        poisons = apply_a1(rng, world, clients, uploads, malicious,
                           acfg["n_targets"], acfg["claim_scope"])
    elif acfg["kind"] == "a2":
        poisons = apply_a2(rng, world, clients, uploads, malicious, acfg["n_targets"])
    elif acfg["kind"] == "a3":
        poisons = apply_a3(rng, world, clients, uploads, malicious)
    elif acfg["kind"] != "none":
        raise ValueError(f"unknown attack: {acfg['kind']}")

    # 3. Server aggregation, with the optional defences.
    sv = build_views(cfg, seed, llm, clients, benign, uploads, condition)
    views, kept_for, key_of, final = sv["views"], sv["kept_for"], sv["key_of"], sv["final"]
    dropped, removed_counts = sv["dropped"], sv["removed_counts"]
    library, merged_library = sv["library"], sv["merged_library"]

    # 4. Benign clients answer every code, without and with the library.
    local = solve_all(llm, benign, world.codes, notes, None, condition)
    fed = solve_all(llm, benign, world.codes, notes, views, condition)

    target_codes = {p["code"] for p in poisons}
    clean_codes = [c for c in world.codes if c not in target_codes]
    parsed, unparsed = metrics.library_stats(library)
    acc_local = metrics.accuracy(world, benign, local)
    acc_fed = metrics.accuracy(world, benign, fed)

    result = {
        "seed": seed,
        "config": cfg,
        "n_benign": len(benign),
        "acc_local": acc_local,
        "acc_fed": acc_fed,
        "collaboration_gain": acc_fed - acc_local,
        "acc_fed_clean_codes": metrics.accuracy(world, benign, fed, clean_codes),
        "knowable_share": metrics.knowable_share(world, clients, benign),
        "library_rules": len(parsed),
        "library_unparsed_lines": unparsed,
        "defence_removed_rules": sum(removed_counts) / len(removed_counts),
        "n_libraries": len(final),
        "dropped_reports_per_recipient": sum(dropped.values()) / len(benign),
        "total_reports": n_lines(uploads),
        **metrics.minority_retention(world, benign, views),
    }
    if condition:
        result.update({f"{k}_local": v for k, v in metrics.conditional_metrics(
            world, benign, local, COND_SUFFIX).items()})
        result.update({f"{k}_fed": v for k, v in metrics.conditional_metrics(
            world, benign, fed, COND_SUFFIX).items()})
    if poisons:
        result["attack"] = metrics.attack_metrics(world, benign, poisons, views, fed)
    if acfg["kind"] == "a2":
        result["a2"] = metrics.a2_metrics(world, benign, poisons, views, fed, COND_SUFFIX)

    # Raw material, kept so that every number can be re-derived by hand.
    result["raw"] = {
        "global_rule": world.global_rule,
        "overrides": world.overrides,
        "condition": world.condition,
        "conditional": world.conditional,
        "clients": [{"idx": c.idx, "site": c.site, "train_codes": c.train_codes,
                     "malicious": c.idx in malicious} for c in clients],
        "poisons": poisons,
        "notes": {str(k): v for k, v in notes.items()},
        "uploads": {str(k): v for k, v in uploads.items()},
        "library": library,
        "library_before_defence": merged_library,
        "views": [{"recipients": [c.idx for c in benign if key_of[c.idx] == key],
                   "excluded": sorted(i for i in uploads if i not in kept_for[first]),
                   "partial": sorted(i for i in kept_for[first]
                                     if len(kept_for[first][i]) < len(uploads[i])),
                   "library": final[key]}
                  for key in final
                  for first in [next(c.idx for c in benign if key_of[c.idx] == key)]],
        "answers_local": {str(k): v for k, v in local.items()},
        "answers_fed": {str(k): v for k, v in fed.items()},
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, nargs="+",
                    help="one or more YAML files; the model is loaded once "
                         "for all configs that share the same llm section")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="results")
    ap.add_argument("--backend", choices=["mock", "hf"],
                    help="override llm.backend from the config")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, object] = {}

    for config in args.config:
        cfg = yaml.safe_load(Path(config).read_text())
        if args.backend:
            cfg["llm"]["backend"] = args.backend
        key = json.dumps(cfg["llm"], sort_keys=True)
        if key not in loaded:
            loaded[key] = make_llm(cfg["llm"])
        name, backend = Path(config).stem, cfg["llm"].get("backend", "mock")

        for seed in args.seeds:
            res = run_once(cfg, seed, loaded[key])
            path = out_dir / f"{name}__{backend}__seed{seed}.json"
            path.write_text(json.dumps(res, indent=2))
            atk = res.get("attack", {})
            print(f"{name} seed {seed}: local {res['acc_local']:.3f}  "
                  f"fed {res['acc_fed']:.3f}  knowable {res.get('knowable_share', float('nan')):.3f}  "
                  f"survival {atk.get('survival')}  asr {atk.get('asr')}  -> {path}",
                  flush=True)


if __name__ == "__main__":
    main()

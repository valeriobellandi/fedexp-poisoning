"""Mean and standard deviation over seeds for every configuration in a folder.

    cd experiments
    PYTHONPATH=src python -m fedexp.summarize results
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

TOP = ["acc_local", "acc_fed", "collaboration_gain", "knowable_share",
       "acc_fed_clean_codes", "minority_retention", "library_unparsed_lines",
       "library_rules", "defence_removed_rules", "n_libraries",
       "dropped_reports_per_recipient", "total_reports"]
ATTACK = ["survival", "survival_contested", "survival_uncontested",
          "survival_same_site", "survival_other_site", "asr",
          "asr_code_in_own_notes", "asr_code_not_in_own_notes", "asr_all_benign",
          "asr_same_site", "asr_other_site"]


def _baseline_key(res: dict) -> str:
    """Everything except the attack, with defaults filled in: an attacked run
    is compared with the benign run that shares this key.

    The dispute defence is left out of the key on purpose: its runs are
    compared with the benign run without it, so the difference includes
    whatever the defence costs.
    """
    cfg = res["config"]
    server = cfg.get("server", {})
    clients = {"upload_scope": "site", **cfg["clients"]}
    return json.dumps({
        "seed": res["seed"], "world": cfg["world"], "clients": clients,
        "aggregator": server.get("aggregator", "single_prompt"),
        "show_client_site": server.get("show_client_site", False),
        "report_order": server.get("report_order", "fixed"),
        "min_support": server.get("min_support", 1),
        "model": cfg["llm"].get("model_id"),
        "load_in_4bit": cfg["llm"].get("load_in_4bit", False),
    }, sort_keys=True)


def _accuracy(res: dict, clients: list[int], codes: set[str] | None = None) -> float:
    """Accuracy with the library for the given clients, from the raw answers."""
    raw = res["raw"]
    sites = {c["idx"]: c["site"] for c in raw["clients"]}
    hits = total = 0
    for idx in clients:
        for code, answer in raw["answers_fed"][str(idx)].items():
            if codes is not None and code not in codes:
                continue
            truth = raw["overrides"].get(sites[idx], {}).get(code, raw["global_rule"][code])
            total += 1
            hits += answer == truth
    return hits / total


def paired(attacked: dict, benign: dict) -> dict:
    """Same seed, same aggregator, same honest clients: with and without attack."""
    honest = [c["idx"] for c in attacked["raw"]["clients"] if not c["malicious"]]
    targets = {p["code"] for p in attacked["raw"]["poisons"]}
    return {
        "paired_acc_benign": _accuracy(benign, honest),
        "paired_acc_attacked": _accuracy(attacked, honest),
        "paired_acc_diff": _accuracy(attacked, honest) - _accuracy(benign, honest),
        "paired_target_acc_benign": _accuracy(benign, honest, targets),
        "paired_target_acc_attacked": _accuracy(attacked, honest, targets),
    }


def _line(key: str, values: list) -> None:
    values = [v for v in values if v is not None]
    if not values:
        return
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"  {key:28s} {mean:.3f} ± {std:.3f}  (n={len(values)})")


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(folder.glob("*.json")):
        groups[path.name.rsplit("__seed", 1)[0]].append(json.loads(path.read_text()))
    if not groups:
        print(f"no result files in {folder}")
        return
    benign = {_baseline_key(r): r
              for runs in groups.values() for r in runs if "attack" not in r}
    for name, runs in groups.items():
        print(f"\n{name}  ({len(runs)} seeds)")
        for key in TOP:
            _line(key, [r.get(key) for r in runs])
        for key in ATTACK:
            _line(key, [r.get("attack", {}).get(key) for r in runs])
        pairs = [paired(r, benign[_baseline_key(r)]) for r in runs
                 if "attack" in r and _baseline_key(r) in benign]
        for key in (pairs[0] if pairs else {}):
            _line(key, [p[key] for p in pairs])


if __name__ == "__main__":
    main()

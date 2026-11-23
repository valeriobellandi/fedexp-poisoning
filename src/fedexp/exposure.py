"""How many poisoned reports reach the benign clients, without running a model.

Disputes and per-recipient libraries depend only on the uploads. This script
takes the notes that the clients distilled in an existing benign run, replays
the attack with many random choices of target codes, and counts what each
defence policy would let through. It measures exposure (false reports that
enter the merge for a recipient), not adoption.

    cd experiments
    PYTHONPATH=src python -m fedexp.exposure results --base shuf6_benign \\
        --malicious 1 5 --targets 1 2 3 5 10 20 40 --draws 200
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

from .attacks import apply_a1
from .prompts import parse_item
from .protocol import disputes, generalise, recipient_uploads
from .world import make_clients, make_world


def rebuild(res: dict):
    """World, clients and honest uploads of a stored benign run."""
    cfg = res["config"]
    rng = random.Random(res["seed"])
    sites = cfg["clients"]["site_types"]
    world = make_world(rng, cfg["world"]["n_codes"], cfg["world"]["n_procedures"],
                       sites, cfg["world"]["overrides_per_site"])
    clients = make_clients(rng, world, sites, cfg["clients"]["coverage"],
                           cfg["clients"]["records_per_code"])
    if world.global_rule != res["raw"]["global_rule"]:
        raise ValueError("stored run does not match the world rebuilt from its seed")
    uploads = {int(k): list(v) for k, v in res["raw"]["notes"].items()}
    if cfg["clients"].get("upload_scope", "site") == "general":
        uploads = {i: generalise(v) for i, v in uploads.items()}
    return world, clients, uploads


def exposure(res: dict, malicious: list[int], n_targets: int, draws: int,
             threshold: int, claim_scope: str = "global") -> dict:
    world, clients, honest_uploads = rebuild(res)
    sites = {c.idx: c.site for c in clients}
    benign = [c.idx for c in clients if c.idx not in malicious]
    mal_sites = {sites[m] for m in malicious}
    peers = [i for i in benign if sites[i] in mal_sites]
    others = [i for i in benign if i not in peers]

    tot = {"draws": 0, "all_attackers_in_dispute": 0, "peer_recipients": 0,
           "peer_recipients_protected": 0}
    reach = {p: {"same": 0, "other": 0} for p in ("exclude", "fallback")}
    for d in range(draws):
        rng = random.Random(f"exposure/{res['seed']}/{n_targets}/{d}")
        uploads = copy.deepcopy(honest_uploads)
        poisons = apply_a1(rng, world, clients, uploads, malicious, n_targets, claim_scope)
        bad = {(p["code"], p["bad"]) for p in poisons}
        pairs = disputes(uploads, sites, threshold)
        in_dispute = {i for pair in pairs for i in pair}
        tot["draws"] += 1
        tot["all_attackers_in_dispute"] += all(m in in_dispute for m in malicious)
        for i in peers:
            tot["peer_recipients"] += 1
            tot["peer_recipients_protected"] += all(
                frozenset((i, m)) in pairs for m in malicious)
        for policy in reach:
            for i in benign:
                kept = recipient_uploads(uploads, sites, i, threshold, policy)
                # Only reports sent by the attackers count: an honest client
                # can report the same wrong procedure by mistake.
                got = {(it[0], it[2]) for m in malicious
                       for it in map(parse_item, kept.get(m, [])) if it}
                reach[policy]["same" if i in peers else "other"] += len(bad & got)
    n = tot["draws"]
    out = {
        "n_targets": n_targets,
        "attackers_all_in_dispute": tot["all_attackers_in_dispute"] / n,
        "same_site_recipient_protected": (
            tot["peer_recipients_protected"] / tot["peer_recipients"]
            if tot["peer_recipients"] else None),
    }
    for policy in reach:
        # Mean number of false reports reaching one recipient (no defence: n_targets).
        out[f"{policy}_same_site"] = reach[policy]["same"] / (n * len(peers)) if peers else None
        out[f"{policy}_other_site"] = reach[policy]["other"] / (n * len(others)) if others else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--base", required=True, help="benign configuration to replay")
    ap.add_argument("--malicious", type=int, nargs="+", required=True)
    ap.add_argument("--targets", type=int, nargs="+", default=[1, 2, 3, 5, 10, 20, 40])
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--threshold", type=int, default=2)
    args = ap.parse_args()

    runs = [json.loads(p.read_text())
            for p in sorted(Path(args.folder).glob(f"{args.base}__*__seed*.json"))]
    if not runs:
        raise SystemExit(f"no runs of {args.base} in {args.folder}")
    print(f"{args.base}: {len(runs)} seeds x {args.draws} draws, malicious {args.malicious}, "
          f"dispute threshold {args.threshold}")
    print("false reports reaching one benign recipient (no defence: all of them)")
    print(f"{'targets':>7} {'attackers':>10} {'same-site':>10} | {'exclude':>17} | {'fallback':>17}")
    print(f"{'':>7} {'disputed':>10} {'protected':>10} | {'same':>8} {'other':>8} | {'same':>8} {'other':>8}")
    for n in args.targets:
        rows = [exposure(r, args.malicious, n, args.draws, args.threshold) for r in runs]

        def mean(key):
            vals = [row[key] for row in rows if row[key] is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        print(f"{n:>7} {mean('attackers_all_in_dispute'):>10.2f} "
              f"{mean('same_site_recipient_protected'):>10.2f} | "
              f"{mean('exclude_same_site'):>8.2f} {mean('exclude_other_site'):>8.2f} | "
              f"{mean('fallback_same_site'):>8.2f} {mean('fallback_other_site'):>8.2f}")


if __name__ == "__main__":
    main()

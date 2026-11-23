"""Look inside the libraries: what the aggregator did with the uploads.

    cd experiments
    PYTHONPATH=src python -m fedexp.diagnose results
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .prompts import ALL, parse_item


def _items(lines: list[str]) -> list[tuple[str, str, str]]:
    return [item for item in map(parse_item, lines) if item]


def _shape(entries: list[tuple[str, str]]) -> str:
    if not entries:
        return "missing"
    scopes = {s for s, _ in entries}
    if scopes == {ALL}:
        return "all only"
    return "site only" if ALL not in scopes else "all + site"


def _kind(scopes) -> str:
    return "+".join(sorted({"all" if s == ALL else "site" for s in scopes}))


def diagnose(runs: list[dict]) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    for res in runs:
        raw = res["raw"]
        lib = _items(raw["library"])
        malicious = {c["idx"] for c in raw["clients"] if c["malicious"]}
        site_of_any = {c["idx"]: c["site"] for c in raw["clients"]}
        uploads = {int(k): _items(v) for k, v in raw["uploads"].items()}
        reported = {c for items in uploads.values() for c, _, _ in items}
        out["totals"]["codes reported but absent from the library"] += len(
            reported - {c for c, _, _ in lib})
        out["totals"]["library lines out of format"] += len(raw["library"]) - len(lib)

        def truth(code: str, site: str) -> str:
            return raw["overrides"].get(site, {}).get(code, raw["global_rule"][code])

        for c in raw["clients"]:
            if c["idx"] in malicious:
                continue
            got = {code: (scope, proc) for code, scope, proc in uploads[c["idx"]]}
            for code in c["train_codes"]:
                scope, proc = got.get(code, (None, None))
                ok = scope in (c["site"], ALL) and proc == truth(code, c["site"])
                out["honest distillation (client, code)"][
                    "correct" if ok else "wrong or missing"] += 1

        for view in raw.get("views", []):
            honest_out = [i for i in view["excluded"] if i not in malicious]
            for idx in view["recipients"]:
                same = site_of_any[idx] in {site_of_any[m] for m in malicious}
                out["recipients: (same site as an attacker, attackers left out of "
                    "its library, honest clients left out, clients kept in part)"][
                    (same if malicious else None,
                     f"{len(malicious & set(view['excluded']))}/{len(malicious)}",
                     len(honest_out), len(view.get("partial", [])))] += 1

        if "attack" not in res:
            for code in sorted(reported):
                site_of = {c["idx"]: c["site"] for c in raw["clients"]}
                reps = {(site_of[i], p) for i, items in uploads.items()
                        for c, _, p in items if c == code}
                sites = "1 site type" if len({s for s, _ in reps}) == 1 else ">=2 site types"
                agree = "agree" if len({p for _, p in reps}) == 1 else "differ"
                shape = _shape([(s, p) for c, s, p in lib if c == code])
                out["how reports were merged (who reported, agreement, shape in library)"][
                    (sites, agree, shape)] += 1
                procs = [p for items in uploads.values() for c, _, p in items if c == code]
                if len(set(procs)) > 1:
                    true_global = raw["global_rule"][code]
                    n_true = procs.count(true_global)
                    n_other = len(procs) - n_true
                    balance = ("majority" if n_true > n_other else
                               "tie" if n_true == n_other else "minority")
                    general = {p for c, s, p in lib if c == code and s == ALL}
                    out["honest reports differ: (reports of the true general rule, "
                        "what the library's general rule is)"][
                        (balance, "true general rule" if general == {true_global}
                         else "another procedure" if general else "none")] += 1
            continue

        # The honest client of the attacker's own site type, if any.
        mal_sites = {c["site"] for c in raw["clients"] if c["idx"] in malicious}
        site_of_client = {c["idx"]: c["site"] for c in raw["clients"]}
        peers = [c for c in raw["clients"]
                 if c["site"] in mal_sites and c["idx"] not in malicious]
        before = _items(raw.get("library_before_defence", raw["library"]))
        for po in raw["poisons"]:
            was = any(c == po["code"] and p == po["bad"] for c, _, p in before)
            now = any(c == po["code"] and p == po["bad"] for c, _, p in lib)
            out["poisoned rule (in the merged library, after the defence)"][(was, now)] += 1
        landings: dict[str, str] = {}
        for po in raw["poisons"]:
            entries = [(s, p) for c, s, p in lib if c == po["code"]]
            bad_scopes = {s for s, p in entries if p == po["bad"]}
            others = {s for s, p in entries if p != po["bad"]}
            if not bad_scopes:
                fate = "dropped"
            else:
                fate = (f"kept at scope {_kind(bad_scopes)}; other rules for the "
                        f"code: {_kind(others) if others else 'none'}")
            out["fate of each poisoned rule"][fate] += 1
            out["(honest clients that saw the code, poison survived)"][
                (po["honest_support"], bool(bad_scopes))] += 1
            landing = ("dropped" if not bad_scopes else
                       "general rule" if bad_scopes == {ALL} else
                       "site exception" if ALL not in bad_scopes else "both")
            landings[po["code"]] = landing
            reporters = [i for i, items in sorted(uploads.items())
                         for c, _, _ in items if c == po["code"]]
            honest = [i for i in reporters if i not in malicious]
            same = sum(site_of_client[i] in mal_sites for i in honest)
            out["(honest reports from the attacker's site type, from other "
                "site types, where the poison landed)"][
                (same, len(honest) - same, landing)] += 1
            if honest:
                out["contested codes: (attacker is the first reporter, where the "
                    "poison landed)"][(reporters[0] in malicious, landing)] += 1
            n_peers = sum(po["code"] in c["train_codes"] for c in peers)
            out["(honest same-site clients that saw the code, poison survived)"][
                (n_peers, bool(bad_scopes))] += 1
        for case in res["attack"]["cases"]:
            same = case.get("same_site",
                            site_of_client[case["client"]] in mal_sites)
            out["cases (same site as attacker, survived, code in own notes, adopted)"][
                (same, case["survived"], case["in_own_notes"], case["adopted"])] += 1
            out["cases (same site as attacker, where the poison landed, code in "
                "own notes, adopted)"][
                (same, landings[case["code"]], case["in_own_notes"], case["adopted"])] += 1
    return out


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    groups: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(folder.glob("*.json")):
        groups[path.name.rsplit("__seed", 1)[0]].append(json.loads(path.read_text()))
    if not groups:
        print(f"no result files in {folder}")
        return
    for name, runs in groups.items():
        print(f"\n=== {name}  ({len(runs)} runs)")
        for title, counter in diagnose(runs).items():
            print(f"  {title}")
            for key, n in sorted(counter.items(), key=lambda kv: str(kv[0])):
                print(f"    {n:4d}  {key}")


if __name__ == "__main__":
    main()

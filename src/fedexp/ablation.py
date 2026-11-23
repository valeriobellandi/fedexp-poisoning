"""Model-free ablations of the dispute defence.

Disputes, exclusion sets and the reports that enter the merge for each
recipient depend only on the uploads, not on the model. This module measures
them while varying the dispute threshold, the coverage of the clients and the
make-up of the attacker's site type. Uploads come from two sources:

  real  the notes that Qwen2.5-7B-Instruct distilled in the runs of
        shuf6_benign, optionally thinned to a lower coverage;
  sim   notes generated from the ground truth with the error model measured
        on the real notes: a seen rule is omitted with probability p_miss,
        and each client adds a Poisson(lam) number of invented rules for
        codes it has not seen, with a random procedure.

Two quantities stand in for the model, per benign recipient:
  false   distinct false rules of the attackers that enter its merge, on
          codes absent from its own notes;
  useful  codes absent from its own notes for which some report that enters
          its merge states the procedure that is correct at its site.
How well they track the model is checked on the real runs (validate_proxies).

    cd experiments
    PYTHONPATH=src python -m fedexp.ablation results ../tables
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .attacks import apply_a1
from .exposure import rebuild
from .prompts import ALL, format_item, parse_item
from .protocol import disputes, recipient_uploads
from .world import make_clients, make_world

POLICIES = ("exclude", "fallback")
N_SWEEP = (1, 2, 3, 5, 10, 20, 40)
SITES6 = ["coastal", "coastal", "mountain", "mountain", "urban", "coastal"]


# --------------------------------------------------------------------------
# Sources of uploads

def error_model(folder: Path, base: str = "shuf6_benign") -> dict:
    """Distillation errors of the real runs: omitted rules among the codes a
    client has seen, and invented rules for codes it has not seen."""
    seen = missing = invented = clients = 0
    for path in sorted(folder.glob(f"{base}__*__seed*.json")):
        raw = json.loads(path.read_text())["raw"]
        for c in raw["clients"]:
            codes = {it[0] for it in map(parse_item, raw["notes"][str(c["idx"])]) if it}
            seen += len(c["train_codes"])
            missing += len(set(c["train_codes"]) - codes)
            invented += len(codes - set(c["train_codes"]))
            clients += 1
    return {"seen": seen, "missing": missing, "invented": invented, "clients": clients,
            "p_miss": missing / seen, "lam": invented / clients}


def real_sources(folder: Path, base: str = "shuf6_benign") -> list[tuple]:
    out = []
    for path in sorted(folder.glob(f"{base}__*__seed*.json")):
        res = json.loads(path.read_text())
        out.append(rebuild(res))
    return out


def thin(rng: random.Random, world, clients, uploads, coverage: float):
    """Lower the coverage of every client by forgetting a random part of the
    codes it has seen, together with its notes about them. Invented rules
    (codes the client never saw) are kept."""
    m = round(coverage * len(world.codes))
    new_clients, new_uploads = [], {}
    for c in clients:
        if m > len(c.train_codes):
            raise ValueError("thinning cannot raise the coverage")
        keep = set(rng.sample(c.train_codes, m))
        forget = set(c.train_codes) - keep
        new_uploads[c.idx] = [line for line in uploads[c.idx]
                              if (it := parse_item(line)) is None or it[0] not in forget]
        new_clients.append(dataclasses.replace(
            c, train_codes=sorted(keep), records=[r for r in c.records if r[0] in keep]))
    return world, new_clients, new_uploads


def _poisson(rng: random.Random, lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def sim_source(seed: int, site_types: list[str], coverage: float, p_miss: float,
               lam: float, n_codes: int = 40, n_procedures: int = 8,
               overrides_per_site: int = 6):
    """A world, its clients and their general-scope uploads, with the
    distillation errors drawn from the error model."""
    rng = random.Random(f"sim/{seed}")
    world = make_world(rng, n_codes, n_procedures, site_types, overrides_per_site)
    clients = make_clients(rng, world, site_types, coverage, 2)
    uploads = {}
    for c in clients:
        lines = [format_item(code, ALL, world.truth(code, c.site))
                 for code in c.train_codes if rng.random() >= p_miss]
        unseen = [code for code in world.codes if code not in c.train_codes]
        for code in rng.sample(unseen, min(_poisson(rng, lam), len(unseen))):
            lines.append(format_item(code, ALL, rng.choice(world.procedures)))
        rng.shuffle(lines)
        uploads[c.idx] = lines
    return world, clients, uploads


# --------------------------------------------------------------------------
# One attack on one set of uploads

def _recipient(world, site: str, own_upload: list[str], kept: dict[int, list[str]],
               me: int, malicious: set[int], bad: set[tuple[str, str]],
               n_codes: int = 40) -> tuple[int, int, int]:
    """(false rules, useful codes, codes absent from the recipient's notes)."""
    own = {it[0] for it in map(parse_item, own_upload) if it}
    false, useful = set(), set()
    for j, lines in kept.items():
        if j == me:
            continue
        for line in lines:
            it = parse_item(line)
            if not it or it[0] in own:
                continue
            code, _, proc = it
            if j in malicious and (code, proc) in bad:
                false.add((code, proc))
            if proc == world.truth(code, site):
                useful.add(code)
    return len(false), len(useful), n_codes - len(own)


class Tally:
    """Running sums keyed by (defence, group, quantity)."""

    def __init__(self):
        self.sum = defaultdict(float)
        self.n = defaultdict(int)

    def add(self, key, value):
        self.sum[key] += value
        self.n[key] += 1

    def mean(self, key):
        return self.sum[key] / self.n[key] if self.n[key] else float("nan")


def attack_once(tally: Tally, rng: random.Random, world, clients, honest_uploads,
                malicious: list[int], n_targets: int, thresholds: list[int]) -> None:
    uploads = copy.deepcopy(honest_uploads)
    bad: set[tuple[str, str]] = set()
    if malicious and n_targets:
        poisons = apply_a1(rng, world, clients, uploads, malicious, n_targets, "global")
        bad = {(p["code"], p["bad"]) for p in poisons}
    sites = {c.idx: c.site for c in clients}
    mal = set(malicious)
    mal_sites = {sites[m] for m in mal}
    benign = [c for c in clients if c.idx not in mal]

    def group(c):
        return "same" if c.site in mal_sites else "other"

    n_codes = len(world.codes)
    for c in benign:
        f, u, unk = _recipient(world, c.site, uploads[c.idx], uploads, c.idx, mal, bad, n_codes)
        tally.add(("none", group(c), "false"), f)
        tally.add(("none", group(c), "useful"), u)
        tally.add(("none", group(c), "useful_share"), u / unk if unk else 1.0)
    for theta in thresholds:
        pairs = disputes(uploads, sites, theta)
        tally.add((theta, "run", "any_dispute"), bool(pairs))
        if mal:
            disputants = {i for pair in pairs for i in pair}
            tally.add((theta, "run", "attackers_disputed"), mal <= disputants)
        for c in benign:
            if group(c) == "same" and mal:
                same_site_mal = [m for m in mal if sites[m] == c.site]
                tally.add((theta, "same", "protected"),
                          all(frozenset((c.idx, m)) in pairs for m in same_site_mal))
            for policy in POLICIES:
                kept = recipient_uploads(uploads, sites, c.idx, theta, policy, pairs)
                f, u, _ = _recipient(world, c.site, uploads[c.idx], kept, c.idx, mal, bad,
                                     n_codes)
                tally.add(((theta, policy), group(c), "false"), f)
                tally.add(((theta, policy), group(c), "useful"), u)


def sweep(sources, malicious, ns, thresholds, draws, thin_to=None, tag=""):
    """One Tally per n, over every source and draw."""
    out = {}
    for n in ns:
        tally = Tally()
        for s, (world, clients, uploads) in enumerate(sources):
            for d in range(draws):
                rng = random.Random(f"ablation/{tag}/{s}/{n}/{d}")
                w, cl, up = world, clients, uploads
                if thin_to is not None:
                    w, cl, up = thin(rng, world, clients, uploads, thin_to)
                attack_once(tally, rng, w, cl, up, malicious, n, thresholds)
        out[n] = tally
    return out


def worst(tallies, defence, group, key="false"):
    """Largest mean over the attack sizes, and the size that reaches it."""
    best_n = max((n for n in tallies if n), key=lambda n: tallies[n].mean((defence, group, key)))
    return tallies[best_n].mean((defence, group, key)), best_n


# --------------------------------------------------------------------------
# Do the proxies track the model?

PROXY_CONFIGS = ["shuf6_benign", "shuf6_a1_f1", "shuf6_a1_f2", "shuf6_a1_urban",
                 "trust6_a1_f1", "trust6_a1_f2", "trustfb6_a1_f1", "trustfb6_a1_f2",
                 "shuf6_a1_f2_n3", "trustfb6_a1_f2_n3", "shuf6_a1_f2_flood",
                 "trustfb6_a1_f2_flood"]


def validate_proxies(folder: Path) -> dict:
    """Per real run and benign recipient: proxies computed on the run's own
    uploads and defence, against the recipient's accuracy on the codes absent
    from its own notes."""
    rows = []
    by_config = defaultdict(list)
    for name in PROXY_CONFIGS:
        for path in sorted(folder.glob(f"{name}__hf__seed*.json")):
            res = json.loads(path.read_text())
            raw, server = res["raw"], res["config"]["server"]
            uploads = {int(k): v for k, v in raw["uploads"].items()}
            sites = {c["idx"]: c["site"] for c in raw["clients"]}
            mal = {c["idx"] for c in raw["clients"] if c["malicious"]}
            bad = {(p["code"], p["bad"]) for p in raw.get("poisons", [])}
            theta = server.get("dispute_threshold", 0)
            policy = server.get("dispute_policy", "exclude")

            class W:  # the world, as the stored run gives it
                @staticmethod
                def truth(code, site):
                    return raw["overrides"].get(site, {}).get(code, raw["global_rule"][code])

            for idx, site in sites.items():
                if idx in mal:
                    continue
                kept = (recipient_uploads(uploads, sites, idx, theta, policy)
                        if theta else uploads)
                f, u, _ = _recipient(W, site, uploads[idx], kept, idx, mal, bad,
                                     len(raw["global_rule"]))
                own = {it[0] for it in map(parse_item, uploads[idx]) if it}
                unknown = [c for c in raw["global_rule"] if c not in own]
                answers = raw["answers_fed"][str(idx)]
                acc = sum(answers[c] == W.truth(c, site) for c in unknown) / len(unknown)
                rows.append((u / len(unknown), f / len(unknown), acc))
                by_config[name].append((u / len(unknown), f / len(unknown), acc))
    data = np.array(rows)
    x = np.column_stack([np.ones(len(data)), data[:, 0], data[:, 1]])
    coef, *_ = np.linalg.lstsq(x, data[:, 2], rcond=None)
    pred = x @ coef
    ss_res = float(((data[:, 2] - pred) ** 2).sum())
    ss_tot = float(((data[:, 2] - data[:, 2].mean()) ** 2).sum())
    configs = {}
    for name, items in by_config.items():
        a = np.array(items)
        xa = np.column_stack([np.ones(len(a)), a[:, 0], a[:, 1]])
        configs[name] = {"useful": float(a[:, 0].mean()), "false": float(a[:, 1].mean()),
                         "accuracy": float(a[:, 2].mean()),
                         "predicted": float((xa @ coef).mean())}
    # Leave one configuration out: fit on the others, predict the one left out.
    for name, items in by_config.items():
        rest = np.array([r for other, its in by_config.items() if other != name for r in its])
        xr = np.column_stack([np.ones(len(rest)), rest[:, 0], rest[:, 1]])
        c_out, *_ = np.linalg.lstsq(xr, rest[:, 2], rcond=None)
        a = np.array(items)
        xa = np.column_stack([np.ones(len(a)), a[:, 0], a[:, 1]])
        configs[name]["predicted_out"] = float((xa @ c_out).mean())
    actual = np.array([v["accuracy"] for v in configs.values()])
    predicted = np.array([v["predicted"] for v in configs.values()])
    predicted_out = np.array([v["predicted_out"] for v in configs.values()])
    return {"n": len(data), "coef": [float(c) for c in coef],
            "corr_config_out": float(np.corrcoef(actual, predicted_out)[0, 1]),
            "max_abs_err_config_out": float(np.abs(actual - predicted_out).max()),
            "r2_recipient": 1 - ss_res / ss_tot,
            "corr_config": float(np.corrcoef(actual, predicted)[0, 1]),
            "max_abs_err_config": float(np.abs(actual - predicted).max()),
            "configs": configs}


# --------------------------------------------------------------------------
# Tables

def _f(x, d=2):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(["% Generated by experiments/src/fedexp/ablation.py. "
                               "Do not edit by hand."] + lines + [""]))
    print(f"wrote {path}", flush=True)


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "../tables")
    draws_real = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    worlds_sim = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    err = error_model(folder)
    report["error_model"] = err
    print(f"error model: p_miss {err['p_miss']:.4f} ({err['missing']}/{err['seen']}), "
          f"lam {err['lam']:.3f} ({err['invented']}/{err['clients']})", flush=True)

    real = real_sources(folder)
    sims = {}

    def sim(sites, rho, count=worlds_sim):
        key = (tuple(sites), rho)
        if key not in sims:
            sims[key] = [sim_source(i, sites, rho, err["p_miss"], err["lam"])
                         for i in range(count)]
        return sims[key]

    # --- Proxies against the model ---------------------------------------
    val = validate_proxies(folder)
    report["proxies"] = val
    print(f"proxies: R2 per recipient {val['r2_recipient']:.2f}, corr per config "
          f"{val['corr_config']:.3f}, max error per config {val['max_abs_err_config']:.3f}; "
          f"left out: corr {val['corr_config_out']:.3f}, max error "
          f"{val['max_abs_err_config_out']:.3f}", flush=True)
    lines = ["\\begin{tabular}{lccccc}", "\\toprule",
             "Configuration & Useful & False & Accuracy & Fit & Left out \\\\",
             "\\midrule"]
    for name, v in val["configs"].items():
        lines.append(f"\\texttt{{{name.replace('_', chr(92) + '_')}}} & {_f(v['useful'])} & "
                     f"{_f(v['false'])} & {_f(v['accuracy'], 3)} & {_f(v['predicted'], 3)} & "
                     f"{_f(v['predicted_out'], 3)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out / "ablation_proxies.tex", lines)

    # --- Simulator against the real notes --------------------------------
    rows = []
    for label, src, draws in (("real", real, draws_real),
                              ("sim", sim(SITES6, 0.4), max(1, draws_real // 20))):
        t = sweep(src, [1, 5], [0, 3, 10, 40], [1, 2], draws, tag=f"check-{label}")
        rows.append([label,
                     t[0].mean((1, "run", "any_dispute")), t[0].mean((2, "run", "any_dispute")),
                     t[3].mean((2, "run", "attackers_disputed")),
                     t[10].mean((2, "run", "attackers_disputed")),
                     t[3].mean(((2, "exclude"), "other", "false")),
                     t[40].mean(((2, "fallback"), "other", "false")),
                     t[0].mean(("none", "other", "useful"))])
    report["sim_check"] = rows
    lines = ["\\begin{tabular}{lccccccc}", "\\toprule",
             " & \\multicolumn{2}{c}{Spurious dispute} & \\multicolumn{2}{c}{Attackers in dispute}"
             " & Exclude & Fallback & Useful \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Notes & $\\theta{=}1$ & $\\theta{=}2$ & $n{=}3$ & $n{=}10$ & $n{=}3$ & $n{=}40$"
             " & no attack \\\\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join([r[0]] + [_f(x) for x in r[1:]]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out / "ablation_simcheck.tex", lines)

    # --- Dispute threshold (real notes, two colluding attackers) ----------
    thetas = [1, 2, 3, 4]
    t_real = sweep(real, [1, 5], (0,) + N_SWEEP, thetas, draws_real, tag="theta")
    t_sim = sweep(sim(SITES6, 0.4), [1, 5], (0,), thetas, 1, tag="theta-sim")
    rows = []
    for th in thetas:
        ex, ex_n = worst(t_real, (th, "exclude"), "other")
        fb, fb_n = worst(t_real, (th, "fallback"), "other")
        base_u = t_real[0].mean(("none", "other", "useful"))
        rows.append({"theta": th,
                     "spurious_real": t_real[0].mean((th, "run", "any_dispute")),
                     "spurious_sim": t_sim[0].mean((th, "run", "any_dispute")),
                     "useful_kept_no_attack": t_real[0].mean(((th, "exclude"), "other", "useful")) / base_u,
                     "disputed_n10": t_real[10].mean((th, "run", "attackers_disputed")),
                     "exclude_worst": ex, "exclude_worst_n": ex_n,
                     "fallback_worst": fb, "fallback_worst_n": fb_n,
                     "useful_kept_flood_exclude": t_real[40].mean(((th, "exclude"), "other", "useful")) / base_u})
    report["threshold"] = rows
    lines = ["\\begin{tabular}{ccccccc}", "\\toprule",
             " & \\multicolumn{2}{c}{Spurious dispute} & Useful kept & Attackers in"
             " & \\multicolumn{2}{c}{Worst false rules} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){6-7}",
             "$\\theta$ & real & sim & no attack & dispute, $n{=}10$ & exclude & fallback \\\\",
             "\\midrule"]
    for r in rows:
        lines.append(f"{r['theta']} & {_f(r['spurious_real'])} & {_f(r['spurious_sim'])} & "
                     f"{_f(r['useful_kept_no_attack'])} & {_f(r['disputed_n10'])} & "
                     f"{_f(r['exclude_worst'])} ($n{{=}}{r['exclude_worst_n']}$) & "
                     f"{_f(r['fallback_worst'])} ($n{{=}}{r['fallback_worst_n']}$) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out / "ablation_threshold.tex", lines)

    # --- Coverage (theta = 2, two colluding attackers) --------------------
    rows = []
    for rho in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        src = sim(SITES6, rho)
        t = sweep(src, [1, 5], (0,) + N_SWEEP, [2], 1, tag=f"rho-{rho}")
        base_u = t[0].mean(("none", "other", "useful"))
        ex, ex_n = worst(t, (2, "exclude"), "other")
        fb, fb_n = worst(t, (2, "fallback"), "other")
        row = {"rho": rho, "useful_share": t[0].mean(("none", "other", "useful_share")),
               "spurious": t[0].mean((2, "run", "any_dispute")),
               "disputed_n10": t[10].mean((2, "run", "attackers_disputed")),
               "exclude_worst": ex, "exclude_worst_n": ex_n,
               "fallback_worst": fb, "fallback_worst_n": fb_n,
               "useful_kept_flood_exclude": t[40].mean(((2, "exclude"), "other", "useful")) / base_u,
               "useful_kept_flood_fallback": t[40].mean(((2, "fallback"), "other", "useful")) / base_u}
        if rho <= 0.4:
            tr = sweep(real, [1, 5], (0, 10), [2], max(1, draws_real // 5), thin_to=rho,
                       tag=f"rho-real-{rho}")
            row["disputed_n10_real"] = tr[10].mean((2, "run", "attackers_disputed"))
            row["spurious_real"] = tr[0].mean((2, "run", "any_dispute"))
        rows.append(row)
        print(f"  coverage {rho} done", flush=True)
    report["coverage"] = rows
    lines = ["\\begin{tabular}{cccccccc}", "\\toprule",
             " & Federation & Spurious & \\multicolumn{2}{c}{Attackers in dispute, $n{=}10$}"
             " & \\multicolumn{2}{c}{Worst false rules} & Useful kept, \\\\",
             "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "$\\rho$ & coverage & dispute & sim & real & exclude & fallback"
             " & flooding (excl./fallb.) \\\\", "\\midrule"]
    for r in rows:
        lines.append(f"{r['rho']:.1f} & {_f(r['useful_share'])} & {_f(r['spurious'])} & "
                     f"{_f(r['disputed_n10'])} & {_f(r.get('disputed_n10_real'))} & "
                     f"{_f(r['exclude_worst'])} ($n{{=}}{r['exclude_worst_n']}$) & "
                     f"{_f(r['fallback_worst'])} ($n{{=}}{r['fallback_worst_n']}$) & "
                     f"{_f(r['useful_kept_flood_exclude'])} / {_f(r['useful_kept_flood_fallback'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out / "ablation_coverage.tex", lines)

    # --- Make-up of the attacker's site type (theta = 2, rho = 0.4) -------
    rows = []
    for h in (0, 1, 2, 3):
        for f in (1, 2, 3):
            sites = ["coastal"] * (h + f) + ["mountain", "mountain", "urban"]
            malicious = list(range(h, h + f))
            src = sim(sites, 0.4)
            t = sweep(src, malicious, N_SWEEP, [2], 1, tag=f"peers-{h}-{f}")
            ex_o, ex_on = worst(t, (2, "exclude"), "other")
            fb_o, fb_on = worst(t, (2, "fallback"), "other")
            row = {"honest": h, "malicious": f,
                   "protected_n10": t[10].mean((2, "same", "protected")) if h else None,
                   "exclude_worst_same": worst(t, (2, "exclude"), "same")[0] if h else None,
                   "exclude_worst_other": ex_o, "exclude_worst_other_n": ex_on,
                   "fallback_worst_other": fb_o, "fallback_worst_other_n": fb_on,
                   "none_n10_other": t[10].mean(("none", "other", "false"))}
            rows.append(row)
        print(f"  peers h={h} done", flush=True)
    report["peers"] = rows
    # The table shows two colluders; the text reports how little f changes.
    lines = ["\\begin{tabular}{ccccccc}", "\\toprule",
             "Honest & Malicious & Protected & \\multicolumn{3}{c}{Worst false rules} & No defence \\\\",
             "\\cmidrule(lr){4-6}",
             "same site & same site & $n{=}10$ & exclude, same & exclude, other"
             " & fallback, other & $n{=}10$, other \\\\", "\\midrule"]
    for r in [r for r in rows if r["malicious"] == 2]:
        lines.append(f"{r['honest']} & {r['malicious']} & {_f(r['protected_n10'])} & "
                     f"{_f(r['exclude_worst_same'])} & "
                     f"{_f(r['exclude_worst_other'])} ($n{{=}}{r['exclude_worst_other_n']}$) & "
                     f"{_f(r['fallback_worst_other'])} ($n{{=}}{r['fallback_worst_other_n']}$) & "
                     f"{_f(r['none_n10_other'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out / "ablation_peers.tex", lines)

    target = Path("ablations")
    target.mkdir(exist_ok=True)
    (target / "ablation.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {target / 'ablation.json'}")


if __name__ == "__main__":
    main()

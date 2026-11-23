"""Compare the model runs of the seventh round with the predictions written
before them (predictions/cov2.json).

    PYTHONPATH=src python -m fedexp.checkpred results predictions/cov2.json

Per configuration: measured acc_fed against the two predictions; the share
of runs in which both attackers are in dispute, measured on the run's own
uploads; and, to separate the two sources of error, the fit applied to the
proxies of the actual uploads with the accuracy measured on own codes
("fit on actual uploads"). The gap between that column and the measured
value is the error of the fit; the gap between it and the predictions is
the error of the simulated or thinned uploads.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from .ablation import _recipient
from .prompts import parse_item
from .protocol import disputes, recipient_uploads


def one_run(res: dict, coef: list[float]) -> dict:
    raw, server = res["raw"], res["config"]["server"]
    uploads = {int(k): v for k, v in raw["uploads"].items()}
    sites = {c["idx"]: c["site"] for c in raw["clients"]}
    mal = {c["idx"] for c in raw["clients"] if c["malicious"]}
    bad = {(p["code"], p["bad"]) for p in raw.get("poisons", [])}
    theta = server.get("dispute_threshold", 0)
    policy = server.get("dispute_policy", "exclude")

    class W:
        @staticmethod
        def truth(code, site):
            return raw["overrides"].get(site, {}).get(code, raw["global_rule"][code])

    codes = list(raw["global_rule"])
    pairs = disputes(uploads, sites, 2)
    out = {"acc_fed": res["acc_fed"],
           "disputed": (mal <= {i for p in pairs for i in p}) if mal else None,
           "useful": [], "false": [], "fit": [], "acc_unknown": [], "own_hits": 0, "own_n": 0}
    for idx, site in sites.items():
        if idx in mal:
            continue
        kept = recipient_uploads(uploads, sites, idx, theta, policy) if theta else uploads
        f, u, _ = _recipient(W, site, uploads[idx], kept, idx, mal, bad, len(codes))
        own = {it[0] for it in map(parse_item, uploads[idx]) if it}
        unknown = [c for c in codes if c not in own]
        ans = raw["answers_fed"][str(idx)]
        own_hits = sum(ans[c] == W.truth(c, site) for c in own)
        out["own_hits"] += own_hits
        out["own_n"] += len(own)
        unk_pred = coef[0] * len(unknown) + coef[1] * u + coef[2] * f
        out["fit"].append((own_hits + unk_pred) / len(codes))
        if unknown:
            out["useful"].append(u / len(unknown))
            out["false"].append(f / len(unknown))
            out["acc_unknown"].append(sum(ans[c] == W.truth(c, site) for c in unknown) / len(unknown))
    return out


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    pred = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else "predictions/cov2.json").read_text())
    coef = pred["coef"]
    report = {}
    print(f"{'config':20s} {'measured':>14s} {'pred sim':>9s} {'pred thin':>9s} "
          f"{'fit actual':>10s} | {'disp meas':>9s} {'sim':>5s} {'thin':>5s} | "
          f"{'useful':>6s} {'false':>6s} {'own acc':>7s}")
    for name, p in pred["configs"].items():
        runs = [one_run(json.loads(f.read_text()), coef)
                for f in sorted(folder.glob(f"{name}__hf__seed*.json"))]
        if not runs:
            print(f"{name}: no runs")
            continue
        acc = [r["acc_fed"] for r in runs]
        fit = statistics.mean(x for r in runs for x in r["fit"])
        disp = [r["disputed"] for r in runs if r["disputed"] is not None]
        row = {
            "n_runs": len(runs),
            "acc_fed": statistics.mean(acc), "acc_fed_sd": statistics.stdev(acc),
            "pred_sim": p["sim"]["acc_fed"], "pred_thin": p["real_thinned"]["acc_fed"],
            "fit_actual": fit,
            "disputed": statistics.mean(disp) if disp else None,
            "pred_disputed_sim": p["sim"]["disputed"], "pred_disputed_thin": p["real_thinned"]["disputed"],
            "useful": statistics.mean(x for r in runs for x in r["useful"]),
            "false": statistics.mean(x for r in runs for x in r["false"]),
            "pred_useful_sim": p["sim"]["useful_share"], "pred_false_sim": p["sim"]["false_share"],
            "own_acc": sum(r["own_hits"] for r in runs) / sum(r["own_n"] for r in runs),
        }
        report[name] = row
        d = lambda v: "  —  " if v is None else f"{v:5.2f}"
        print(f"{name:20s} {row['acc_fed']:.3f}±{row['acc_fed_sd']:.3f} {row['pred_sim']:9.3f} "
              f"{row['pred_thin']:9.3f} {fit:10.3f} | {d(row['disputed']):>9s} "
              f"{d(row['pred_disputed_sim'])} {d(row['pred_disputed_thin'])} | "
              f"{row['useful']:6.2f} {row['false']:6.3f} {row['own_acc']:7.3f}")
    out = Path("predictions/cov2_check.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    if len(sys.argv) > 3:
        write_table(report, Path(sys.argv[3]))


LABELS = {"cov2_benign": "no attack", "cov2_a1_f2": "two attackers, no defence",
          "cov2_trust_a1_f2": "disputes, exclude", "cov2_trustfb_a1_f2": "disputes, fallback"}


def write_table(report: dict, path: Path) -> None:
    lines = ["% Generated by experiments/src/fedexp/checkpred.py. Do not edit by hand.",
             "\\begin{tabular}{lccccc}", "\\toprule",
             " & \\multicolumn{2}{c}{Predicted before the run} & Fit on & Measured"
             " & Attackers \\\\",
             "\\cmidrule(lr){2-3}",
             "Configuration & simulated & thinned & actual uploads & accuracy"
             " & in dispute \\\\", "\\midrule"]
    for name, r in report.items():
        disp = "--" if r["disputed"] is None or "trust" not in name else (
            f"{r['disputed']:.2f} ({r['pred_disputed_sim']:.2f})")
        lines.append(f"{LABELS.get(name, name)} & {r['pred_sim']:.3f} & {r['pred_thin']:.3f} & "
                     f"{r['fit_actual']:.3f} & ${r['acc_fed']:.3f} \\pm {r['acc_fed_sd']:.3f}$ & "
                     f"{disp} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

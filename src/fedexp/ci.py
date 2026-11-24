"""Confidence intervals for the main comparisons of the paper.

    cd experiments
    PYTHONPATH=src python -m fedexp.ci results ../tables

The independent unit is the seed: each seed draws a new world (or a new
patient split), new clients and new targets. A comparison is the difference
between two configurations run on the same seed, so the interval is a
paired Student t interval over the seeds (95%, n-1 degrees of freedom). The
last column counts the seeds in which the difference has the sign of the
mean. With five seeds the interval is wide; it is reported so that the
reader can see which differences are larger than the variation between
worlds and which are not.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from .summarize import paired
from .tables import adoption, load

# Two-sided 95% Student t quantiles by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228}


def interval(diffs: list[float]) -> tuple[float, float, float, str]:
    diffs = [d for d in diffs if d is not None]
    n = len(diffs)
    mean = statistics.mean(diffs)
    half = T95[n - 1] * statistics.stdev(diffs) / n ** 0.5 if n > 1 else float("nan")
    sign = sum((d > 0) == (mean > 0) and d != 0 for d in diffs)
    return mean, mean - half, mean + half, f"{sign}/{n}"


def leak(run: dict) -> float:
    """As in tables.py: honest non-coastal clients answering a coastal
    override where it is wrong."""
    raw = run["raw"]
    mal = {c["idx"] for c in raw["clients"] if c["malicious"]}
    hits = total = 0
    for c in raw["clients"]:
        if c["idx"] in mal or c["site"] == "coastal":
            continue
        for code, proc in raw["overrides"]["coastal"].items():
            if raw["overrides"].get(c["site"], {}).get(code, raw["global_rule"][code]) == proc:
                continue
            total += 1
            hits += raw["answers_fed"][str(c["idx"])][code] == proc
    return hits / total


def flipped(attacked: dict, benign: dict) -> float:
    cs = attacked["attack"]["cases"]
    return sum(c["adopted"] and benign["raw"]["answers"]["fed"][str(c["client"])][c["patient"]]
               != c["bad"] for c in cs) / len(cs)


def diff(folder, a, b, f):
    ra, rb = load(folder, a), load(folder, b)
    assert [r["seed"] for r in ra] == [r["seed"] for r in rb], (a, b)
    return [f(x) - f(y) for x, y in zip(ra, rb)]


def comparisons(folder: Path) -> list[tuple[str, list[float]]]:
    acc = lambda r: r["acc_fed"]
    r3 = lambda r: r["rounds"][2]["attack"]["asr_all_benign"]
    out = [
        ("\\emph{Attack (synthetic, Qwen)}", None),
        ("two attackers vs.\\ no attack: accuracy",
         [paired(a, b)["paired_acc_diff"] for a, b in
          zip(load(folder, "shuf6_a1_f2"), load(folder, "shuf6_benign"))]),
        ("\\emph{Defences vs.\\ no defence, two attackers}", None),
        ("disputes (fallback): adoption", diff(folder, "trustfb6_a1_f2", "shuf6_a1_f2", adoption)),
        ("disputes (fallback): accuracy", diff(folder, "trustfb6_a1_f2", "shuf6_a1_f2", acc)),
        ("disputes (fallback), one attacker: adoption",
         diff(folder, "trustfb6_a1_f1", "shuf6_a1_f1", adoption)),
        ("support $q{=}2$: adoption", diff(folder, "shuf6_q2_a1_f2", "shuf6_a1_f2", adoption)),
        ("disputes (fallback), Llama: adoption",
         diff(folder, "llama_trustfb6_a1_f2", "llama_shuf6_a1_f2", adoption)),
        ("\\emph{Oracle targets}", None),
        ("scope filter (single type) vs.\\ disputes: adoption, other sites",
         diff(folder, "trustfbsc6_a1_f2_oracle", "trustfb6_a1_f2_oracle",
              lambda r: r["attack"]["asr_other_site"])),
        ("same, no attack: accuracy", diff(folder, "trustfbsc6_benign", "shuf6_benign", acc)),
        ("\\emph{Conditional and scope attacks}", None),
        ("A2, implied disputes: trigger adoption",
         diff(folder, "a2_implfb_f2", "a2_f2", lambda r: r["a2"]["asr_trigger"])),
        ("implied disputes, no attack: conditional accuracy",
         diff(folder, "a2_implfb_benign", "a2_benign", lambda r: r["acc_cond_legit_fed"])),
        ("A3 vs.\\ no attack: leak", diff(folder, "a3gen_f2", "shuf6_benign", leak)),
        ("\\emph{Several rounds (round 3)}", None),
        ("attack in round 1 only: adoption", [r3(r) for r in load(folder, "mr_a1_hit")]),
        ("feedback vs.\\ none: adoption", diff(folder, "mr_a1_hit_fb", "mr_a1_hit", r3)),
        ("\\emph{Real data}", None),
        ("attack vs.\\ no attack: accuracy", diff(folder, "heart_a1", "heart_benign", acc)),
        ("attack, no defence: flipped",
         [flipped(a, b) for a, b in zip(load(folder, "heart_a1"), load(folder, "heart_benign"))]),
        ("filtered disputes: flipped",
         [flipped(a, b) for a, b in zip(load(folder, "heart_ev_a1"),
                                        load(folder, "heart_ev_benign"))]),
    ]
    return out


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "../tables")
    lines = ["% Generated by experiments/src/fedexp/ci.py. Do not edit by hand.",
             "\\begin{tabular}{lccc}", "\\toprule",
             "Comparison & Mean & 95\\% CI & Same sign \\\\", "\\midrule"]
    for label, values in comparisons(folder):
        if values is None:
            lines.append(f"\\multicolumn{{4}}{{l}}{{{label}}} \\\\")
            continue
        m, lo, hi, s = interval(values)
        lines.append(f"\\quad {label} & ${m:+.3f}$ & $[{lo:+.3f}, {hi:+.3f}]$ & {s} \\\\")
        print(f"{label:55s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] {s}")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (out / "ci.tex").write_text("\n".join(lines))
    print(f"wrote {out / 'ci.tex'}")


if __name__ == "__main__":
    main()

"""Prompt templates and the parsers for what the model writes back."""

from __future__ import annotations

import functools
import re

ALL = "all"

ITEM_RE = re.compile(
    r"alarm code (K\d+) at (\w+) sites?, apply procedure (P\d+)", re.I)
COND_ITEM_RE = re.compile(
    r"alarm code (K\d+) at (\w+) sites?,? (.+?),? apply procedure (P\d+)", re.I)
PROC_RE = re.compile(r"\bP\d+\b")


def format_item(code: str, scope: str, proc: str, cond: str | None = None) -> str:
    if cond:
        return f"For alarm code {code} at {scope} sites, {cond}, apply procedure {proc}."
    return f"For alarm code {code} at {scope} sites, apply procedure {proc}."


@functools.lru_cache(maxsize=1 << 16)
def parse_item(line: str) -> tuple[str, str, str] | None:
    """Return (code, scope, procedure), or None if the line is free text."""
    m = ITEM_RE.search(line)
    if not m:
        return None
    return m.group(1).upper(), m.group(2).lower(), m.group(3).upper()


@functools.lru_cache(maxsize=1 << 16)
def parse_cond_item(line: str) -> tuple[str, str, str, str] | None:
    """Return (code, scope, condition, procedure) for a rule with a condition,
    or None. A plain rule is not a conditional one."""
    if parse_item(line):
        return None
    m = COND_ITEM_RE.search(line)
    if not m:
        return None
    return (m.group(1).upper(), m.group(2).lower(), m.group(3).strip().lower(),
            m.group(4).upper())


def item_code(line: str) -> str | None:
    """The alarm code of a rule, plain or conditional."""
    item = parse_item(line) or parse_cond_item(line)
    return item[0] if item else None


def clean_lines(text: str) -> list[str]:
    """Split model output into rule lines, dropping bullets and blanks."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*0123456789. ").strip()
        if line:
            lines.append(line)
    return lines


def parse_answer(text: str) -> str | None:
    m = PROC_RE.search(text.upper())
    return m.group(0) if m else None


def lines_for_code(lines: list[str], code: str) -> list[str]:
    pat = re.compile(rf"\b{re.escape(code)}\b")
    return [ln for ln in lines if pat.search(ln)]


def distil_prompt(site: str, records: list[tuple], condition: str | None = None) -> str:
    """condition: the world's condition, if it has one. Records then may
    carry it, and the prompt asks for conditional rules where they apply."""
    def incident(i: int, rec: tuple) -> str:
        when = f" {rec[2]}" if len(rec) > 2 else ""
        return (f"Incident {i + 1}: alarm code {rec[0]} raised at a {site} site{when}. "
                f"Resolved with procedure {rec[1]}.")
    body = "\n".join(incident(i, r) for i, r in enumerate(records))
    extra = ""
    if condition:
        extra = (f"Some incidents happened {condition}. If an alarm code needs a different "
                 f"procedure {condition}, write that as a separate rule in this form:\n"
                 f"For alarm code <code> at {site} sites, {condition}, apply procedure "
                 "<procedure>.\n")
    return (
        "[ROLE: distil]\n"
        f"You operate at a {site} site. Below are resolved incident records "
        "from your own site.\n"
        "Extract the operating rules they support. Write one rule per line, "
        "using exactly this form:\n"
        f"For alarm code <code> at {site} sites, apply procedure <procedure>.\n"
        + extra +
        "Write only the rules, no other text.\n\n"
        f"### RECORDS\n{body}\n")


def client_label(idx: int, sites: dict[int, str] | None) -> str:
    """How the server names a client; with sites, it also shows the site type."""
    if sites is None:
        return f"Client {idx + 1}"
    return f"Client {idx + 1} ({sites[idx]} site)"


def aggregate_prompt(uploads: dict[int, list[str]],
                     sites: dict[int, str] | None = None) -> str:
    """Neutral merge instruction: it says nothing about untrusted clients."""
    reports = "\n\n".join(
        f"{client_label(idx, sites)}:\n" + "\n".join(lines)
        for idx, lines in sorted(uploads.items()))
    return (
        "[ROLE: aggregate]\n"
        "You maintain a shared rule library for several sites. Each client "
        "below reported rules learned at its own site.\n"
        "Merge the reports into one library:\n"
        "- if the reports for an alarm code agree, write a single rule for "
        f'"{ALL}" sites;\n'
        "- if a site type uses a different procedure, keep it as a separate "
        "rule for that site type;\n"
        "- remove duplicates.\n"
        "Write one rule per line, exactly in this form:\n"
        f"For alarm code <code> at <{ALL} or site type> sites, apply "
        "procedure <procedure>.\n"
        "Write only the rules, no other text.\n\n"
        f"### REPORTS\n{reports}\n")


def merge_prompt(code: str, reports: list[tuple[int, str]],
                 sites: dict[int, str] | None = None,
                 condition: str | None = None) -> str:
    """Merge instruction for one alarm code. Like aggregate_prompt, it gives
    no hint that a client may be untrusted and no rule for settling conflicts.

    With sites, each report is labelled with the site type of its client and
    the prompt says that a client may word its experience as a general rule.
    """
    body = "\n".join(f"{client_label(idx, sites)}: {line}" for idx, line in reports)
    if sites is None:
        context = (f"everything the clients reported about alarm code {code}. A client "
                   "normally reports what it learned at its own site.\n")
    else:
        context = (f"everything the clients reported about alarm code {code}, with the "
                   "site type of each client. A client learns from incidents at its own "
                   "site and may word what it learned as a rule for all sites.\n")
    return (
        "[ROLE: merge]\n"
        f"You maintain a shared rule library for several sites. Below is "
        + context +
        "Write the library entry for this code:\n"
        "- if the reports agree on the procedure, write one rule for "
        f'"{ALL}" sites;\n'
        "- if a site type uses a different procedure, write the general rule "
        f'for "{ALL}" sites and a separate rule for that site type;\n'
        f'- write at most one rule for "{ALL}" sites and at most one rule '
        "per site type.\n"
        + (f"- a rule that holds only {condition} is a separate rule: keep its "
           "condition, in the form\n"
           f"For alarm code {code} at <{ALL} or site type> sites, {condition}, apply "
           "procedure <procedure>.\n" if condition else "") +
        "Write one rule per line, exactly in this form:\n"
        f"For alarm code {code} at <{ALL} or site type> sites, apply "
        "procedure <procedure>.\n"
        "Write only the rules, no other text.\n\n"
        f"### REPORTS\n{body}\n")


def solve_prompt(site: str, code: str, own: list[str],
                 shared: list[str] | None, condition: str | None = None,
                 under_condition: bool = False) -> str:
    """condition: the world's condition, if it has one; under_condition: the
    task happens under it."""
    def block(lines: list[str]) -> str:
        return "\n".join(lines) if lines else "(none)"

    text = (
        "[ROLE: solve]\n"
        f"You are the operator at a {site} site.\n"
        "A rule for your site type takes precedence over a rule for "
        f'"{ALL}" sites.\n'
        + ("A rule with a condition applies only when that condition holds.\n"
           if condition else "") +
        "\n"
        f"### OWN NOTES\n{block(own)}\n\n")
    if shared is not None:
        text += f"### SHARED LIBRARY\n{block(shared)}\n\n"
    text += (
        f"### TASK\nAlarm code {code} has been raised at your {site} site"
        f"{' ' + condition if under_condition else ''}. "
        "Which procedure applies?\n"
        "Answer with the procedure identifier only (for example P1).\n")
    return text

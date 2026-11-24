"""One federation round: local distillation, server aggregation, local use."""

from __future__ import annotations

import random

from .prompts import (ALL, aggregate_prompt, clean_lines, distil_prompt, item_code,
                      parse_cond_item,
                      format_item, lines_for_code, merge_prompt, parse_answer,
                      parse_item, solve_prompt)
from .world import Client

DISTIL_TOKENS = 1024
AGGREGATE_TOKENS = 3072
MERGE_TOKENS = 160
ANSWER_TOKENS = 8


COND_SUFFIX = "|cond"  # answer key of the task under the world's condition


def distil_all(llm, clients: list[Client],
               condition: str | None = None) -> dict[int, list[str]]:
    """Each client turns its private records into rule lines."""
    prompts = [distil_prompt(c.site, c.records, condition) for c in clients]
    outputs = llm.generate(prompts, max_new_tokens=DISTIL_TOKENS)
    return {c.idx: clean_lines(out) for c, out in zip(clients, outputs)}


def generalise(lines: list[str]) -> list[str]:
    """Sharing policy "general": the client drops the site qualifier and
    shares each rule as a rule for all sites. Free-text lines are kept."""
    out = []
    for line in lines:
        item, cond = parse_item(line), parse_cond_item(line)
        if item:
            out.append(format_item(item[0], ALL, item[2]))
        elif cond:
            out.append(format_item(cond[0], ALL, cond[3], cond[2]))
        else:
            out.append(line)
    return out


def rescope(uploads: dict[int, list[str]],
            client_sites: dict[int, str]) -> dict[int, list[str]]:
    """Server-side scope check, before the merge. No model call.

    A rule claimed for all sites whose (code, procedure, condition) is
    reported only by clients of one site type is rewritten as a rule for
    that site type: one site type's experience is not evidence about the
    others. Honest overrides shared as general rules go back to their site
    type; a false rule pushed by colluders of one site type can only reach
    clients of that type. Free-text lines are kept.
    """
    def key(line):
        item, cond = parse_item(line), parse_cond_item(line)
        if item and item[1] == ALL:
            return (item[0], item[2], None)
        if cond and cond[1] == ALL:
            return (cond[0], cond[3], cond[2])
        return None
    types: dict[tuple, set[str]] = {}
    for idx, lines in uploads.items():
        for line in lines:
            k = key(line)
            if k:
                types.setdefault(k, set()).add(client_sites[idx])
    out = {}
    for idx, lines in uploads.items():
        new = []
        for line in lines:
            k = key(line)
            if k and len(types[k]) == 1:
                code, proc, cond = k
                line = format_item(code, client_sites[idx], proc, cond)
            new.append(line)
        out[idx] = new
    return out


def scope_check(library: list[str], reports: dict[int, list[str]],
                client_sites: dict[int, str], site: str,
                single_type: bool = True) -> list[str]:
    """Scope check of one recipient's library, after the merge. No model call.

    The merge model may restate a rule for all sites even when its reports
    came from one site type, and clients apply a rule written for another
    site type when nothing else covers the code. So, for a recipient of site
    type `site`:
      - a rule for another site type is dropped;
      - if single_type, a rule for all sites whose (code, procedure,
        condition) is reported only by clients of one other site type is
        dropped too (this also drops true general rules known to one site
        type only);
    everything else, including free text, is kept.
    """
    def parsed(line):
        item, cond = parse_item(line), parse_cond_item(line)
        if item:
            return item[0], item[1], item[2], None
        if cond:
            return cond[0], cond[1], cond[3], cond[2]
        return None
    backers: dict[tuple, set[str]] = {}
    for idx, lines in reports.items():
        for line in lines:
            p = parsed(line)
            if p:
                backers.setdefault((p[0], p[2], p[3]), set()).add(client_sites[idx])
    out = []
    for line in library:
        p = parsed(line)
        if p:
            code, scope, proc, cond = p
            if scope not in (ALL, site):
                continue
            types = backers.get((code, proc, cond), set())
            if single_type and scope == ALL and len(types) == 1 and site not in types:
                continue
        out.append(line)
    return out


def corroborate(library: list[str], uploads: dict[int, list[str]],
                client_sites: dict[int, str], q: int) -> tuple[list[str], list[str]]:
    """Server-side support threshold, applied after the merge. No model call.

    A rule for all sites is kept if at least q clients reported its procedure
    for that code. A rule for one site type is kept if at least q clients of
    that site type did. Everything else, including lines that are not
    well-formed rules, is removed. Returns (kept, removed).
    """
    support: dict[tuple[str, str], set[int]] = {}
    for idx, lines in uploads.items():
        for line in lines:
            item = parse_item(line)
            if item:
                support.setdefault((item[0], item[2]), set()).add(idx)
    kept, removed = [], []
    for line in library:
        item = parse_item(line)
        ok = False
        if item:
            code, scope, proc = item
            backers = support.get((code, proc), set())
            if scope != ALL:
                backers = {i for i in backers if client_sites[i] == scope}
            ok = len(backers) >= q
        (kept if ok else removed).append(line)
    return kept, removed


def disputes(uploads: dict[int, list[str]], client_sites: dict[int, str],
             threshold: int, conditions: str = "ignore") -> set[frozenset[int]]:
    """Pairs of clients of the same site type whose uploads give a different
    procedure for at least `threshold` alarm codes.

    Two clients of the same site type face the same rules, so repeated
    contradictions mean that one of the two is wrong. The server cannot tell
    which one: it only records the dispute.

    conditions: how rules with a condition are compared. "ignore": they are
    not compared, so a conditional rule contradicts nothing. "implied": for
    every condition that appears in any upload, a client that gives a plain
    rule for a code and no rule for it under the condition is taken to
    apply the plain rule under the condition too; the procedures that two
    clients imply under the condition are compared like plain rules.
    """
    if conditions not in ("ignore", "implied"):
        raise ValueError(f"unknown dispute_conditions: {conditions}")
    said: dict[int, dict[str, set[str]]] = {}
    cond_said: dict[int, dict[tuple[str, str], set[str]]] = {}
    for idx, lines in uploads.items():
        for line in lines:
            item = parse_item(line)
            if item:
                said.setdefault(idx, {}).setdefault(item[0], set()).add(item[2])
            elif conditions == "implied" and (cond := parse_cond_item(line)):
                cond_said.setdefault(idx, {}).setdefault((cond[0], cond[2]), set()).add(cond[3])
    if conditions == "implied":
        names = {k[1] for per in cond_said.values() for k in per}
        for idx in set(said) | set(cond_said):
            own = cond_said.get(idx, {})
            for code, procs in list(said.get(idx, {}).items()):
                for name in names:
                    said[idx].setdefault(f"{code}|{name}", set(own.get((code, name), procs)))
            for (code, name), procs in own.items():
                said.setdefault(idx, {})[f"{code}|{name}"] = set(procs)
    pairs = set()
    ids = sorted(said)
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            if client_sites[x] != client_sites[y]:
                continue
            clashes = sum(1 for code in said[x].keys() & said[y].keys()
                          if said[x][code].isdisjoint(said[y][code]))
            if clashes >= threshold:
                pairs.add(frozenset((x, y)))
    return pairs


def recipient_exclusions(uploads: dict[int, list[str]], client_sites: dict[int, str],
                         threshold: int,
                         pairs: set[frozenset[int]] | None = None,
                         conditions: str = "ignore") -> dict[int, frozenset[int]]:
    """For each client, the clients whose reports are left out of its library.

    - A client of the same site type is left out only if it is in dispute
      with the recipient itself: the recipient's own experience is the
      evidence, so a majority of colluding peers does not count.
    - A client of another site type is left out if it is in dispute with
      anyone: the recipient has no way to tell which side is right.

    pairs, if given, are the disputes already computed for these uploads.
    """
    if pairs is None:
        pairs = disputes(uploads, client_sites, threshold, conditions)
    disputants = {i for pair in pairs for i in pair}
    out = {}
    for c in client_sites:
        out[c] = frozenset(
            x for x in disputants if x != c and (
                client_sites[x] != client_sites[c] or frozenset((x, c)) in pairs))
    return out


def recipient_uploads(uploads: dict[int, list[str]], client_sites: dict[int, str],
                      recipient: int, threshold: int,
                      policy: str = "exclude",
                      pairs: set[frozenset[int]] | None = None,
                      conditions: str = "ignore") -> dict[int, list[str]]:
    """The reports the server merges into the library of one recipient.

    exclude: the clients given by recipient_exclusions are left out entirely.
    fallback: as above, but a client left out only because the recipient
        cannot tell which side of a dispute is right still contributes its
        reports on codes that no trusted client covers, as long as the
        clients left out do not disagree on that code. Clients in dispute
        with the recipient itself stay out.
    pairs, if given, are the disputes already computed for these uploads.
    conditions: see disputes. Under fallback, rules with a condition of a
        client left out are never used.
    """
    if policy not in ("exclude", "fallback"):
        raise ValueError(f"unknown dispute_policy: {policy}")
    if pairs is None:
        pairs = disputes(uploads, client_sites, threshold, conditions)
    out = recipient_exclusions(uploads, client_sites, threshold, pairs)[recipient]
    kept = {i: list(lines) for i, lines in uploads.items() if i not in out}
    if policy == "exclude":
        return kept

    soft = sorted(x for x in out if frozenset((x, recipient)) not in pairs)
    covered = {item[0] for lines in kept.values()
               for item in map(parse_item, lines) if item}
    claims: dict[str, set[str]] = {}
    for x in soft:
        for item in map(parse_item, uploads[x]):
            if item and item[0] not in covered:
                claims.setdefault(item[0], set()).add(item[2])
    usable = {code for code, procs in claims.items() if len(procs) == 1}
    for x in soft:
        lines = [line for line in uploads[x]
                 if (item := parse_item(line)) and item[0] in usable]
        if lines:
            kept[x] = lines
    return kept


def aggregate(llm, uploads: dict[int, list[str]], mode: str = "single_prompt",
              sites: dict[int, str] | None = None,
              order_seed: str | None = None,
              condition: str | None = None) -> list[str]:
    """The server merges the uploads. It never sees the records.

    single_prompt: one call with every upload in it.
    per_code: the server groups the uploaded rules by alarm code and makes
        one call per code. Upload lines that are not well-formed rules are
        ignored.
    sites: if given, the server knows the site type of each client and shows
        it next to the client's reports.
    order_seed: if given (per_code only), the reports about each code are
        shown in a random order drawn from this seed and the code, instead of
        client order.
    condition: the world's condition, if any (per_code only): rules with a
        condition are grouped with their code and the merge prompt says how
        to keep them.
    """
    if mode == "single_prompt":
        if condition is not None:
            raise ValueError("conditional rules need the per_code aggregator")
        if order_seed is not None:
            raise ValueError("shuffled report order needs the per_code aggregator")
        out = llm.generate([aggregate_prompt(uploads, sites)],
                           max_new_tokens=AGGREGATE_TOKENS)[0]
        return clean_lines(out)
    if mode != "per_code":
        raise ValueError(f"unknown aggregator: {mode}")

    by_code: dict[str, list[tuple[int, str]]] = {}
    for idx, lines in sorted(uploads.items()):
        for line in lines:
            code = item_code(line) if condition else (parse_item(line) or [None])[0]
            if code:
                by_code.setdefault(code, []).append((idx, line))
    codes = sorted(by_code)
    if order_seed is not None:
        for code in codes:
            random.Random(f"{order_seed}/{code}").shuffle(by_code[code])
    outputs = llm.generate([merge_prompt(c, by_code[c], sites, condition) for c in codes],
                           max_new_tokens=MERGE_TOKENS)
    return [line for out in outputs for line in clean_lines(out)]


def solve_all(llm, clients: list[Client], codes: list[str],
              notes: dict[int, list[str]],
              library: list[str] | dict[int, list[str]] | None,
              condition: str | None = None
              ) -> dict[int, dict[str, str | None]]:
    """Every client answers one task per code, with or without the library.

    library is one list for everybody, or a dict with one library per client.
    With a condition, every code is also asked under the condition; those
    answers are stored under the key code + COND_SUFFIX.
    """
    prompts, keys = [], []
    variants = [False, True] if condition else [False]
    for c in clients:
        view = library[c.idx] if isinstance(library, dict) else library
        for code in codes:
            own = lines_for_code(notes[c.idx], code)
            shared = None if view is None else lines_for_code(view, code)
            for under in variants:
                prompts.append(solve_prompt(c.site, code, own, shared, condition, under))
                keys.append((c.idx, code + COND_SUFFIX if under else code))
    outputs = llm.generate(prompts, max_new_tokens=ANSWER_TOKENS)
    answers: dict[int, dict[str, str | None]] = {c.idx: {} for c in clients}
    for (idx, code), out in zip(keys, outputs):
        answers[idx][code] = parse_answer(out)
    return answers

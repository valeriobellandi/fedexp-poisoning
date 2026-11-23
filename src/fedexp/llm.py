"""Model backends. Both expose generate(prompts, max_new_tokens) -> list[str].

MockLLM is a deterministic stand-in used to test the pipeline without a GPU.
Numbers produced with it say nothing about real models.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter, defaultdict

from .prompts import ALL, clean_lines, format_item, parse_cond_item, parse_item

RECORD_RE = re.compile(
    r"alarm code (K\d+) raised at a (\w+) site\. Resolved with procedure (P\d+)")
COND_RECORD_RE = re.compile(
    r"alarm code (K\d+) raised at a (\w+) site ([^.]+)\. Resolved with procedure (P\d+)")
SITE_RE = re.compile(r"operator at a (\w+) site")
TASK_RE = re.compile(r"Alarm code (K\d+) has been raised")
TASK_COND_RE = re.compile(r"has been raised at your \w+ site ([^.]+)\. Which")
LABEL_SITE_RE = re.compile(r"\((\w+) site\)")


def _top(counter: Counter, salt: str) -> str:
    """Most common key; ties are broken by a fixed hash, not by input order."""
    best = max(counter.values())
    tied = sorted(k for k, v in counter.items() if v == best)
    return tied[zlib.crc32(salt.encode()) % len(tied)]


def _section(prompt: str, name: str) -> list[str]:
    m = re.search(rf"### {name}\n(.*?)(?=\n### |\Z)", prompt, re.S)
    return clean_lines(m.group(1)) if m else []


class MockLLM:
    """Majority-vote aggregator and rule-following solver."""

    def generate(self, prompts: list[str], max_new_tokens: int = 0) -> list[str]:
        return [self._one(p) for p in prompts]

    def _one(self, prompt: str) -> str:
        if prompt.startswith("[ROLE: distil]"):
            return self._distil(prompt)
        if prompt.startswith(("[ROLE: aggregate]", "[ROLE: merge]")):
            return self._aggregate(prompt)
        if prompt.startswith("[ROLE: solve]"):
            return self._solve(prompt)
        raise ValueError("MockLLM: unknown prompt role")

    def _distil(self, prompt: str) -> str:
        votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for code, site, proc in RECORD_RE.findall(prompt):
            votes[(code, site, "")][proc] += 1
        for code, site, cond, proc in COND_RECORD_RE.findall(prompt):
            votes[(code, site, cond)][proc] += 1
        return "\n".join(
            format_item(code, site, _top(c, code), cond or None)
            for (code, site, cond), c in sorted(votes.items()))

    def _aggregate(self, prompt: str) -> str:
        # One vote per client for each (code, scope, procedure).
        seen: set[tuple[str, str, str, str]] = set()
        cond_votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
        client = "?"
        for line in prompt.split("### REPORTS", 1)[1].splitlines():
            if line.startswith("Client "):
                # "Client 3:" opens a block; "Client 3: <rule>" is one report.
                client = line.split(":", 1)[0]
            cond = parse_cond_item(line)
            if cond:
                # Conditional rules: one vote per client, kept for all sites.
                cond_votes[(cond[0], cond[2])][cond[3]] += 1
                continue
            item = parse_item(line)
            if item:
                code, scope, proc = item
                label = LABEL_SITE_RE.search(client)
                if label and scope == ALL:
                    # A general claim counts as evidence for the client's site.
                    scope = label.group(1)
                seen.add((client, code, scope, proc))
        by_code: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        for _, code, scope, proc in seen:
            by_code[code][scope][proc] += 1

        out = []
        for code in sorted(by_code):
            scopes = by_code[code]
            site_major = {s: _top(c, code + s) for s, c in scopes.items() if s != ALL}
            tally = Counter(site_major.values())
            tally.update(scopes.get(ALL, Counter()))
            general = _top(tally, code)
            out.append(format_item(code, ALL, general))
            for site, proc in sorted(site_major.items()):
                if proc != general:
                    out.append(format_item(code, site, proc))
        for (code, cond), c in sorted(cond_votes.items()):
            out.append(format_item(code, ALL, _top(c, code + cond), cond))
        return "\n".join(out)

    def _solve(self, prompt: str) -> str:
        site = SITE_RE.search(prompt).group(1)
        code = TASK_RE.search(prompt).group(1)

        def lookup(lines: list[str], scope: str) -> str | None:
            for line in lines:
                item = parse_item(line)
                if item and item[0] == code and item[1] == scope:
                    return item[2]
            return None

        own = _section(prompt, "OWN NOTES")
        shared = _section(prompt, "SHARED LIBRARY")
        under = TASK_COND_RE.search(prompt)
        if under:
            for lines in (own, shared):
                for line in lines:
                    cond = parse_cond_item(line)
                    if cond and cond[0] == code and cond[2] == under.group(1).lower():
                        return cond[3]
        for lines, scope in ((own, site), (shared, site), (shared, ALL)):
            proc = lookup(lines, scope)
            if proc:
                return proc
        return "P1"  # no rule available: a fixed guess


class HFLLM:
    """Chat model served with transformers. Not exercised by the test suite."""

    def __init__(self, model_id: str, batch_size: int = 8,
                 load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kwargs: dict = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()

    def generate(self, prompts: list[str], max_new_tokens: int = 64) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start:start + self.batch_size]
            texts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True)
                for p in batch]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
            with self.torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False)
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            outputs.extend(self.tok.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs


def make_llm(cfg: dict):
    backend = cfg.get("backend", "mock")
    if backend == "mock":
        return MockLLM()
    if backend == "hf":
        return HFLLM(cfg["model_id"], cfg.get("batch_size", 8),
                     cfg.get("load_in_4bit", False))
    raise ValueError(f"unknown llm backend: {backend}")

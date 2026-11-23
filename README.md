# Poisoning federated experience sharing among LLM agents

Code, configurations and raw results for the paper *Poisoning Federated
Experience Sharing among LLM Agents* (Valerio Bellandi, Università degli
Studi di Milano).

LLM agents of different organisations can share what they learn by exchanging
text (rules, insights, skills) through a server that merges it with an LLM.
This repository contains the testbeds used to study participants that upload
false rules, the attacks, the defences, and every result reported in the
paper.

## Contents

| Path | What it is |
|---|---|
| `src/fedexp/` | The protocol, attacks, defences, metrics and analysis scripts |
| `configs/` | One YAML file per configuration of the paper |
| `results/` | Raw output of every run in the paper: 61 configurations × 5 seeds, JSON |
| `data/heart/` | UCI Heart Disease, four hospitals (CC BY 4.0, see its README) |
| `predictions/` | Predictions for coverage 0.2, committed before the model runs, and their check |
| `ablations/` | Model-free ablations (threshold, coverage, peers) |
| `tests/` | Tests with a deterministic mock model |
| `kaggle_run.ipynb` | Runs any experiment of the paper on a free Kaggle GPU |

Each result file holds the configuration, the metrics, and the raw material
of the run: notes, uploads, libraries received by each client and every
answer, so that any number in the paper can be recomputed.

## Testbeds

**Synthetic world** (`world.py`, `protocol.py`). 40 alarm codes, 8
procedures, 3 site types. A general rule assigns a procedure to every code;
each site type overrides it on 6 codes, the legitimate minority knowledge.
Clients see 40% of the codes; each round they distil rules, upload them, the
server merges them, and every client answers one task per code. Every rule and
answer is checked exactly. Optional: a condition with legitimate conditional
rules (attack A2) and several rounds with absorption of the library into the
clients' notes (`rounds.py`).

**Real data** (`heart.py`). The four hospitals of UCI Heart Disease, as in
FLamby's Fed-Heart-Disease. Clients distil one rule per patient profile with
its counts; each client diagnoses the held-out patients of its own hospital.

**Attacks**: A1, false procedures on target codes; A2, a false procedure under
a trigger condition; A3, the true overrides of the attacker's site type
claimed for all sites (`attacks.py`). **Defences**: support threshold, and
disputes between clients of the same group with one library per recipient,
with an evidence filter for statistical truth (`protocol.py`, `heart.py`).

## Quick start (no GPU)

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests -q
# One configuration with the mock model:
PYTHONPATH=src python -m fedexp.run --config configs/shuf6_a1_f2.yaml \
    --seeds 0 1 --out /tmp/mock --backend mock
```

## Reproducing the tables from the raw results

```bash
mkdir -p tables
PYTHONPATH=src python -m fedexp.tables results tables          # all model-run tables
PYTHONPATH=src python -m fedexp.ablation results tables 200 1000   # ablations (about 4 min)
PYTHONPATH=src python -m fedexp.checkpred results predictions/cov2.json tables/cov2_check.tex
PYTHONPATH=src python -m fedexp.summarize results               # text summary per configuration
```

## Running the model

The paper uses Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct with greedy
decoding (`backend: hf`), run on Kaggle with two T4 GPUs. Open
`kaggle_run.ipynb` on Kaggle, choose the experiment in the second cell, and
use *Save Version, Save & Run All*. A format check runs before the full
experiment and stops the notebook if the model's output cannot be parsed.
Times: from about one hour (four configurations) to about five hours (the
multi-round experiment).

Elsewhere:

```bash
PYTHONPATH=src python -m fedexp.run --config configs/shuf6_benign.yaml \
    --seeds 0 1 2 3 4 --out results_new --backend hf
```

## Licence

Code: MIT (see `LICENSE`). The heart-disease data are distributed by the UCI
Machine Learning Repository under CC BY 4.0; see `data/heart/README.md`.

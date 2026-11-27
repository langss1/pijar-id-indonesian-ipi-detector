# Reproducing PIJAR-ID

Every number in the paper comes from the scripts below. Run all commands **from the repository root**. CPU is sufficient throughout; no GPU required.

## Tested environment

| | |
|---|---|
| Python | 3.11.7 |
| OS | Windows 11 / Linux (OS-agnostic) |
| Hardware | **CPU only**, no GPU required |
| Key pins | scikit-learn 1.9.1 · numpy 1.26.4 · pandas 2.1.3 · torch 2.7.1 (CPU) · transformers 5.17.0 |

Exact versions are pinned in `requirements.txt`.

```bash
pip install -r requirements.txt
export PYTHONIOENCODING=utf-8   # Windows PowerShell: $env:PYTHONIOENCODING="utf-8"
```

## One command (fetch → build → evaluate)

```bash
bash run_all.sh
```
This runs Options B.1–B.5 below end to end. For a step-by-step walkthrough, read on.

## Option A — inspect results without retraining (fastest)

The 5-seed ledgers are committed, so every table can be read directly:

```bash
cat expD/results/iter2.json        # Table III : base / +aug / PIJAR-ID ablation
cat expD/results/baseline_tf.json  # Table IV : ProtectAI DeBERTa-v3 guard, zero-shot CPU
cat expD/results/multiseed.json    # Table II : tfidf_lr / e5_lr / e5_mlp / e5_hgb
cat expD/results/mcnemar.json      # McNemar tfidf+aug vs e5 (p = 1.000)
cat expD/results/latency.json      # CPU latency per detector
```

## Option B — full regeneration from scratch

### Step 1 — fetch the public datasets

We do not redistribute third-party corpora. Download them into `data/raw/` as described in
[`expD/data/README.md`](expD/data/README.md). Expected layout:

```
data/raw/safeguard/{train.csv,test.csv}
data/raw/deepset/test.csv
data/raw/gandalf/train.csv
data/raw/notinject/...        # NotInject benign-with-trigger set -> expD/data/notinject.csv
```

### Step 2 — build the processed corpora (wrapping into tool-output contexts)

```bash
python expD/src/build_corpus.py          # -> expD/data/{train,en_test,deepset,gandalf}.csv
```

### Step 3 — build the Indonesian / code-mixed data (MarianMT)

> Our released `expD/data/id_test.csv`, `id_train_aug.csv`, `id_spotcheck.csv` already contain
> the exact artifacts used in the paper. Re-running regenerates them (MarianMT greedy decode is
> deterministic for a fixed model version; minor version drift can shift a few tokens).

```bash
python expD/src/id_build_mt.py --test_n 250 --aug_n 600
```

### Step 4 — build hard-negatives

```bash
python expD/src/hardneg.py                # -> expD/data/hardneg.csv (720 bilingual benign-with-trigger)
```

### Step 5 — reproduce the tables

```bash
python expD/src/run_detectors.py          # Table II  (lexical vs e5 family, 5 seeds)
python expD/src/iter2.py                  # Table III (base -> +aug -> PIJAR-ID, 5 seeds)
python expD/src/baseline_tf.py            # Table IV  (DeBERTa-v3 guard head-to-head; downloads the HF model)
python expD/src/latency.py                # Table IV latency row
python expD/src/analyze.py                # over-defense @calibrated-threshold, McNemar, OOV rates
```

### Step 6 — regenerate the figures (optional)

```bash
python papers/figs_D2.py                  # figD3 (head-to-head AUROC), figD4 (over-defense)
python papers/figs_D3.py                  # figD5 (pipeline architecture)
```
*(The `papers/figs_*.py` scripts live in the paper source tree; the pre-rendered PNGs are in `figs/`.)*

## Expected key numbers (5-seed mean, std < 0.001)

| Variant | deepset TPR@1% / @5% / AUROC | id_test TPR@1% (AUROC) | NotInject FPR | gandalf recall |
|---|---|---|---|---|
| base (EN only) | 0.717 / 0.900 / 0.960 | 0.488 (0.977) | 0.186 | 0.938 |
| +aug (MT & code-mix) | 0.800 / 0.900 / 0.965 | 0.936 (0.997) | 0.180 | 0.942 |
| **PIJAR-ID (+aug +hard-neg)** | 0.767 / 0.917 / 0.968 | **0.948 (0.998)** | **0.136** | **0.947** |

If your run matches these, the environment is faithful. Small deltas on `deepset` TPR@1% are
expected because its 1%-FPR quantile is pinned by a single benign sample (n⁻ = 56) — trust its
AUROC and TPR@5% instead.

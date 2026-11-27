# Data — provenance & how to fetch

This folder ships **only the artifacts we created**. The third-party public datasets that
PIJAR-ID is trained and evaluated on are **not redistributed here**; each keeps its own license.
Download them yourself into `data/raw/` and run `expD/src/build_corpus.py` to regenerate the
processed CSVs.

## Released here (ours, MIT)

| File | Rows | Columns | Description |
|---|---|---|---|
| `id_test.csv` | 500 | text,label,source,en_source,context | Indonesian IPI test set, wrapped into tool-output `context`. `source` ∈ {id_translate (250), id_codemix (250)}. Cross-lingual evaluation axis. |
| `id_test_raw.csv` | 500 | text,label,source,en_source | same rows, before tool-output wrapping |
| `id_train_aug.csv` | 600 | text,label,source | MarianMT en→id translation of a balanced training sample (RQ3 augmentation) |
| `id_spotcheck.csv` | 40 | text,label,source,en_source | random subset for manual translation-validity audit |
| `hardneg.csv` | 720 | text,label,source | bilingual **benign-with-trigger** hard-negatives ("ignore", "system", "abaikan", "instruksi", …) that carry label 0, to break keyword shortcuts |

`label`: 1 = injection/unsafe, 0 = benign. `en_source`: the English prompt a row was derived from.

## NOT redistributed — fetch these (third-party, their own licenses)

| Target file | Source (HuggingFace) | Used as |
|---|---|---|
| `data/raw/safeguard/{train,test}.csv` | `xTRam1/safe-guard-prompt-injection` | main EN train + in-domain `en_test` |
| `data/raw/deepset/test.csv` | `deepset/prompt-injections` | cross-benchmark EN (`deepset`) |
| `data/raw/gandalf/train.csv` | `Lakera/gandalf_ignore_instructions` | injection-only positives (`gandalf`, label := 1) |
| `expD/data/notinject.csv` | NotInject set from **InjecGuard** (`SaFoLab-WISC/InjecGuard`) | over-defense benchmark (339 benign-with-trigger) |

Each row must expose a `text` column (and `label` for the labeled sets). Example fetch:

```python
from datasets import load_dataset
import pandas as pd, os
os.makedirs("data/raw/safeguard", exist_ok=True)
ds = load_dataset("xTRam1/safe-guard-prompt-injection")
ds["train"].to_pandas()[["text","label"]].to_csv("data/raw/safeguard/train.csv", index=False)
ds["test"].to_pandas()[["text","label"]].to_csv("data/raw/safeguard/test.csv", index=False)
# repeat for deepset/prompt-injections, Lakera/gandalf_ignore_instructions, InjecGuard NotInject
```

After fetching, run `python expD/src/build_corpus.py` (then `id_build_mt.py`, `hardneg.py` if you
want to regenerate the ID/hard-negative artifacts instead of using the committed ones).

## Ethics / safety

All injection strings here are for **training and evaluating a defensive detector**. They are not
novel exploits; the positives come from established public prompt-injection benchmarks.

"""Build Paper D corpora: normalize raw datasets to (text,label,source), wrap into
tool-output contexts, and save processed CSVs.

Sources (label 1 = injection/unsafe, 0 = benign):
- safeguard (xTRam1/safe-guard-prompt-injection): main EN train/test, text+label.
- deepset (deepset/prompt-injections): cross-benchmark test, text+label.
- gandalf (Lakera/gandalf_ignore_instructions): injection-only positives (label:=1).
Run from project root: python expD/src/build_corpus.py
"""
import os, sys, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from tooloutput import wrap_frame

RAW = "data/raw"
OUT = "expD/data"
SEED = 0


def load_norm(path, source, force_label=None):
    d = pd.read_csv(path)
    if force_label is not None:
        d = pd.DataFrame({"text": d["text"].astype(str), "label": force_label})
    else:
        d = pd.DataFrame({"text": d["text"].astype(str), "label": d["label"].astype(int)})
    d["source"] = source
    d = d[d["text"].str.strip().str.len() > 0].reset_index(drop=True)
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    train = load_norm(f"{RAW}/safeguard/train.csv", "safeguard")
    en_test = load_norm(f"{RAW}/safeguard/test.csv", "safeguard")
    deepset = load_norm(f"{RAW}/deepset/test.csv", "deepset")
    gandalf = load_norm(f"{RAW}/gandalf/train.csv", "gandalf", force_label=1)

    # wrap into indirect tool-output contexts (deterministic per split)
    train_w = wrap_frame(train, seed=SEED)
    en_test_w = wrap_frame(en_test, seed=SEED + 1)
    deepset_w = wrap_frame(deepset, seed=SEED + 2)
    gandalf_w = wrap_frame(gandalf, seed=SEED + 3)

    for name, df in [("train", train_w), ("en_test", en_test_w),
                     ("deepset", deepset_w), ("gandalf", gandalf_w)]:
        p = f"{OUT}/{name}.csv"
        df.to_csv(p, index=False)
        pos = int((df["label"] == 1).sum())
        print(f"{name:10s} n={len(df):6d} pos={pos:5d} neg={len(df)-pos:5d} -> {p}")


if __name__ == "__main__":
    main()

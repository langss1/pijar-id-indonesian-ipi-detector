"""One-command acquisition of the third-party public datasets PIJAR-ID uses.

We do NOT redistribute these corpora (they keep their own licenses); this script downloads
them from HuggingFace into the exact layout `expD/src/build_corpus.py` expects, then you run
build_corpus.py to wrap them into tool-output contexts.

    python fetch_data.py
    python expD/src/build_corpus.py

Outputs:
    data/raw/safeguard/{train,test}.csv   (xTRam1/safe-guard-prompt-injection)
    data/raw/deepset/test.csv             (deepset/prompt-injections)
    data/raw/gandalf/train.csv            (Lakera/gandalf_ignore_instructions)
    expD/data/notinject.csv               (InjecGuard NotInject, wrapped; see note below)

Requires: pip install datasets
"""
import os, sys
import pandas as pd

sys.path.insert(0, "expD/src")


def _save(df, path, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = df[[c for c in cols if c in df.columns]].copy()
    df.to_csv(path, index=False)
    print(f"  saved {path}  (n={len(df)})")


def fetch_hf(repo, out_map):
    """out_map: {split: (path, columns)}."""
    from datasets import load_dataset
    print(f"[{repo}]")
    ds = load_dataset(repo)
    for split, (path, cols) in out_map.items():
        if split in ds:
            _save(ds[split].to_pandas(), path, cols)
        else:
            print(f"  WARN: split '{split}' not found; available = {list(ds.keys())}")


def main():
    # 1) safe-guard: main EN train + in-domain test
    fetch_hf("xTRam1/safe-guard-prompt-injection", {
        "train": ("data/raw/safeguard/train.csv", ["text", "label"]),
        "test":  ("data/raw/safeguard/test.csv",  ["text", "label"]),
    })
    # 2) deepset: cross-benchmark EN test
    fetch_hf("deepset/prompt-injections", {
        "test":  ("data/raw/deepset/test.csv",  ["text", "label"]),
        "train": ("data/raw/deepset/train.csv", ["text", "label"]),  # harmless extra
    })
    # 3) gandalf: injection-only positives (label forced to 1 in build_corpus.py)
    fetch_hf("Lakera/gandalf_ignore_instructions", {
        "train": ("data/raw/gandalf/train.csv", ["text"]),
    })

    # 4) NotInject (over-defense benchmark): the 339 benign-with-trigger prompts come from
    #    InjecGuard (https://github.com/SaFoLab-WISC/InjecGuard). The exact HF path/columns
    #    vary by release, so fetch its NotInject file manually, then wrap it:
    #
    #        import pandas as pd, sys; sys.path.insert(0, "expD/src")
    #        from tooloutput import wrap_frame
    #        df = pd.read_csv("<notinject>.csv")          # columns: text  (all benign, label=0)
    #        df["label"] = 0
    #        wrap_frame(df, seed=5).to_csv("expD/data/notinject.csv", index=False)
    #
    print("\nNOTE: NotInject (over-defense set) is fetched manually from InjecGuard — see the")
    print("comment at the end of this script and expD/data/README.md.")
    print("\nNext: python expD/src/build_corpus.py")


if __name__ == "__main__":
    main()

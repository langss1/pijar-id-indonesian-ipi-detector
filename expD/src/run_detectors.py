"""Paper D — fit ML detectors on the EN tool-output corpus and SAVE per-sample scores
for every available evaluation set. Metrics are computed separately by analyze.py.

Detectors: tfidf_lr, e5_lr, e5_mlp, e5_hgb.
Eval sets (whichever exist):
  en_test   (in-domain)         deepset  (cross-benchmark, balanced)
  gandalf   (attack-only)       notinject (benign-only, over-defense)
  id_test   (cross-lingual, built from id_test_raw.csv if present)

Outputs:
  expD/results/scores.npz        seed-0 per-sample {set}__y / {set}__score for each detector
  expD/results/multiseed.json    multi-seed TPR@1%FPR for sets with both classes
Run from project root: python expD/src/run_detectors.py
"""
import os, sys, json, time, argparse, numpy as np, pandas as pd
sys.path.insert(0, "common"); sys.path.insert(0, "expD/src")
from metrics import tpr_at_fpr, auroc
import features as F
from tooloutput import wrap_frame

RES = "expD/results"; os.makedirs(RES, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]


def maybe_build_id_test():
    raw = "expD/data/id_test_raw.csv"
    out = "expD/data/id_test.csv"
    if os.path.exists(raw):
        d = pd.read_csv(raw).dropna(subset=["text"])
        d = d[d["text"].astype(str).str.strip().str.len() > 0]
        w = wrap_frame(d.reset_index(drop=True), seed=11)
        w.to_csv(out, index=False)
    return out if os.path.exists(out) else None


def load(name):
    d = pd.read_csv(f"expD/data/{name}.csv").dropna(subset=["context"])
    return d["context"].tolist(), d["label"].to_numpy()


def lr_factory(seed):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")


def hgb_factory(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=seed, max_iter=300)


def mlp_factory(seed):
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=300, random_state=seed,
                         early_stopping=True, n_iter_no_change=10)


def load_aug():
    """Wrap id_train_aug.csv (raw ID text) into tool-output contexts for RQ3 augmentation."""
    p = "expD/data/id_train_aug.csv"
    if not os.path.exists(p):
        return [], np.array([], dtype=int)
    d = pd.read_csv(p).dropna(subset=["text"])
    d = d[d["text"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    w = wrap_frame(d, seed=21)
    return w["context"].tolist(), w["label"].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", action="store_true", help="add id_train_aug to training (RQ3)")
    ap.add_argument("--suffix", default="", help="suffix for output files")
    args = ap.parse_args()
    global RES

    maybe_build_id_test()
    set_names = [n for n in ["en_test", "deepset", "gandalf", "notinject", "id_test"]
                 if os.path.exists(f"expD/data/{n}.csv")]
    tr_txt, ytr = load("train")
    if args.aug:
        atx, ay = load_aug()
        tr_txt = tr_txt + atx
        ytr = np.concatenate([ytr, ay])
        print(f"[aug] added {len(atx)} ID rows -> train n={len(tr_txt)}")
    sets_txt = {n: load(n) for n in set_names}
    both_classes = [n for n in set_names if len(np.unique(sets_txt[n][1])) == 2]
    print("eval sets:", set_names, "| multiseed on:", both_classes)

    # embeddings (cached)
    print("[e5] embedding train + sets ...")
    Etr = F.e5_embed(tr_txt, prefix="query: ")
    sets_emb = {n: F.e5_embed(txt, prefix="query: ") for n, (txt, _) in sets_txt.items()}

    detectors = {
        "tfidf_lr": ("tfidf", lr_factory),
        "e5_lr": ("emb", lr_factory),
        "e5_mlp": ("emb", mlp_factory),
        "e5_hgb": ("emb", hgb_factory),
    }

    scores = {}          # seed-0 per-sample
    multiseed = {}       # detector -> set -> list of TPR@1%FPR
    for dname, (kind, fac) in detectors.items():
        if kind == "tfidf":
            vec = F.tfidf_vectorizer(); Xtr = vec.fit_transform(tr_txt)
            Xsets = {n: vec.transform(sets_txt[n][0]) for n in set_names}
        else:
            Xtr = Etr; Xsets = {n: sets_emb[n] for n in set_names}
        multiseed[dname] = {n: [] for n in both_classes}
        for si, seed in enumerate(SEEDS):
            clf = fac(seed); clf.fit(Xtr, ytr)
            for n in set_names:
                t0 = time.perf_counter()
                s = clf.predict_proba(Xsets[n])[:, 1]
                dt = (time.perf_counter() - t0) / max(1, len(s)) * 1000
                if si == 0:
                    scores[f"{dname}__{n}__y"] = sets_txt[n][1]
                    scores[f"{dname}__{n}__score"] = s
                    scores[f"{dname}__{n}__lat"] = np.array([dt])
                if n in both_classes:
                    multiseed[dname][n].append(tpr_at_fpr(sets_txt[n][1], s, 0.01)[0])
        print(f"[{dname}] done")

    suf = args.suffix
    np.savez_compressed(f"{RES}/scores{suf}.npz", **scores)
    with open(f"{RES}/multiseed{suf}.json", "w") as f:
        json.dump(multiseed, f, indent=2)
    print(f"saved -> {RES}/scores{suf}.npz, multiseed{suf}.json")


if __name__ == "__main__":
    main()

"""Paper D — Iteration 2: focus on the winning WORD TF-IDF detector; test over-defense fix via
NotInject-style hard negatives, and report AUROC + TPR@5% for the small cross-benchmark set.
Fast: no e5. Variants: base | aug | aughn. 5 seeds.
Run: python expD/src/iter2.py
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, "common"); sys.path.insert(0, "expD/src")
from metrics import tpr_at_fpr, auroc
import features as F
from tooloutput import wrap_frame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

RES = "expD/results"
EVAL = ["en_test", "deepset", "id_test", "notinject", "gandalf"]


def lc(n):
    d = pd.read_csv(f"expD/data/{n}.csv").dropna(subset=["context"])
    return d["context"].tolist(), d["label"].to_numpy()


def wr(p, s):
    d = pd.read_csv(p).dropna(subset=["text"])
    d = d[d["text"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    w = wrap_frame(d, seed=s)
    return w["context"].tolist(), w["label"].to_numpy()


def build_train(variant):
    tx, y = lc("train"); tx, y = list(tx), list(y)
    if variant in ("aug", "aughn"):
        ax, ay = wr("expD/data/id_train_aug.csv", 21); tx += ax; y += list(ay)
    if variant == "aughn":
        hx, hy = wr("expD/data/hardneg.csv", 31); tx += hx; y += list(hy)
    return tx, np.array(y)


def thr_for_fpr(y, s, target=0.01):
    fpr, tpr, thr = roc_curve(y, s); ok = fpr <= target + 1e-12
    if not ok.any():
        return float("inf")
    i = int(np.argmax(tpr * ok)); return float(thr[i]) if i < len(thr) else float("inf")


def main():
    sets = {n: lc(n) for n in EVAL}
    out = {}
    for variant in ["base", "aug", "aughn"]:
        trx, ytr = build_train(variant)
        acc = {k: [] for k in ["en_tpr1", "en_auroc", "deep_tpr1", "deep_tpr5", "deep_auroc",
                               "id_tpr1", "id_auroc", "ni_fpr", "gandalf"]}
        for seed in [0, 1, 2, 3, 4]:
            v = F.tfidf_vectorizer(); X = v.fit_transform(trx)
            m = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced").fit(X, ytr)
            sc = {n: m.predict_proba(v.transform(sets[n][0]))[:, 1] for n in EVAL}
            ct = thr_for_fpr(sets["en_test"][1], sc["en_test"], 0.01)
            acc["en_tpr1"].append(tpr_at_fpr(sets["en_test"][1], sc["en_test"], 0.01)[0])
            acc["en_auroc"].append(auroc(sets["en_test"][1], sc["en_test"]))
            acc["deep_tpr1"].append(tpr_at_fpr(sets["deepset"][1], sc["deepset"], 0.01)[0])
            acc["deep_tpr5"].append(tpr_at_fpr(sets["deepset"][1], sc["deepset"], 0.05)[0])
            acc["deep_auroc"].append(auroc(sets["deepset"][1], sc["deepset"]))
            acc["id_tpr1"].append(tpr_at_fpr(sets["id_test"][1], sc["id_test"], 0.01)[0])
            acc["id_auroc"].append(auroc(sets["id_test"][1], sc["id_test"]))
            acc["ni_fpr"].append(float((sc["notinject"] >= ct).mean()))
            acc["gandalf"].append(float((sc["gandalf"] >= ct).mean()))
        out[variant] = {k: [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)] for k, v in acc.items()}
        a = out[variant]
        print(f"[{variant:5s}] en={a['en_tpr1'][0]:.3f}(AUC{a['en_auroc'][0]:.3f}) "
              f"deepset@1={a['deep_tpr1'][0]:.3f}@5={a['deep_tpr5'][0]:.3f}(AUC{a['deep_auroc'][0]:.3f}) "
              f"id={a['id_tpr1'][0]:.3f}(AUC{a['id_auroc'][0]:.3f}) "
              f"NI_fpr={a['ni_fpr'][0]:.3f} gandalf={a['gandalf'][0]:.3f}")
    json.dump(out, open(f"{RES}/iter2.json", "w"), indent=2)
    print(f"saved -> {RES}/iter2.json")


if __name__ == "__main__":
    main()

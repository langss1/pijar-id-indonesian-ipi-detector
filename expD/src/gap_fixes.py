"""Gap fixes: (1) strict leakage-clean of id_test, (2) bootstrap 95% CIs for PIJAR-ID.

- Removes any id_test row whose text appears in id_train_aug OR whose en_source appears in train
  (makes the cross-lingual set provably disjoint from training at the source level).
- Reports AUROC and TPR@1%FPR with percentile bootstrap 95% CIs on en_test, deepset, id_test
  (raw) and id_test_clean. Writes expD/results/ci.json and expD/data/id_test_clean.csv.
Run: python expD/src/gap_fixes.py
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, "common"); sys.path.insert(0, "expD/src")
from metrics import tpr_at_fpr, auroc, bootstrap_ci
import features as F
from tooloutput import wrap_frame
from sklearn.linear_model import LogisticRegression

DATA, RES = "expD/data", "expD/results"


def wrapped(name, seed):
    d = pd.read_csv(f"{DATA}/{name}.csv").dropna(subset=["text"])
    d = d[d["text"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    return wrap_frame(d, seed=seed)


def lc(name):
    d = pd.read_csv(f"{DATA}/{name}.csv").dropna(subset=["context"])
    return d["context"].tolist(), d["label"].to_numpy()


def ci_row(y, s, n_boot=2000):
    a, t = auroc(y, s), tpr_at_fpr(y, s, 0.01)[0]
    a_lo, a_hi = bootstrap_ci(auroc, y, s, n_boot=n_boot)
    t_lo, t_hi = bootstrap_ci(lambda yy, ss: tpr_at_fpr(yy, ss, 0.01)[0], y, s, n_boot=n_boot)
    return {"auroc": round(a, 3), "auroc_ci": [round(a_lo, 3), round(a_hi, 3)],
            "tpr1": round(t, 3), "tpr1_ci": [round(t_lo, 3), round(t_hi, 3)], "n": int(len(y))}


def main():
    # ---- build PIJAR-ID training set (train + id_train_aug + hardneg) ----
    tr = wrapped("train", 0); trx, ytr = tr["context"].tolist(), list(tr["label"])
    aug = wrapped("id_train_aug", 21); trx += aug["context"].tolist(); ytr += list(aug["label"])
    hn = wrapped("hardneg", 31); trx += hn["context"].tolist(); ytr += list(hn["label"])
    ytr = np.asarray(ytr)

    # ---- leakage-clean id_test ----
    train_text = set(pd.read_csv(f"{DATA}/train.csv")["text"].astype(str))
    aug_text = set(pd.read_csv(f"{DATA}/id_train_aug.csv")["text"].astype(str))
    idt = pd.read_csv(f"{DATA}/id_test.csv")
    leaked = idt["text"].astype(str).isin(aug_text) | idt["en_source"].astype(str).isin(train_text)
    clean = idt[~leaked].reset_index(drop=True)
    clean.to_csv(f"{DATA}/id_test_clean.csv", index=False)
    print(f"id_test: {len(idt)} -> clean {len(clean)} (removed {int(leaked.sum())} leaked rows)")

    # ---- fit once (seed 0) ----
    vec = F.tfidf_vectorizer(); X = vec.fit_transform(trx)
    clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced").fit(X, ytr)
    sc = lambda texts: clf.predict_proba(vec.transform(texts))[:, 1]

    out = {}
    for name in ["en_test", "deepset", "id_test"]:
        tx, y = lc(name); out[name] = ci_row(y, sc(tx))
    cy = clean["label"].to_numpy(); out["id_test_clean"] = ci_row(cy, sc(clean["context"].tolist()))

    json.dump(out, open(f"{RES}/ci.json", "w"), indent=2)
    print("\n== PIJAR-ID with 95% bootstrap CI ==")
    for k, v in out.items():
        print(f"  {k:14s} n={v['n']:4d}  AUROC {v['auroc']:.3f} CI{v['auroc_ci']}  "
              f"TPR@1% {v['tpr1']:.3f} CI{v['tpr1_ci']}")


if __name__ == "__main__":
    main()

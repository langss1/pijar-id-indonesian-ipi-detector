"""Export the trained PIJAR-ID model as a single joblib for the demo/serving backend.

Trains EXACTLY the published PIJAR-ID configuration:
  word TF-IDF (1-2 gram, sublinear, L2) + L2 logistic regression (C=4, class_weight=balanced)
  on  Wrap(train) ∪ Wrap(id_train_aug) ∪ Wrap(hardneg).
The decision threshold is calibrated to 1% FPR on Wrap(en_test) (τ@1%FPR), matching the paper.

Output: demo/pijar_id_model.joblib  (vectorizer + logreg + threshold + metadata)
Run from project root (needs expD/data/*.csv present): python expD/src/export_model.py
"""
import os, sys, json, joblib, numpy as np, pandas as pd
sys.path.insert(0, "common"); sys.path.insert(0, "expD/src")
import features as F
from tooloutput import wrap_frame
from metrics import tpr_at_fpr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

DATA = "expD/data"
OUT = "demo/pijar_id_model.joblib"
SEED = 0


def load_wrapped(name, seed):
    d = pd.read_csv(f"{DATA}/{name}.csv").dropna(subset=["text"])
    d = d[d["text"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    w = wrap_frame(d, seed=seed)
    return w["context"].tolist(), list(w["label"].to_numpy())


def thr_for_fpr(y, s, target=0.01):
    fpr, tpr, thr = roc_curve(y, s)
    ok = fpr <= target + 1e-12
    if not ok.any():
        return float("inf")
    i = int(np.argmax(tpr * ok))
    return float(thr[i]) if i < len(thr) else float("inf")


def main():
    os.makedirs("demo", exist_ok=True)
    # --- assemble PIJAR-ID training corpus: EN + MT-aug + hard-negatives ---
    tx, y = load_wrapped("train", SEED)
    ax, ay = load_wrapped("id_train_aug", 21); tx += ax; y += ay
    hx, hy = load_wrapped("hardneg", 31);      tx += hx; y += hy
    y = np.asarray(y)
    print(f"train corpus: n={len(tx)} pos={int((y==1).sum())} neg={int((y==0).sum())}")

    vec = F.tfidf_vectorizer()
    X = vec.fit_transform(tx)
    clf = LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced").fit(X, y)

    # --- calibrate threshold at 1% FPR on held-out en_test ---
    ex, ey = load_wrapped("en_test", SEED + 1)
    es = clf.predict_proba(vec.transform(ex))[:, 1]
    tau = thr_for_fpr(np.asarray(ey), es, 0.01)
    tpr1, _ = tpr_at_fpr(np.asarray(ey), es, 0.01)
    print(f"calibrated tau@1%FPR = {tau:.4f} (en_test TPR@1%FPR = {tpr1:.3f})")

    bundle = {
        "vectorizer": vec,
        "logreg": clf,
        "threshold": float(tau),
        "feature_names": vec.get_feature_names_out().tolist(),
        "meta": {
            "name": "PIJAR-ID",
            "config": "word TF-IDF(1-2gram,sublinear,L2) + LogReg(C=4,balanced)",
            "train_corpus": "Wrap(train) + Wrap(id_train_aug) + Wrap(hardneg)",
            "threshold_calibration": "1% FPR on Wrap(en_test)",
            "n_features": int(len(vec.get_feature_names_out())),
        },
    }
    joblib.dump(bundle, OUT, compress=3)
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"saved {OUT} ({size_mb:.2f} MB), n_features={bundle['meta']['n_features']}")
    json.dump(bundle["meta"], open("demo/model_meta.json", "w"), indent=2)


if __name__ == "__main__":
    main()

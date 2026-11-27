"""Light end-to-end CPU latency: TF-IDF+LR vs e5+LR (no heavy transformer).
Measures the full inference path per sample on a 200-row probe. Low RAM.
Run: python expD/src/latency.py
"""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, "common"); sys.path.insert(0, "expD/src")
import features as F
from sklearn.linear_model import LogisticRegression

RES = "expD/results"


def load(name):
    d = pd.read_csv(f"expD/data/{name}.csv").dropna(subset=["context"])
    return d["context"].tolist(), d["label"].to_numpy()


def main():
    tr, ytr = load("train")
    probe = load("en_test")[0][:200]
    out = {}

    vec = F.tfidf_vectorizer(); Xtr = vec.fit_transform(tr)
    lr = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(Xtr, ytr)
    t0 = time.perf_counter()
    for t in probe:
        lr.predict_proba(vec.transform([t]))
    out["tfidf_lr_e2e_ms"] = round((time.perf_counter() - t0) / len(probe) * 1000, 4)

    Etr = F.e5_embed(tr, prefix="query: ")
    lr2 = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced").fit(Etr, ytr)
    model = F._load_e5()
    t0 = time.perf_counter()
    for t in probe:
        e = model.encode(["query: " + t], convert_to_numpy=True, normalize_embeddings=True)
        lr2.predict_proba(e)
    out["e5_lr_e2e_ms"] = round((time.perf_counter() - t0) / len(probe) * 1000, 4)
    out["speedup_e5_over_tfidf"] = round(out["e5_lr_e2e_ms"] / out["tfidf_lr_e2e_ms"], 1)

    json.dump(out, open(f"{RES}/latency.json", "w"), indent=2)
    print("end-to-end latency ms/sample (CPU):", out)


if __name__ == "__main__":
    main()

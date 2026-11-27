"""Paper D — Iteration 1: FUSION detector (word+char TF-IDF union e5) + over-defense hard negatives.
Compares single feature spaces vs fusion, across training variants:
  base | +aug (id_train_aug) | +aug+hn (id_train_aug + hardneg)
Eval sets: en_test (in-domain), deepset (cross-benchmark EN), id_test (cross-lingual),
           notinject (over-defense, benign-only), gandalf (attack-only recall).
Metrics: TPR@1%FPR (5 seeds) + NotInject FPR at en_test-calibrated 1%FPR threshold + latency.
Run: python expD/src/fuse.py
"""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, "common"); sys.path.insert(0, "expD/src")
from metrics import tpr_at_fpr, auroc
import features as F
from tooloutput import wrap_frame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import roc_curve

RES = "expD/results"; os.makedirs(RES, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
EVAL = ["en_test", "deepset", "id_test", "notinject", "gandalf"]


def char_vec():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                           max_features=50000, sublinear_tf=True, lowercase=True)


def load_ctx(name):
    d = pd.read_csv(f"expD/data/{name}.csv").dropna(subset=["context"])
    return d["context"].tolist(), d["label"].to_numpy()


def wrap_raw(path, seed):
    d = pd.read_csv(path).dropna(subset=["text"])
    d = d[d["text"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    w = wrap_frame(d, seed=seed)
    return w["context"].tolist(), w["label"].to_numpy()


def build_train(variant):
    tx, y = load_ctx("train")
    tx, y = list(tx), list(y)
    if variant in ("aug", "aughn"):
        ax, ay = wrap_raw("expD/data/id_train_aug.csv", 21)
        tx += ax; y += list(ay)
    if variant == "aughn":
        hx, hy = wrap_raw("expD/data/hardneg.csv", 31)
        tx += hx; y += list(hy)
    return tx, np.array(y)


def thr_for_fpr(y, s, target=0.01):
    fpr, tpr, thr = roc_curve(y, s)
    ok = fpr <= target + 1e-12
    if not ok.any():
        return float("inf")
    idx = int(np.argmax(tpr * ok))
    return float(thr[idx]) if idx < len(thr) else float("inf")


def eval_scores(score_fn, sets_txt):
    """score_fn: texts->prob. Returns dict of per-set scores + timing on en_test."""
    out = {}
    for n in EVAL:
        t0 = time.perf_counter()
        s = score_fn(sets_txt[n][0])
        out[n] = s
        if n == "en_test":
            out["_lat_ms"] = (time.perf_counter() - t0) / max(1, len(s)) * 1000
    return out


def main():
    sets_txt = {n: load_ctx(n) for n in EVAL}
    variants = ["base", "aug", "aughn"]
    # detectors by feature recipe
    recipes = ["word", "char", "e5", "fuse_concat", "fuse_avg"]

    results = {}  # variant -> recipe -> metrics
    for variant in variants:
        tr_txt, ytr = build_train(variant)
        print(f"\n=== variant={variant} train n={len(tr_txt)} pos={int(ytr.sum())} ===")
        # precompute feature spaces
        wv = F.tfidf_vectorizer(); Xw_tr = wv.fit_transform(tr_txt)
        cv = char_vec(); Xc_tr = cv.fit_transform(tr_txt)
        Etr = F.e5_embed(tr_txt, prefix="query: ")
        Xw_set = {n: wv.transform(sets_txt[n][0]) for n in EVAL}
        Xc_set = {n: cv.transform(sets_txt[n][0]) for n in EVAL}
        Eset = {n: F.e5_embed(sets_txt[n][0], prefix="query: ") for n in EVAL}

        results[variant] = {r: {n: [] for n in ["en_test", "deepset", "id_test"]} |
                               {"notinject_fpr": [], "gandalf_rec": [], "_lat": []}
                            for r in recipes}

        for seed in SEEDS:
            def mk(): return LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced")
            # fit single-space models
            mw = mk().fit(Xw_tr, ytr)
            mc = mk().fit(Xc_tr, ytr)
            me = mk().fit(Etr, ytr)
            # fusion-concat model
            Xf_tr = hstack([Xw_tr, Xc_tr, csr_matrix(Etr)]).tocsr()
            mf = mk().fit(Xf_tr, ytr)

            def score_word(n): return mw.predict_proba(Xw_set[n])[:, 1]
            def score_char(n): return mc.predict_proba(Xc_set[n])[:, 1]
            def score_e5(n): return me.predict_proba(Eset[n])[:, 1]
            def score_fc(n): return mf.predict_proba(
                hstack([Xw_set[n], Xc_set[n], csr_matrix(Eset[n])]).tocsr())[:, 1]
            def score_fa(n): return (score_word(n) + score_char(n) + score_e5(n)) / 3.0
            scorers = {"word": score_word, "char": score_char, "e5": score_e5,
                       "fuse_concat": score_fc, "fuse_avg": score_fa}

            for r in recipes:
                sc = {n: scorers[r](n) for n in EVAL}
                cal_thr = thr_for_fpr(sets_txt["en_test"][1], sc["en_test"], 0.01)
                for n in ["en_test", "deepset", "id_test"]:
                    results[variant][r][n].append(tpr_at_fpr(sets_txt[n][1], sc[n], 0.01)[0])
                results[variant][r]["notinject_fpr"].append(float((sc["notinject"] >= cal_thr).mean()))
                results[variant][r]["gandalf_rec"].append(float((sc["gandalf"] >= cal_thr).mean()))
                if seed == 0:
                    t0 = time.perf_counter(); scorers[r]("en_test");
                    results[variant][r]["_lat"].append(
                        (time.perf_counter()-t0)/max(1,len(sets_txt["en_test"][1]))*1000)

        # aggregate + print
        for r in recipes:
            d = results[variant][r]
            agg = {k: (round(float(np.mean(v)), 3), round(float(np.std(v)), 3)) if v else None
                   for k, v in d.items()}
            results[variant][r] = agg
            print(f"  [{r:11s}] en={agg['en_test'][0]:.3f} deepset(EN-shift)={agg['deepset'][0]:.3f} "
                  f"id={agg['id_test'][0]:.3f} NI_fpr={agg['notinject_fpr'][0]:.3f} "
                  f"gandalf={agg['gandalf_rec'][0]:.3f} lat={agg['_lat'][0]:.2f}ms")

    json.dump(results, open(f"{RES}/fuse.json", "w"), indent=2)
    print(f"\nsaved -> {RES}/fuse.json")


if __name__ == "__main__":
    main()

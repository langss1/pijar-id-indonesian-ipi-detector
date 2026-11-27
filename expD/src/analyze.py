"""Paper D — compute the full metric table from saved scores.

Reads expD/results/scores.npz (+ multiseed.json), writes expD/results/summary.csv and
prints a markdown table. Key metrics:
  - TPR@1%FPR, TPR@0.1%FPR, AUROC on both-class sets (en_test, deepset, id_test).
  - Over-defense: FPR on notinject at the threshold calibrated to 1%FPR on en_test.
  - gandalf recall at the same en_test-calibrated threshold.
  - EN->ID drop (en_test vs id_test TPR@1%FPR).
Run: python expD/src/analyze.py
"""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, "common")
from metrics import tpr_at_fpr, auroc, bootstrap_ci

RES = "expD/results"
Z = np.load(f"{RES}/scores.npz", allow_pickle=True)
MS = json.load(open(f"{RES}/multiseed.json"))
ZA = np.load(f"{RES}/scores_aug.npz", allow_pickle=True) if os.path.exists(f"{RES}/scores_aug.npz") else None
DETECTORS = ["tfidf_lr", "e5_lr", "e5_mlp", "e5_hgb"]
# per-row source for id_test (aligned with scores order)
ID_SRC = None
if os.path.exists("expD/data/id_test.csv"):
    ID_SRC = pd.read_csv("expD/data/id_test.csv")["source"].to_numpy()


def g(d, s, suf, Zsrc=None):
    Zsrc = Zsrc if Zsrc is not None else Z
    k = f"{d}__{s}__{suf}"
    return Zsrc[k] if k in Zsrc.files else None


def threshold_for_fpr(y, score, target):
    """Smallest threshold achieving FPR<=target on this (calibration) set."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, score)
    ok = fpr <= target + 1e-12
    if not ok.any():
        return float("inf")
    idx = np.argmax(tpr * ok)
    return float(thr[idx]) if idx < len(thr) else float("inf")


rows = []
for d in DETECTORS:
    cal_y, cal_s = g(d, "en_test", "y"), g(d, "en_test", "score")
    cal_thr = threshold_for_fpr(cal_y, cal_s, 0.01)
    rec = {"detector": d}
    # both-class sets
    for s in ["en_test", "deepset", "id_test"]:
        y, sc = g(d, s, "y"), g(d, s, "score")
        if y is None:
            continue
        ms = MS.get(d, {}).get(s, [])
        rec[f"{s}_tpr1"] = float(np.mean(ms)) if ms else tpr_at_fpr(y, sc, 0.01)[0]
        rec[f"{s}_tpr1_sd"] = float(np.std(ms)) if ms else 0.0
        rec[f"{s}_tpr01"] = tpr_at_fpr(y, sc, 0.001)[0]
        rec[f"{s}_auroc"] = auroc(y, sc)
        lo, hi = bootstrap_ci(lambda yt, ss: tpr_at_fpr(yt, ss, 0.01)[0], y, sc, n_boot=1000)
        rec[f"{s}_tpr1_ci"] = f"[{lo:.2f},{hi:.2f}]"
    # over-defense on notinject (benign-only): FPR at calibrated threshold
    ni_y, ni_s = g(d, "notinject", "y"), g(d, "notinject", "score")
    if ni_s is not None:
        rec["notinject_fpr"] = float((ni_s >= cal_thr).mean())
    # gandalf recall at calibrated threshold (attack-only)
    gd_y, gd_s = g(d, "gandalf", "y"), g(d, "gandalf", "score")
    if gd_s is not None:
        rec["gandalf_recall@cal"] = float((gd_s >= cal_thr).mean())
    # latency
    lat = g(d, "en_test", "lat")
    rec["lat_ms"] = float(lat[0]) if lat is not None else float("nan")
    # EN->ID drop
    if "id_test_tpr1" in rec and "en_test_tpr1" in rec:
        rec["en_minus_id_tpr1"] = rec["en_test_tpr1"] - rec["id_test_tpr1"]
    # id_test split by condition (translate vs codemix)
    y, sc = g(d, "id_test", "y"), g(d, "id_test", "score")
    if y is not None and ID_SRC is not None and len(ID_SRC) == len(y):
        for cond in ["id_translate", "id_codemix"]:
            m = ID_SRC == cond
            if m.sum() > 0 and len(np.unique(y[m])) == 2:
                rec[f"{cond}_tpr1"] = tpr_at_fpr(y[m], sc[m], 0.01)[0]
                rec[f"{cond}_auroc"] = auroc(y[m], sc[m])
    # RQ3: +ID augmentation effect on id_test
    if ZA is not None:
        ya, sca = g(d, "id_test", "y", ZA), g(d, "id_test", "score", ZA)
        if ya is not None:
            rec["id_test_tpr1_aug"] = tpr_at_fpr(ya, sca, 0.01)[0]
            if "id_test_tpr1" in rec:
                rec["aug_gain_tpr1"] = rec["id_test_tpr1_aug"] - rec["id_test_tpr1"]
    rows.append(rec)

df = pd.DataFrame(rows)
os.makedirs(RES, exist_ok=True)
df.to_csv(f"{RES}/summary.csv", index=False)

# markdown table of the headline numbers
cols = ["detector", "en_test_tpr1", "deepset_tpr1", "id_test_tpr1", "en_minus_id_tpr1",
        "notinject_fpr", "lat_ms"]
have = [c for c in cols if c in df.columns]
print("\n### Paper D — headline (TPR@1%FPR unless noted)\n")
print("| " + " | ".join(have) + " |")
print("|" + "|".join(["---"] * len(have)) + "|")
for _, r in df.iterrows():
    cells = []
    for c in have:
        v = r[c]
        cells.append(f"{v:.3f}" if isinstance(v, (int, float, np.floating)) else str(v))
    print("| " + " | ".join(cells) + " |")
print(f"\nfull table -> {RES}/summary.csv")

# RQ2 / RQ3 focused views
print("\n### RQ2 — EN vs ID (TPR@1%FPR)\n")
print("| detector | en_test | id_translate | id_codemix | EN−ID |")
print("|---|---|---|---|---|")
for _, r in df.iterrows():
    def v(c): return f"{r[c]:.3f}" if c in r and isinstance(r[c], (int, float, np.floating)) else "-"
    print(f"| {r['detector']} | {v('en_test_tpr1')} | {v('id_translate_tpr1')} | {v('id_codemix_tpr1')} | {v('en_minus_id_tpr1')} |")

if "id_test_tpr1_aug" in df.columns:
    print("\n### RQ3 — adding a small ID training slice (TPR@1%FPR on id_test)\n")
    print("| detector | id_test (EN-only train) | id_test (+ID aug) | gain |")
    print("|---|---|---|---|")
    for _, r in df.iterrows():
        def v(c): return f"{r[c]:.3f}" if c in r and isinstance(r[c], (int, float, np.floating)) else "-"
        print(f"| {r['detector']} | {v('id_test_tpr1')} | {v('id_test_tpr1_aug')} | {v('aug_gain_tpr1')} |")

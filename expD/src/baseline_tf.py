"""Paper D — transformer GUARD baseline on CPU (inference only), for a PIDS-Bench-style head-to-head
against our cheap word TF-IDF. Scores a published prompt-injection classifier on the same eval sets.
Tries a list of HF models and uses the first that loads. Metrics: TPR@1%FPR, AUROC, NotInject FPR
at en_test-calibrated threshold, gandalf recall, CPU latency.
Run: python expD/src/baseline_tf.py
"""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, "common")
from metrics import tpr_at_fpr, auroc
from sklearn.metrics import roc_curve

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
RES = "expD/results"
EVAL = ["en_test", "deepset", "id_test", "notinject", "gandalf"]
MODELS = [
    "protectai/deberta-v3-base-prompt-injection-v2",
    "deepset/deberta-v3-base-injection",
    "fmops/distilbert-prompt-injection",
]


def lc(n):
    d = pd.read_csv(f"expD/data/{n}.csv").dropna(subset=["context"])
    return d["context"].tolist(), d["label"].to_numpy()


def thr_for_fpr(y, s, target=0.01):
    fpr, tpr, thr = roc_curve(y, s); ok = fpr <= target + 1e-12
    if not ok.any():
        return float("inf")
    i = int(np.argmax(tpr * ok)); return float(thr[i]) if i < len(thr) else float("inf")


def load_pipe():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    for name in MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            mdl = AutoModelForSequenceClassification.from_pretrained(name)
            mdl.eval()
            id2label = {int(k): v.upper() for k, v in mdl.config.id2label.items()}
            # find the index of the "injection/unsafe" class
            pos = None
            for i, lab in id2label.items():
                if any(k in lab for k in ["INJECT", "UNSAFE", "JAILBREAK", "LABEL_1", "TRUE", "1"]):
                    pos = i
            if pos is None:
                pos = 1
            print(f"[loaded] {name} | labels={id2label} | pos_idx={pos}")
            return name, tok, mdl, pos, torch
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:120]}")
    return None


def score(texts, tok, mdl, pos, torch, bs=16, maxlen=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            enc = tok(batch, truncation=True, max_length=maxlen, padding=True, return_tensors="pt")
            logits = mdl(**enc).logits
            p = torch.softmax(logits, dim=-1)[:, pos].cpu().numpy()
            out.append(p)
    return np.concatenate(out)


def main():
    loaded = load_pipe()
    if loaded is None:
        print("NO MODEL LOADED (offline or download blocked). Falling back to published numbers.")
        json.dump({"status": "no_model"}, open(f"{RES}/baseline_tf.json", "w"))
        return
    name, tok, mdl, pos, torch = loaded
    sets = {n: lc(n) for n in EVAL}
    sc = {}
    for n in EVAL:
        t0 = time.perf_counter()
        sc[n] = score(sets[n][0], tok, mdl, pos, torch)
        dt = (time.perf_counter() - t0) / max(1, len(sc[n])) * 1000
        if n == "en_test":
            lat = dt
    ct = thr_for_fpr(sets["en_test"][1], sc["en_test"], 0.01)
    res = {"model": name, "lat_ms": round(lat, 2)}
    for n in ["en_test", "deepset", "id_test"]:
        res[f"{n}_tpr1"] = round(tpr_at_fpr(sets[n][1], sc[n], 0.01)[0], 3)
        res[f"{n}_tpr5"] = round(tpr_at_fpr(sets[n][1], sc[n], 0.05)[0], 3)
        res[f"{n}_auroc"] = round(auroc(sets[n][1], sc[n]), 3)
    res["notinject_fpr"] = round(float((sc["notinject"] >= ct).mean()), 3)
    res["gandalf_recall"] = round(float((sc["gandalf"] >= ct).mean()), 3)
    json.dump(res, open(f"{RES}/baseline_tf.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

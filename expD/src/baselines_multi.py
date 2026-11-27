"""Gap 2: evaluate ADDITIONAL transformer guard baselines (beyond DeBERTa-v3) on CPU, zero-shot.
Same protocol as baseline_tf.py but loops over several published guards and saves each.
Output: expD/results/baselines_multi.json
Run: python expD/src/baselines_multi.py
"""
import os, sys, json, time, numpy as np, pandas as pd
sys.path.insert(0, "common")
from metrics import tpr_at_fpr, auroc
from sklearn.metrics import roc_curve

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
RES = "expD/results"
EVAL = ["en_test", "deepset", "id_test", "notinject", "gandalf"]
MODELS = [
    "fmops/distilbert-prompt-injection",
    "deepset/deberta-v3-base-injection",
    "leolee99/InjecGuard",
]


def lc(n):
    d = pd.read_csv(f"expD/data/{n}.csv").dropna(subset=["context"])
    return d["context"].tolist(), d["label"].to_numpy()


def thr_for_fpr(y, s, target=0.01):
    fpr, tpr, thr = roc_curve(y, s); ok = fpr <= target + 1e-12
    if not ok.any():
        return float("inf")
    i = int(np.argmax(tpr * ok)); return float(thr[i]) if i < len(thr) else float("inf")


def load(name, torch):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name); mdl.eval()
    id2 = {int(k): v.upper() for k, v in mdl.config.id2label.items()}
    pos = next((i for i, l in id2.items()
                if any(k in l for k in ["INJECT", "UNSAFE", "JAILBREAK", "LABEL_1", "TRUE", "1"])), 1)
    return tok, mdl, pos, id2


def score(texts, tok, mdl, pos, torch, bs=16, maxlen=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], truncation=True, max_length=maxlen, padding=True, return_tensors="pt")
            out.append(torch.softmax(mdl(**enc).logits, dim=-1)[:, pos].cpu().numpy())
    return np.concatenate(out)


def main():
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    sets = {n: lc(n) for n in EVAL}
    results = {}
    for name in MODELS:
        try:
            tok, mdl, pos, id2 = load(name, torch)
            print(f"[loaded] {name} labels={id2} pos={pos}")
        except Exception as e:
            print(f"[skip] {name}: {type(e).__name__}: {str(e)[:120]}"); continue
        sc, lat = {}, None
        for n in EVAL:
            t0 = time.perf_counter(); sc[n] = score(sets[n][0], tok, mdl, pos, torch)
            if n == "en_test":
                lat = (time.perf_counter()-t0)/max(1, len(sc[n]))*1000
        ct = thr_for_fpr(sets["en_test"][1], sc["en_test"], 0.01)
        r = {"lat_ms": round(lat, 2)}
        for n in ["en_test", "deepset", "id_test"]:
            r[f"{n}_tpr1"] = round(tpr_at_fpr(sets[n][1], sc[n], 0.01)[0], 3)
            r[f"{n}_auroc"] = round(auroc(sets[n][1], sc[n]), 3)
        r["notinject_fpr"] = round(float((sc["notinject"] >= ct).mean()), 3)
        r["gandalf_recall"] = round(float((sc["gandalf"] >= ct).mean()), 3)
        results[name] = r
        print(f"  -> {json.dumps(r)}")
    json.dump(results, open(f"{RES}/baselines_multi.json", "w"), indent=2)
    print(f"saved {RES}/baselines_multi.json ({len(results)} models)")


if __name__ == "__main__":
    main()

"""Shared metrics: TPR at fixed FPR, AUROC, bootstrap CI, calibration (ECE)."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, f1_score


def tpr_at_fpr(y_true, scores, fpr_target):
    """Highest TPR achievable while FPR <= fpr_target. Returns (tpr, threshold)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    fpr, tpr, thr = roc_curve(y_true, scores)
    ok = fpr <= fpr_target + 1e-12
    if not ok.any():
        return 0.0, float("inf")
    idx = np.argmax(tpr * ok)  # best tpr among allowed fpr points
    return float(tpr[idx]), float(thr[idx]) if idx < len(thr) else float("inf")


def auroc(y_true, scores):
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return float("nan")


def fpr_at_threshold(y_true, scores, threshold):
    y_true = np.asarray(y_true)
    pred = (np.asarray(scores) >= threshold).astype(int)
    neg = y_true == 0
    if neg.sum() == 0:
        return float("nan")
    return float((pred[neg] == 1).mean())


def f1_at_threshold(y_true, scores, threshold):
    pred = (np.asarray(scores) >= threshold).astype(int)
    return float(f1_score(y_true, pred, zero_division=0))


def ece(y_true, prob, n_bins=10):
    """Expected calibration error for probabilistic scores in [0,1]."""
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(y_true)
    for i in range(n_bins):
        m = (prob >= bins[i]) & (prob < bins[i + 1] if i < n_bins - 1 else prob <= bins[i + 1])
        if m.sum() == 0:
            continue
        conf = prob[m].mean()
        acc = y_true[m].mean()
        e += (m.sum() / n) * abs(conf - acc)
    return float(e)


def bootstrap_ci(metric_fn, y_true, scores, n_boot=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a metric_fn(y_true, scores)->float."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], scores[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi


def summarize(y_true, scores, prob=None):
    """One-call summary dict used by all detectors."""
    t1, th1 = tpr_at_fpr(y_true, scores, 0.01)
    t01, th01 = tpr_at_fpr(y_true, scores, 0.001)
    out = {
        "auroc": auroc(y_true, scores),
        "tpr@1fpr": t1,
        "tpr@0.1fpr": t01,
        "n": int(len(y_true)),
        "n_pos": int(np.sum(np.asarray(y_true) == 1)),
    }
    if prob is not None:
        out["ece"] = ece(y_true, prob)
    return out
